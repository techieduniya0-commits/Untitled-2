import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { Type, ThinkingLevel } from '@google/genai';
import type {
  AudioAssessmentCategory,
  AudioProfile,
  AudioSegmentFinding,
  AudioSupportingSignals,
  DetectorResult,
  EvidencePolarity,
  ExtractedMetadata,
  PrimaryAudioDetectorSummary,
  WatermarkFinding,
} from '../../src/types/forensics.ts';
import {
  GEMINI_FORENSIC_MODEL,
  type GeminiRetryEvent,
  runStructuredGeminiAnalysisWithMeta,
} from '../geminiClient.ts';
import type { ProvenanceInspectionOutput } from '../provenanceVerification/index.ts';
import {
  classifyAudioSegmentsWithPretrainedModel,
  getConfiguredAudioClassifier,
} from './classifier.ts';

const execFileAsync = promisify(execFile);

const TARGET_SAMPLE_RATE = 16000;
const MAX_DECODE_SECONDS = 120;
const SEGMENT_WINDOW_SEC = 4.0;
const SEGMENT_HOP_SEC = 2.0;
const MAX_SEGMENTS_EVALUATED = 12;

// Thresholds for mapping raw uncalibrated model score (0..1) into explicit PDF Section 2 categories
const SYNTHETIC_SCORE_THRESHOLD = 0.62;
const GENUINE_SCORE_THRESHOLD = 0.38;

interface GeminiAudioExplanationResponse {
  explanationSummary: string;
  keyObservations: string[];
  caveats: string[];
}

const AUDIO_EXPLANATION_SCHEMA = {
  type: Type.OBJECT,
  properties: {
    explanationSummary: { type: Type.STRING },
    keyObservations: { type: Type.ARRAY, items: { type: Type.STRING } },
    caveats: { type: Type.ARRAY, items: { type: Type.STRING } },
  },
  required: ['explanationSummary', 'keyObservations', 'caveats'],
};

function formatSec(sec: number): string {
  const safe = Math.max(0, sec);
  const m = Math.floor(safe / 60);
  const s = safe - m * 60;
  return `${String(m).padStart(2, '0')}:${s.toFixed(1).padStart(4, '0')}`;
}

async function decodeAudioTo16kMonoPcm(filePath: string): Promise<{
  samples: Float32Array;
  durationSec: number;
  truncatedAtMaxDuration: boolean;
}> {
  const { stdout } = await execFileAsync(
    'ffmpeg',
    [
      '-v',
      'error',
      '-i',
      filePath,
      '-vn',
      '-ac',
      '1',
      '-ar',
      String(TARGET_SAMPLE_RATE),
      '-f',
      's16le',
      '-t',
      String(MAX_DECODE_SECONDS),
      'pipe:1',
    ],
    {
      encoding: 'buffer',
      maxBuffer: 20 * 1024 * 1024,
      timeout: 30_000,
    },
  );

  const buf = stdout as unknown as Buffer;
  const sampleCount = Math.floor(buf.length / 2);
  const samples = new Float32Array(sampleCount);
  for (let i = 0; i < sampleCount; i++) {
    samples[i] = buf.readInt16LE(i * 2) / 32768;
  }
  const durationSec = sampleCount / TARGET_SAMPLE_RATE;
  return {
    samples,
    durationSec,
    truncatedAtMaxDuration: durationSec >= MAX_DECODE_SECONDS - 0.05,
  };
}

interface SpectralMetrics {
  spectralCentroidHz: number | null;
  highFreqEnergyRatio: number | null;
  spectralFlatness: number | null;
  bandEnergyRatios: {
    sub300Hz: number;
    speechCore300To3400Hz: number;
    upperHarmonics3400To6000Hz: number;
    highBand6000To8000Hz: number;
  };
}

/**
 * Computes windowed DFT spectral energy across 4 perceptually meaningful speech/mel-aligned bands:
 * sub-300 Hz, speech core (300-3400 Hz), upper harmonics (3400-6000 Hz), and high band (6000-8000 Hz).
 */
function computeSpectralMetrics(samples: Float32Array, sampleRate: number): SpectralMetrics {
  const N = 512;
  if (samples.length < N) {
    return {
      spectralCentroidHz: null,
      highFreqEnergyRatio: null,
      spectralFlatness: null,
      bandEnergyRatios: {
        sub300Hz: 0,
        speechCore300To3400Hz: 0,
        upperHarmonics3400To6000Hz: 0,
        highBand6000To8000Hz: 0,
      },
    };
  }

  const windowCount = Math.min(8, Math.max(1, Math.floor(samples.length / N)));
  const half = N / 2;
  const binHz = sampleRate / N;
  const avgMag = new Float64Array(half);
  let validWindows = 0;

  for (let w = 0; w < windowCount; w++) {
    const start = Math.floor((w / windowCount) * (samples.length - N));
    let energy = 0;
    for (let n = 0; n < N; n++) energy += samples[start + n] * samples[start + n];
    if (energy < 1e-6) continue; // skip near-silent windows
    validWindows++;

    for (let k = 1; k < half; k++) {
      let re = 0;
      let im = 0;
      const angleStep = (-2 * Math.PI * k) / N;
      for (let n = 0; n < N; n++) {
        // Hann window reduces spectral leakage
        const hann = 0.5 * (1 - Math.cos((2 * Math.PI * n) / (N - 1)));
        const x = samples[start + n] * hann;
        const a = angleStep * n;
        re += x * Math.cos(a);
        im += x * Math.sin(a);
      }
      avgMag[k] += Math.sqrt(re * re + im * im);
    }
  }

  if (validWindows === 0) {
    return {
      spectralCentroidHz: null,
      highFreqEnergyRatio: null,
      spectralFlatness: null,
      bandEnergyRatios: {
        sub300Hz: 0,
        speechCore300To3400Hz: 0,
        upperHarmonics3400To6000Hz: 0,
        highBand6000To8000Hz: 0,
      },
    };
  }

  let magSum = 0;
  let weightedFreqSum = 0;
  let highFreqMag = 0;
  let sub300 = 0;
  let speechCore = 0;
  let upperHarm = 0;
  let highBand = 0;
  let logMagSum = 0;
  let activeBins = 0;

  for (let k = 1; k < half; k++) {
    const mag = avgMag[k] / validWindows;
    const freq = k * binHz;
    magSum += mag;
    weightedFreqSum += mag * freq;

    if (freq < 300) sub300 += mag;
    else if (freq < 3400) speechCore += mag;
    else if (freq < 6000) upperHarm += mag;
    else highBand += mag;

    if (freq >= 4000) highFreqMag += mag;
    if (mag > 1e-8) {
      logMagSum += Math.log(mag);
      activeBins++;
    }
  }

  const centroid = magSum > 0 ? Math.round(weightedFreqSum / magSum) : null;
  const highRatio = magSum > 0 ? Number((highFreqMag / magSum).toFixed(4)) : null;
  const geomMean = activeBins > 0 ? Math.exp(logMagSum / activeBins) : 0;
  const arithMean = activeBins > 0 ? magSum / activeBins : 0;
  const flatness = arithMean > 0 ? Number(Math.min(1, geomMean / arithMean).toFixed(4)) : null;

  return {
    spectralCentroidHz: centroid,
    highFreqEnergyRatio: highRatio,
    spectralFlatness: flatness,
    bandEnergyRatios: {
      sub300Hz: magSum > 0 ? Number((sub300 / magSum).toFixed(3)) : 0,
      speechCore300To3400Hz: magSum > 0 ? Number((speechCore / magSum).toFixed(3)) : 0,
      upperHarmonics3400To6000Hz: magSum > 0 ? Number((upperHarm / magSum).toFixed(3)) : 0,
      highBand6000To8000Hz: magSum > 0 ? Number((highBand / magSum).toFixed(3)) : 0,
    },
  };
}

/**
 * Estimates fundamental pitch (F0 in Hz, human speech range 75-380 Hz) via normalized autocorrelation.
 * Strictly used as a supporting acoustic descriptor, NEVER as a standalone synthetic-speech rule.
 */
function estimatePitchHz(samples: Float32Array, sampleRate: number): number | null {
  const frameSize = 800; // 50 ms at 16 kHz
  if (samples.length < frameSize) return null;

  const minLag = Math.floor(sampleRate / 380); // ~42 samples
  const maxLag = Math.min(Math.floor(sampleRate / 75), Math.floor(frameSize / 2)); // ~213 samples

  const stride = Math.max(frameSize, Math.floor((samples.length - frameSize) / 10));
  const pitches: number[] = [];

  for (let start = 0; start + frameSize <= samples.length && pitches.length < 12; start += stride) {
    let r0 = 0;
    for (let i = 0; i < frameSize; i++) {
      const s = samples[start + i];
      r0 += s * s;
    }
    const rms = Math.sqrt(r0 / frameSize);
    if (rms < 0.012 || r0 <= 1e-7) continue;

    let bestLag = -1;
    let bestCorr = 0;
    for (let lag = minLag; lag <= maxLag; lag++) {
      let sum = 0;
      const limit = frameSize - lag;
      for (let i = 0; i < limit; i++) {
        sum += samples[start + i] * samples[start + i + lag];
      }
      const norm = sum / r0;
      if (norm > bestCorr) {
        bestCorr = norm;
        bestLag = lag;
      }
    }

    // Require clear periodicity (>0.45 normalized autocorrelation) to count as voiced speech frame
    if (bestLag > 0 && bestCorr >= 0.45) {
      pitches.push(sampleRate / bestLag);
    }
  }

  if (pitches.length === 0) return null;
  pitches.sort((a, b) => a - b);
  return Math.round(pitches[Math.floor(pitches.length / 2)]);
}

function computeSegmentAcousticStats(samples: Float32Array, sampleRate: number): {
  rmsDb: number;
  zeroCrossingRate: number;
  isSpeechActive: boolean;
  spectralCentroidHz: number | null;
  pitchEstimateHz: number | null;
} {
  let sumSq = 0;
  let zc = 0;
  for (let i = 0; i < samples.length; i++) {
    const s = samples[i];
    sumSq += s * s;
    if (i > 0 && ((samples[i - 1] >= 0 && s < 0) || (samples[i - 1] < 0 && s >= 0))) {
      zc++;
    }
  }
  const rms = samples.length > 0 ? Math.sqrt(sumSq / samples.length) : 0;
  const rmsDb = Number((20 * Math.log10(Math.max(1e-6, rms))).toFixed(1));
  const zcr = Number((zc / Math.max(1, samples.length)).toFixed(4));
  const spec = computeSpectralMetrics(samples, sampleRate);
  const pitch = estimatePitchHz(samples, sampleRate);

  // Speech-active check: energy above -44 dB and not pure DC/ultra-high-frequency hiss
  const isSpeechActive = rmsDb > -44 && zcr >= 0.004 && zcr <= 0.48;

  return {
    rmsDb,
    zeroCrossingRate: zcr,
    isSpeechActive,
    spectralCentroidHz: spec.spectralCentroidHz,
    pitchEstimateHz: pitch,
  };
}

function buildOverlappingSegments(samples: Float32Array, sampleRate: number): Array<{
  index: number;
  startSec: number;
  endSec: number;
  samples: Float32Array;
}> {
  const totalSec = samples.length / sampleRate;
  if (samples.length === 0) return [];

  const winSamples = Math.round(SEGMENT_WINDOW_SEC * sampleRate);
  const hopSamples = Math.round(SEGMENT_HOP_SEC * sampleRate);

  if (samples.length <= winSamples) {
    return [
      {
        index: 0,
        startSec: 0,
        endSec: Number(totalSec.toFixed(2)),
        samples,
      },
    ];
  }

  const starts: number[] = [];
  for (let pos = 0; pos + winSamples <= samples.length; pos += hopSamples) {
    starts.push(pos);
  }
  const lastStart = samples.length - winSamples;
  if (starts.length === 0 || lastStart - starts[starts.length - 1] >= Math.round(0.75 * sampleRate)) {
    starts.push(lastStart);
  }

  // Uniformly downsample overlapping segment windows if recording exceeds MAX_SEGMENTS_EVALUATED
  let selectedStarts = starts;
  if (starts.length > MAX_SEGMENTS_EVALUATED) {
    selectedStarts = [];
    for (let i = 0; i < MAX_SEGMENTS_EVALUATED; i++) {
      const idx = Math.round((i * (starts.length - 1)) / (MAX_SEGMENTS_EVALUATED - 1));
      selectedStarts.push(starts[idx]);
    }
  }

  return selectedStarts.map((startSample, idx) => {
    const endSample = Math.min(samples.length, startSample + winSamples);
    return {
      index: idx,
      startSec: Number((startSample / sampleRate).toFixed(2)),
      endSec: Number((endSample / sampleRate).toFixed(2)),
      samples: samples.subarray(startSample, endSample),
    };
  });
}

/**
 * Documented segment aggregation formula (PDF Section 4):
 * Filters to speech-active segments, then combines mean raw score (60%) with top-2 upper-tail mean (40%)
 * so partial synthetic splices inside longer recordings are not masked while single transient spikes do not dominate.
 */
function aggregateRawSegmentScores(activeScores: number[]): {
  rawAggregatedScore: number;
  aggregationMethod: string;
} {
  const sortedDesc = [...activeScores].sort((a, b) => b - a);
  const meanAll = activeScores.reduce((acc, v) => acc + v, 0) / activeScores.length;
  const topKCount = Math.min(2, sortedDesc.length);
  const topKMean = sortedDesc.slice(0, topKCount).reduce((acc, v) => acc + v, 0) / topKCount;
  const combined = activeScores.length === 1 ? meanAll : 0.6 * meanAll + 0.4 * topKMean;

  return {
    rawAggregatedScore: Number(Math.max(0, Math.min(1, combined)).toFixed(4)),
    aggregationMethod:
      activeScores.length === 1
        ? 'Single speech-active segment raw model output (no multi-segment averaging needed).'
        : `Speech-active overlapping segment aggregation: 0.60 × mean(active segments = ${meanAll.toFixed(3)}) + 0.40 × top-${topKCount} mean (${topKMean.toFixed(3)}) across ${activeScores.length} speech-active segment(s).`,
  };
}

function mapRawScoreToAssessmentCategory(options: {
  classifierFailed: boolean;
  durationSec: number;
  speechActiveCount: number;
  rawAggregatedScore: number | null;
}): AudioAssessmentCategory {
  if (options.classifierFailed) {
    return 'Analysis unavailable because of a technical failure';
  }
  if (options.durationSec < 1.0 || options.speechActiveCount === 0 || options.rawAggregatedScore === null) {
    return 'Inconclusive / insufficient evidence';
  }
  if (options.rawAggregatedScore > SYNTHETIC_SCORE_THRESHOLD) {
    return 'Likely synthetic or spoofed speech';
  }
  if (options.rawAggregatedScore < GENUINE_SCORE_THRESHOLD) {
    return 'Likely genuine human speech';
  }
  return 'Inconclusive / insufficient evidence';
}

function classifyPolarityFromCategory(category: AudioAssessmentCategory): EvidencePolarity {
  if (category === 'Likely synthetic or spoofed speech') return 'ai_indicator';
  if (category === 'Likely genuine human speech') return 'natural_capture';
  return 'neutral';
}

export interface AudioPipelineOutput {
  detectors: DetectorResult[];
  watermarks: WatermarkFinding[];
  audioProfile: AudioProfile;
  geminiStatus: 'completed' | 'partially_completed' | 'failed' | 'unavailable';
  geminiModelUsed: string | null;
  geminiFailureDetail: string | null;
}

export async function runAudioDetectionPipeline(options: {
  filePath: string;
  metadata: ExtractedMetadata;
  provenance: ProvenanceInspectionOutput;
  sourceTrack?: 'standalone_audio' | 'extracted_video_audio';
  skipGeminiExplanation?: boolean;
  onStageProgress?: (
    stageId: 'media_preprocessing' | 'forensic_analysis' | 'ai_assessment',
    detail: string,
    statusOverride?: 'running' | 'retrying',
  ) => void;
}): Promise<AudioPipelineOutput> {
  const {
    filePath,
    metadata,
    provenance,
    sourceTrack = 'standalone_audio',
    skipGeminiExplanation = false,
    onStageProgress,
  } = options;
  const tSignalStart = Date.now();
  const trackPrefix = sourceTrack === 'extracted_video_audio' ? '[Video Audio Track] ' : '';

  onStageProgress?.(
    'media_preprocessing',
    `${trackPrefix}Decoding audio to 16 kHz mono PCM via FFmpeg and slicing overlapping ${SEGMENT_WINDOW_SEC}s segments (${SEGMENT_HOP_SEC}s hop).`,
  );

  let decoded: { samples: Float32Array; durationSec: number; truncatedAtMaxDuration: boolean };
  try {
    decoded = await decodeAudioTo16kMonoPcm(filePath);
  } catch (err) {
    const errMsg = err instanceof Error ? err.message : 'FFmpeg failed to decode audio stream.';
    const configured = getConfiguredAudioClassifier();
    const failedPrimary: PrimaryAudioDetectorSummary = {
      ran: false,
      status: 'technical_failure',
      backend: configured.backend,
      modelId: configured.modelId,
      modelBase: configured.modelBase,
      license: configured.license,
      inputSpec: configured.inputSpec,
      rawAggregatedScore: null,
      scoreLabel: 'Unavailable (audio decode failure)',
      segmentWindowSec: SEGMENT_WINDOW_SEC,
      segmentHopSec: SEGMENT_HOP_SEC,
      segmentsEvaluated: 0,
      speechActiveSegments: 0,
      aggregationMethod: 'Not executed due to decode failure.',
      inferenceDurationMs: 0,
      errorDetail: errMsg,
    };

    const failDetector: DetectorResult = {
      id: sourceTrack === 'extracted_video_audio' ? 'video_audio_deepfake_classifier' : 'pretrained_audio_deepfake_classifier',
      methodName: `${trackPrefix}Pretrained Audio Deepfake Classifier (Wav2Vec2)`,
      category: 'signal',
      status: 'failed',
      ran: false,
      resultSummary: `Analysis unavailable because of a technical failure — FFmpeg could not decode audio to 16 kHz mono PCM: ${errMsg}`,
      rawSignalScore: null,
      signalPolarity: 'neutral',
      reliability: 'High',
      reliabilityWeight: 0.85,
      calibrationStatus: 'Uncalibrated (Experimental)',
      applicability: '16 kHz mono speech anti-spoofing classification across overlapping segments',
      evidenceStrength: 'Low',
      evidence: [`Technical failure during FFmpeg audio decoding: ${errMsg}`],
      limitations: [
        'Technical failures are NEVER converted into synthetic-speech or genuine-speech classifications.',
      ],
      toolUsed: configured.modelId,
      correlationGroup: 'pretrained_audio_antispoofing',
      providerGroup: 'trained_audio_model',
      durationMs: Date.now() - tSignalStart,
      detectorVersion: '4.0',
    };

    return {
      detectors: [failDetector],
      watermarks: provenance.provenanceWatermarks,
      audioProfile: {
        sourceTrack,
        assessmentCategory: 'Analysis unavailable because of a technical failure',
        primaryDetector: failedPrimary,
        processingLimitations: [
          `FFmpeg failed to decode audio stream: ${errMsg}`,
          'No classification was generated from this technical failure.',
        ],
        waveformPeaks: [],
        spectralCentroidHz: null,
        highFreqEnergyRatio: null,
        silenceRatio: 1,
        dynamicRangeDb: 0,
        clippingRate: 0,
        noiseFloorDb: -96,
        spectralFlatness: null,
        speechDetected: false,
        syntheticIndicatorCount: 0,
        detectedIndicators: [],
        uncertainSignals: [],
        naturalCues: [],
        segments: [],
      },
      geminiStatus: 'unavailable',
      geminiModelUsed: null,
      geminiFailureDetail: null,
    };
  }

  const { samples, durationSec, truncatedAtMaxDuration } = decoded;

  // Waveform visualization peaks (64 bins)
  const binCount = 64;
  const binSize = Math.max(1, Math.floor(samples.length / binCount));
  const waveformPeaks: number[] = [];
  for (let b = 0; b < binCount; b++) {
    const start = b * binSize;
    const end = Math.min(samples.length, start + binSize);
    let maxAbs = 0;
    for (let i = start; i < end; i++) {
      const v = Math.abs(samples[i]);
      if (v > maxAbs) maxAbs = v;
    }
    waveformPeaks.push(Number(maxAbs.toFixed(3)));
  }

  // Global acoustic supporting measurements (RMS frames, silence/pauses, clipping, noise floor)
  const frameLen = 800; // 50ms at 16 kHz
  const frameCount = Math.max(1, Math.floor(samples.length / frameLen));
  let silentFrames = 0;
  let clippedSamples = 0;
  let maxRms = 0;
  let pauseCount = 0;
  let inPause = false;
  let consecutiveSilentFrames = 0;
  const frameRmsDbs: number[] = [];

  for (let f = 0; f < frameCount; f++) {
    const start = f * frameLen;
    const end = Math.min(samples.length, start + frameLen);
    let sq = 0;
    for (let i = start; i < end; i++) {
      const s = samples[i];
      sq += s * s;
      if (Math.abs(s) >= 0.98) clippedSamples++;
    }
    const rms = Math.sqrt(sq / Math.max(1, end - start));
    if (rms > maxRms) maxRms = rms;
    const db = 20 * Math.log10(Math.max(1e-6, rms));
    frameRmsDbs.push(db);
    if (rms < 0.008) {
      silentFrames++;
      consecutiveSilentFrames++;
      if (!inPause && consecutiveSilentFrames >= 4) {
        inPause = true;
        pauseCount++;
      }
    } else {
      consecutiveSilentFrames = 0;
      inPause = false;
    }
  }

  const sortedFrameDbs = [...frameRmsDbs].sort((a, b) => a - b);
  const p15Db = sortedFrameDbs[Math.floor(sortedFrameDbs.length * 0.15)] ?? -70;
  const noiseFloorDb = Number(p15Db.toFixed(1));
  const silenceRatio = Number((silentFrames / frameCount).toFixed(3));
  const clippingRate = Number((clippedSamples / Math.max(1, samples.length)).toFixed(5));
  const peakDb = 20 * Math.log10(Math.max(1e-6, maxRms));
  const dynamicRangeDb = Number(Math.max(0, peakDb - p15Db).toFixed(1));

  const globalSpectral = computeSpectralMetrics(samples, TARGET_SAMPLE_RATE);

  // Build overlapping segments (4.0s window, 2.0s hop)
  const rawSegments = buildOverlappingSegments(samples, TARGET_SAMPLE_RATE);
  const segmentAcoustics = rawSegments.map((seg) =>
    computeSegmentAcousticStats(seg.samples, TARGET_SAMPLE_RATE),
  );

  // Detect inter-segment splice / discontinuity transitions (supporting signal only)
  const spliceDeltas: Array<number | null> = [null];
  let spliceBoundariesDetected = 0;
  for (let i = 1; i < segmentAcoustics.length; i++) {
    const rmsDelta = Number(Math.abs(segmentAcoustics[i].rmsDb - segmentAcoustics[i - 1].rmsDb).toFixed(1));
    spliceDeltas.push(rmsDelta);
    const prevC = segmentAcoustics[i - 1].spectralCentroidHz;
    const curC = segmentAcoustics[i].spectralCentroidHz;
    const centroidJump = prevC !== null && curC !== null ? Math.abs(curC - prevC) : 0;
    if (
      segmentAcoustics[i].isSpeechActive &&
      segmentAcoustics[i - 1].isSpeechActive &&
      (rmsDelta >= 12 || centroidJump >= 1300)
    ) {
      spliceBoundariesDetected++;
    }
  }

  // Pitch statistics across segments (supporting prosody descriptor only — NEVER a standalone AI rule!)
  const voicedPitches = segmentAcoustics
    .map((s) => s.pitchEstimateHz)
    .filter((p): p is number => p !== null && Number.isFinite(p));
  const pitchMeanHz =
    voicedPitches.length > 0
      ? Math.round(voicedPitches.reduce((a, b) => a + b, 0) / voicedPitches.length)
      : null;
  const pitchStdHz =
    voicedPitches.length >= 2 && pitchMeanHz !== null
      ? Number(
          Math.sqrt(
            voicedPitches.reduce((acc, p) => acc + Math.pow(p - pitchMeanHz, 2), 0) /
              voicedPitches.length,
          ).toFixed(1),
        )
      : null;

  const speechActiveCount = segmentAcoustics.filter((s) => s.isSpeechActive).length;
  const speechDetected = speechActiveCount > 0 && silenceRatio < 0.96 && durationSec >= 0.4;
  const signalDurationMs = Date.now() - tSignalStart;

  // Step 2: Run REAL Pretrained Audio Deepfake Classifier across overlapping segments (PDF Sections 2, 3, 4)
  onStageProgress?.(
    'forensic_analysis',
    `${trackPrefix}Running pretrained Wav2Vec2 deepfake voice detector across ${rawSegments.length} overlapping 16 kHz mono segment(s).`,
  );

  const configured = getConfiguredAudioClassifier();
  const perSegmentRawScore = new Map<number, { fakeScore: number; rawLabel: string }>();
  let primaryDetectorSummary: PrimaryAudioDetectorSummary;
  let classifierFailed = false;
  let classifierError: string | null = null;
  let rawAggregatedScore: number | null = null;

  if (!speechDetected || rawSegments.length === 0) {
    primaryDetectorSummary = {
      ran: false,
      status: 'skipped_no_speech',
      backend: configured.backend,
      modelId: configured.modelId,
      modelBase: configured.modelBase,
      license: configured.license,
      inputSpec: configured.inputSpec,
      rawAggregatedScore: null,
      scoreLabel: 'Not applicable — insufficient speech energy',
      segmentWindowSec: SEGMENT_WINDOW_SEC,
      segmentHopSec: SEGMENT_HOP_SEC,
      segmentsEvaluated: 0,
      speechActiveSegments: 0,
      aggregationMethod: 'Skipped because no speech-active segments exceeded the -44 dB RMS threshold.',
      inferenceDurationMs: 0,
      errorDetail: null,
    };
  } else {
    const inference = await classifyAudioSegmentsWithPretrainedModel({
      segments: rawSegments,
      onProgress: (done, total, modelId) => {
        onStageProgress?.(
          'forensic_analysis',
          `${trackPrefix}Pretrained classifier (${modelId}): evaluating overlapping segment ${done + 1}/${total}…`,
        );
      },
    });

    if (!inference.ok) {
      classifierFailed = true;
      classifierError = inference.error;
      primaryDetectorSummary = {
        ran: false,
        status: 'technical_failure',
        backend: inference.descriptor.backend,
        modelId: inference.descriptor.modelId,
        modelBase: inference.descriptor.modelBase,
        license: inference.descriptor.license,
        inputSpec: inference.descriptor.inputSpec,
        rawAggregatedScore: null,
        scoreLabel: 'Unavailable — model inference technical failure',
        segmentWindowSec: SEGMENT_WINDOW_SEC,
        segmentHopSec: SEGMENT_HOP_SEC,
        segmentsEvaluated: 0,
        speechActiveSegments: speechActiveCount,
        aggregationMethod: 'Not computed due to pretrained model inference failure.',
        inferenceDurationMs: inference.durationMs,
        errorDetail: inference.error,
      };
    } else {
      for (const item of inference.results) {
        perSegmentRawScore.set(item.index, {
          fakeScore: item.result.rawFakeScore,
          rawLabel: item.result.rawLabel,
        });
      }

      const activeScores: number[] = [];
      for (let i = 0; i < rawSegments.length; i++) {
        const scoreEntry = perSegmentRawScore.get(rawSegments[i].index);
        if (segmentAcoustics[i].isSpeechActive && scoreEntry) {
          activeScores.push(scoreEntry.fakeScore);
        }
      }

      const agg =
        activeScores.length > 0
          ? aggregateRawSegmentScores(activeScores)
          : {
              rawAggregatedScore: null,
              aggregationMethod: 'No speech-active segments available for aggregation.',
            };

      rawAggregatedScore = agg.rawAggregatedScore;
      primaryDetectorSummary = {
        ran: true,
        status: 'completed',
        backend: inference.descriptor.backend,
        modelId: inference.descriptor.modelId,
        modelBase: inference.descriptor.modelBase,
        license: inference.descriptor.license,
        inputSpec: inference.descriptor.inputSpec,
        rawAggregatedScore,
        scoreLabel:
          rawAggregatedScore !== null
            ? `${(rawAggregatedScore * 100).toFixed(1)}% raw synthetic/spoof score (uncalibrated)`
            : 'No speech-active score',
        segmentWindowSec: SEGMENT_WINDOW_SEC,
        segmentHopSec: SEGMENT_HOP_SEC,
        segmentsEvaluated: inference.results.length,
        speechActiveSegments: activeScores.length,
        aggregationMethod: agg.aggregationMethod,
        inferenceDurationMs: inference.durationMs,
        errorDetail: null,
      };
    }
  }

  const assessmentCategory = mapRawScoreToAssessmentCategory({
    classifierFailed,
    durationSec,
    speechActiveCount,
    rawAggregatedScore,
  });

  // Build per-segment report rows preserving raw per-segment model outputs (PDF Section 4 & 10)
  const segments: AudioSegmentFinding[] = rawSegments.map((seg, i) => {
    const ac = segmentAcoustics[i];
    const modelOut = perSegmentRawScore.get(seg.index);
    const spliceDeltaDb = spliceDeltas[i] ?? null;
    const rawScore = modelOut ? modelOut.fakeScore : null;
    const rawLabel = modelOut ? modelOut.rawLabel : null;
    const flagged = Boolean(
      ac.isSpeechActive && rawScore !== null && rawScore > SYNTHETIC_SCORE_THRESHOLD,
    );

    const parts: string[] = [];
    if (!ac.isSpeechActive) {
      parts.push('Low-energy / non-speech segment (excluded from primary aggregation).');
    } else if (rawScore !== null) {
      parts.push(
        `Raw model score: ${(rawScore * 100).toFixed(1)}% synthetic/spoof (${rawLabel ?? 'scored'}).`,
      );
    } else if (classifierFailed) {
      parts.push('Primary classifier unavailable due to technical failure.');
    }
    if (ac.spectralCentroidHz !== null) {
      parts.push(`Centroid ${ac.spectralCentroidHz} Hz.`);
    }
    if (ac.pitchEstimateHz !== null) {
      parts.push(`Est. F0 ${ac.pitchEstimateHz} Hz.`);
    }
    if (spliceDeltaDb !== null && spliceDeltaDb >= 12 && ac.isSpeechActive) {
      parts.push(`Abrupt inter-segment loudness shift (${spliceDeltaDb} dB — supporting cue only).`);
    }

    return {
      startSec: seg.startSec,
      endSec: seg.endSec,
      timestampLabel: `${formatSec(seg.startSec)}–${formatSec(seg.endSec)}`,
      rmsDb: ac.rmsDb,
      zeroCrossingRate: ac.zeroCrossingRate,
      observation: parts.join(' '),
      flagged,
      isSpeechActive: ac.isSpeechActive,
      rawModelScore: rawScore,
      rawModelLabel: rawLabel,
      spectralCentroidHz: ac.spectralCentroidHz,
      pitchEstimateHz: ac.pitchEstimateHz,
      spliceDeltaDb,
    };
  });

  // Supporting Forensic Signals (PDF Section 5 — strictly supporting observations, NEVER standalone proof)
  const encoderLower = (metadata.encoder ?? '').toLowerCase();
  const ttsEncoderMatch =
    /elevenlabs|playht|murf|resemble|wellsaid|azure\s?tts|polly|coqui|bark|xtts|fish\s?audio|cartesia|openai\s?tts/i.test(
      encoderLower,
    );

  const b = globalSpectral.bandEnergyRatios;
  const melBandSummary =
    globalSpectral.spectralCentroidHz !== null
      ? `Band energy split: <300 Hz ${(b.sub300Hz * 100).toFixed(1)}%, speech core 300–3400 Hz ${(b.speechCore300To3400Hz * 100).toFixed(1)}%, upper harmonics 3.4–6 kHz ${(b.upperHarmonics3400To6000Hz * 100).toFixed(1)}%, high band 6–8 kHz ${(b.highBand6000To8000Hz * 100).toFixed(1)}% (centroid ${globalSpectral.spectralCentroidHz} Hz, flatness ${globalSpectral.spectralFlatness ?? 'N/A'}).`
      : 'Insufficient voiced spectral energy to compute frequency-band ratios.';

  const pitchProsodySummary =
    pitchMeanHz !== null
      ? `Estimated voiced F0 mean ${pitchMeanHz} Hz${pitchStdHz !== null ? ` (across-segment SD ${pitchStdHz} Hz)` : ''}, ${pauseCount} pause transition(s), silence ratio ${(silenceRatio * 100).toFixed(1)}%. Supporting descriptor only — steady or monotone delivery occurs naturally in human speech and is NOT proof of synthetic audio.`
      : `Could not isolate clear periodic vocal F0 across sampled frames (${pauseCount} pause transition(s), silence ratio ${(silenceRatio * 100).toFixed(1)}%).`;

  const acousticArtifactsSummary = `Dynamic range ${dynamicRangeDb} dB, 15th-percentile noise floor ${noiseFloorDb} dB, sample clipping rate ${(clippingRate * 100).toFixed(2)}%.${noiseFloorDb <= -68 ? ' Very low noise floor observed (can result from studio noise gating, conferencing suppression like Krisp/Zoom, or clean synthetic generation — supporting context only).' : ''}`;

  const editSpliceSummary =
    spliceBoundariesDetected > 0
      ? `Detected ${spliceBoundariesDetected} abrupt inter-segment loudness/spectral transition(s) — may reflect speaker changes, edits/cuts, or audio splicing (supporting cue only).`
      : 'No abrupt inter-segment loudness or spectral discontinuities detected across sampled windows.';

  const codecMetadataSummary = ttsEncoderMatch
    ? `Container encoder tag ("${metadata.encoder}") matches a known synthetic speech tool.`
    : `Container/codec: ${metadata.audioFormat || metadata.containerFormat || 'Unknown'} / ${metadata.codec || metadata.audioCodec || 'Unknown'}, ${metadata.sampleRate ?? 'unknown'} Hz, ${metadata.bitrate ? `${Math.round(metadata.bitrate / 1000)} kbps` : 'bitrate unknown'}, encoder: ${metadata.encoder || 'Not present'} (lossy codecs & re-encoding can degrade anti-spoofing accuracy).`;

  const supportingSignals: AudioSupportingSignals = {
    melBandSummary,
    bandEnergyRatios: b,
    pitchProsodySummary,
    pitchMeanHz,
    pitchStdHz,
    pauseCount,
    acousticArtifactsSummary,
    editSpliceSummary,
    spliceBoundariesDetected,
    codecMetadataSummary,
  };

  // Assemble transparent indicator buckets for UI (clearly separating primary model findings vs supporting/uncertain cues)
  const detectedIndicators: string[] = [];
  const uncertainSignals: string[] = [];
  const naturalCues: string[] = [];

  if (assessmentCategory === 'Likely synthetic or spoofed speech' && rawAggregatedScore !== null) {
    detectedIndicators.push(
      `Pretrained neural classifier (${primaryDetectorSummary.modelId}) returned a raw aggregated synthetic/spoof score of ${(rawAggregatedScore * 100).toFixed(1)}% across ${primaryDetectorSummary.speechActiveSegments} speech-active overlapping segment(s).`,
    );
    const highSegs = segments.filter((s) => s.flagged);
    if (highSegs.length > 0) {
      detectedIndicators.push(
        `${highSegs.length} of ${primaryDetectorSummary.speechActiveSegments} speech-active segment(s) exceeded the 62% raw synthetic threshold (${highSegs
          .slice(0, 4)
          .map((s) => `${s.timestampLabel}: ${((s.rawModelScore ?? 0) * 100).toFixed(0)}%`)
          .join(', ')}).`,
      );
    }
  } else if (assessmentCategory === 'Likely genuine human speech' && rawAggregatedScore !== null) {
    naturalCues.push(
      `Pretrained neural classifier (${primaryDetectorSummary.modelId}) returned a low raw synthetic/spoof score of ${(rawAggregatedScore * 100).toFixed(1)}% (genuine speech score ${((1 - rawAggregatedScore) * 100).toFixed(1)}%) across ${primaryDetectorSummary.speechActiveSegments} speech-active segment(s).`,
    );
  } else if (assessmentCategory === 'Inconclusive / insufficient evidence') {
    if (rawAggregatedScore !== null) {
      uncertainSignals.push(
        `Pretrained classifier raw score (${(rawAggregatedScore * 100).toFixed(1)}%) fell in the ambiguous middle band (38%–62%), so the evidence is inconclusive.`,
      );
    } else {
      uncertainSignals.push(
        'Insufficient clear speech duration or energy to reach a reliable classification.',
      );
    }
  }

  if (ttsEncoderMatch) {
    detectedIndicators.push(`Embedded audio encoder tag ("${metadata.encoder}") matches a known TTS/voice-synthesis platform.`);
  }
  if (noiseFloorDb <= -68 && speechDetected) {
    uncertainSignals.push(
      `Near-digital-zero noise floor (${noiseFloorDb} dB) — supporting observation only; common in both noise-gated/conferencing audio and synthetic speech.`,
    );
  }
  if (spliceBoundariesDetected > 0) {
    uncertainSignals.push(
      `${spliceBoundariesDetected} abrupt inter-segment loudness/spectral jump(s) observed — could reflect normal cuts, speaker turns, or partial splicing.`,
    );
  }
  if (globalSpectral.highFreqEnergyRatio !== null && globalSpectral.highFreqEnergyRatio < 0.04 && speechDetected) {
    uncertainSignals.push(
      `Low >4 kHz spectral energy (${(globalSpectral.highFreqEnergyRatio * 100).toFixed(1)}%) — frequently caused by telephony/MP3/AAC bandwidth limits as well as older neural vocoders.`,
    );
  }
  if (noiseFloorDb > -64 && noiseFloorDb < -25 && dynamicRangeDb >= 14 && speechDetected) {
    naturalCues.push(
      `Natural ambient room floor (${noiseFloorDb} dB) and organic loudness dynamic range (${dynamicRangeDb} dB) — supporting acoustic observation.`,
    );
  }

  const processingLimitations: string[] = [
    'Raw neural classifier scores are uncalibrated model outputs (NOT calibrated real-world probabilities) and cannot guarantee 100% certainty.',
    'Performance degrades on heavily compressed codecs (low-bitrate MP3/AAC/Opus), telephony/WhatsApp voice notes, loud background music/noise, overlapping speakers, or unseen voice-cloning architectures.',
    ...(durationSec < 2.0
      ? [`Short recording duration (${durationSec.toFixed(2)}s) limits multi-segment reliability.`]
      : []),
    ...(truncatedAtMaxDuration
      ? [`Audio longer than ${MAX_DECODE_SECONDS}s was truncated to the first ${MAX_DECODE_SECONDS}s for bounded server processing.`]
      : []),
    ...(rawSegments.length === MAX_SEGMENTS_EVALUATED && durationSec > 26
      ? [`Evaluated ${MAX_SEGMENTS_EVALUATED} uniformly distributed overlapping ${SEGMENT_WINDOW_SEC}s windows across ${durationSec.toFixed(1)}s.`]
      : []),
    'Supporting acoustic metrics (frequency bands, pitch/prosody, noise floor, splice jumps, codec metadata) are contextual observations only and are never treated as standalone proof of AI generation.',
  ];

  const experimentalMethodsDisclosed: string[] = [
    'Implemented (Primary): Real pretrained Wav2Vec2 deepfake voice classification over overlapping 16 kHz mono segments.',
    'Implemented (Supporting / Deterministic): FFmpeg 16 kHz mono decoding, multi-band windowed DFT spectral ratios, autocorrelation F0 pitch & pause descriptors, noise floor/dynamic range/clipping, inter-segment splice jump detection, and container/codec metadata inspection.',
    'Implemented (Explanation Only): Gemini language explanation grounded strictly on the actual pretrained detector and supporting signal outputs.',
    'Not Integrated / Unavailable: Speaker-identity voiceprint biometric matching against an enrolled target speaker, cryptographic audio watermark verification (e.g., AudioSeal/SynthID Audio), and calibrated benchmark threshold fitting on a custom deployment dataset.',
  ];

  // Step 3: Gemini Explanation ONLY (PDF Sections 2 & 9 — explains actual detector findings, NEVER replaces or invents them!)
  let geminiExplanation: GeminiAudioExplanationResponse | null = null;
  let geminiModelUsed: string | null = null;
  let geminiError: string | null = null;
  const tGemStart = Date.now();

  if (!skipGeminiExplanation && !classifierFailed) {
    onStageProgress?.(
      'ai_assessment',
      `${trackPrefix}Generating grounded forensic explanation of pretrained audio detector & supporting acoustic findings.`,
    );

    const prompt = [
      'You are writing a strictly grounded, honest forensic explanation for TruthLens based ONLY on the real pretrained audio deepfake classifier outputs and supporting acoustic measurements below.',
      'CRITICAL RULES:',
      '1. You are NOT the primary audio detector. Do NOT invent new findings, do NOT override the pretrained classifier category, and do NOT claim calibrated probability or 100% proof.',
      '2. Explain what the pretrained Wav2Vec2 classifier actually measured across overlapping segments, how the documented aggregation produced the category, and what the supporting acoustic signals add as context (not standalone proof).',
      '3. Never use simplistic rules like "flat pitch means AI" or "missing metadata means AI".',
      '',
      `Audio Source: ${sourceTrack === 'extracted_video_audio' ? 'Audio track extracted from uploaded video' : 'Standalone uploaded audio file'} (${metadata.fileName})`,
      `Decoded Duration: ${durationSec.toFixed(2)}s (16 kHz mono PCM)`,
      `Assessment Category: ${assessmentCategory}`,
      `Primary Pretrained Model: ${primaryDetectorSummary.modelId} (License: ${primaryDetectorSummary.license}, Base: ${primaryDetectorSummary.modelBase ?? 'Wav2Vec2'})`,
      `Raw Aggregated Model Score (Uncalibrated): ${rawAggregatedScore !== null ? `${(rawAggregatedScore * 100).toFixed(1)}% synthetic/spoof` : 'N/A'}`,
      `Aggregation Method: ${primaryDetectorSummary.aggregationMethod}`,
      `Per-Segment Raw Outputs:`,
      ...segments.map(
        (s) =>
          `- ${s.timestampLabel} [speechActive=${s.isSpeechActive}]: rawFakeScore=${s.rawModelScore !== null && s.rawModelScore !== undefined ? `${(s.rawModelScore * 100).toFixed(1)}% (${s.rawModelLabel})` : 'N/A'}, RMS=${s.rmsDb} dB, ZCR=${s.zeroCrossingRate}, F0=${s.pitchEstimateHz ?? 'N/A'} Hz`,
      ),
      '',
      'Supporting Forensic Signals (Contextual Only — NOT Standalone Proof):',
      `- Frequency/Mel-Band Distribution: ${melBandSummary}`,
      `- Pitch/Prosody/Pauses: ${pitchProsodySummary}`,
      `- Acoustic Artifacts: ${acousticArtifactsSummary}`,
      `- Edit/Splice Transitions: ${editSpliceSummary}`,
      `- Codec & Container Metadata: ${codecMetadataSummary}`,
      '',
      'Return JSON with:',
      '- explanationSummary: 2-3 clear sentences accurately summarizing the pretrained classifier result, segment consistency, and uncertainty.',
      '- keyObservations: 2-4 concise bullet observations referencing ONLY the real numbers above.',
      '- caveats: 2-3 honest limitations (uncalibrated raw score, codec/noise effects, supporting signals not standalone proof).',
    ].join('\n');

    try {
      const res = await runStructuredGeminiAnalysisWithMeta<GeminiAudioExplanationResponse>({
        parts: [{ text: prompt }],
        responseSchema: AUDIO_EXPLANATION_SCHEMA,
        thinkingLevel: ThinkingLevel.LOW,
        timeoutMs: 22_000,
        onRetry: (evt: GeminiRetryEvent) => {
          onStageProgress?.(
            'ai_assessment',
            `${trackPrefix}Gemini explanation retrying (${evt.attempt}/${evt.totalAttempts} → ${evt.nextModel}): ${evt.reason}`,
            'retrying',
          );
        },
      });
      geminiExplanation = res.data;
      geminiModelUsed = res.modelUsed;
    } catch (err) {
      geminiError = err instanceof Error ? err.message : 'Gemini explanation step unavailable.';
    }
  }
  const gemDurationMs = Date.now() - tGemStart;

  // Convert raw 0..1 score to 0..100 scale for scoringEngine ONLY when primary classifier succeeded on speech
  const primaryScore100 =
    !classifierFailed && rawAggregatedScore !== null && assessmentCategory !== 'Analysis unavailable because of a technical failure'
      ? Math.max(5, Math.min(95, Math.round(rawAggregatedScore * 100)))
      : null;

  const primaryEvidenceStrength =
    primaryScore100 !== null && (primaryScore100 >= 78 || primaryScore100 <= 22) && primaryDetectorSummary.speechActiveSegments >= 2
      ? 'Strong'
      : primaryScore100 !== null && (primaryScore100 > 62 || primaryScore100 < 38)
        ? 'Moderate'
        : 'Low';

  const idPrefix = sourceTrack === 'extracted_video_audio' ? 'video_audio_' : '';

  const detectors: DetectorResult[] = [
    classifierFailed
      ? {
          id: `${idPrefix}pretrained_audio_deepfake_classifier`,
          methodName: `${trackPrefix}Pretrained Audio Deepfake Classifier (Wav2Vec2)`,
          category: 'signal',
          status: 'failed',
          ran: false,
          resultSummary: `Analysis unavailable because of a technical failure — pretrained classifier (${primaryDetectorSummary.modelId}) did not complete: ${classifierError}`,
          rawSignalScore: null,
          signalPolarity: 'neutral',
          reliability: 'High',
          reliabilityWeight: 0.85,
          calibrationStatus: 'Uncalibrated (Experimental)',
          applicability: 'Primary neural speech anti-spoofing classifier on overlapping 16 kHz mono segments',
          evidenceStrength: 'Low',
          evidence: [
            `Backend: ${primaryDetectorSummary.backend} · Model: ${primaryDetectorSummary.modelId} (${primaryDetectorSummary.license})`,
            `Technical error: ${classifierError}`,
            'Per forensic reporting policy, technical failures are excluded from scoring and never treated as synthetic-speech evidence.',
          ],
          limitations: processingLimitations,
          toolUsed: `${primaryDetectorSummary.modelId} (Technical Failure)`,
          correlationGroup: `${idPrefix}pretrained_audio_antispoofing`,
          providerGroup: 'trained_audio_model',
          durationMs: primaryDetectorSummary.inferenceDurationMs,
          detectorVersion: '4.0',
        }
      : !speechDetected || rawAggregatedScore === null
        ? {
            id: `${idPrefix}pretrained_audio_deepfake_classifier`,
            methodName: `${trackPrefix}Pretrained Audio Deepfake Classifier (Wav2Vec2)`,
            category: 'signal',
            status: 'not_applicable',
            ran: false,
            resultSummary: `${assessmentCategory} — no speech-active segments exceeded the -44 dB RMS threshold (${durationSec.toFixed(2)}s decoded).`,
            rawSignalScore: null,
            signalPolarity: 'neutral',
            reliability: 'High',
            reliabilityWeight: 0.85,
            calibrationStatus: 'Uncalibrated (Experimental)',
            applicability: 'Speech recordings with active vocal energy (>= 1.0s)',
            evidenceStrength: 'Low',
            evidence: [
              `Decoded ${durationSec.toFixed(2)}s at 16 kHz mono PCM; silence ratio ${(silenceRatio * 100).toFixed(1)}%.`,
              'No speech-active windows were present for neural anti-spoofing classification.',
            ],
            limitations: processingLimitations,
            toolUsed: `${primaryDetectorSummary.modelId} (Skipped — no speech)`,
            correlationGroup: `${idPrefix}pretrained_audio_antispoofing`,
            providerGroup: 'trained_audio_model',
            durationMs: 0,
            detectorVersion: '4.0',
          }
        : {
            id: `${idPrefix}pretrained_audio_deepfake_classifier`,
            methodName: `${trackPrefix}Pretrained Audio Deepfake Classifier (Wav2Vec2)`,
            category: 'signal',
            status: 'completed',
            ran: true,
            resultSummary: `${assessmentCategory} — raw uncalibrated synthetic/spoof score ${(rawAggregatedScore * 100).toFixed(1)}% across ${primaryDetectorSummary.speechActiveSegments} speech-active overlapping segment(s) (${primaryDetectorSummary.modelId}).`,
            rawSignalScore: primaryScore100,
            signalPolarity: classifyPolarityFromCategory(assessmentCategory),
            reliability: 'High',
            reliabilityWeight: 0.85,
            calibrationStatus: 'Uncalibrated (Experimental)',
            applicability: 'Primary trained speech anti-spoofing classifier over overlapping 16 kHz mono segments',
            evidenceStrength: primaryEvidenceStrength,
            evidence: [
              `Model: ${primaryDetectorSummary.modelId} | Base: ${primaryDetectorSummary.modelBase} | License: ${primaryDetectorSummary.license}`,
              `Input preprocessing: FFmpeg 16 kHz mono PCM, ${SEGMENT_WINDOW_SEC}s overlapping windows (${SEGMENT_HOP_SEC}s hop).`,
              `Aggregation: ${primaryDetectorSummary.aggregationMethod}`,
              ...segments
                .filter((s) => s.rawModelScore !== null && s.rawModelScore !== undefined)
                .slice(0, 6)
                .map(
                  (s) =>
                    `Segment ${s.timestampLabel} (${s.isSpeechActive ? 'speech-active' : 'low-energy'}): raw synthetic score ${((s.rawModelScore ?? 0) * 100).toFixed(1)}% (raw top label: ${s.rawModelLabel})`,
                ),
            ],
            limitations: processingLimitations,
            toolUsed: `${primaryDetectorSummary.modelId} (${primaryDetectorSummary.backend})`,
            correlationGroup: `${idPrefix}pretrained_audio_antispoofing`,
            providerGroup: 'trained_audio_model',
            durationMs: primaryDetectorSummary.inferenceDurationMs,
            detectorVersion: '4.0',
          },
    {
      id: `${idPrefix}audio_spectral_analysis`,
      methodName: `${trackPrefix}Supporting Signal: Multi-Band Spectral & Frequency Profile`,
      category: 'signal',
      status: 'completed',
      ran: true,
      resultSummary: `${melBandSummary} (Supporting observation only — not standalone proof.)`,
      rawSignalScore: null,
      signalPolarity: 'neutral',
      reliability: 'Moderate',
      reliabilityWeight: 0.35,
      calibrationStatus: 'Deterministic Check',
      applicability: 'Supporting frequency-domain energy ratios across <300 Hz, 300–3400 Hz, 3.4–6 kHz, and 6–8 kHz bands',
      evidenceStrength: 'Low',
      evidence: [
        melBandSummary,
        `Spectral centroid: ${globalSpectral.spectralCentroidHz ?? 'N/A'} Hz | >4 kHz energy ratio: ${globalSpectral.highFreqEnergyRatio !== null ? `${(globalSpectral.highFreqEnergyRatio * 100).toFixed(1)}%` : 'N/A'} | Spectral flatness: ${globalSpectral.spectralFlatness ?? 'N/A'}.`,
      ],
      limitations: [
        'MP3/AAC/Opus compression, telephony filters, and microphone frequency response alter spectral rolloff on genuine human recordings. Never used as standalone proof of AI generation.',
      ],
      toolUsed: 'FFmpeg 16 kHz PCM + Windowed Multi-Band DFT',
      correlationGroup: `${idPrefix}audio_supporting_signals`,
      providerGroup: 'deterministic_signal',
      durationMs: signalDurationMs,
      detectorVersion: '4.0',
    },
    {
      id: `${idPrefix}audio_prosody_and_artifacts`,
      methodName: `${trackPrefix}Supporting Signal: Pitch, Prosody, Pauses & Acoustic Artifacts`,
      category: 'signal',
      status: 'completed',
      ran: true,
      resultSummary: `${pitchProsodySummary} ${acousticArtifactsSummary}`,
      rawSignalScore: null,
      signalPolarity: 'neutral',
      reliability: 'Low',
      reliabilityWeight: 0.25,
      calibrationStatus: 'Deterministic Check',
      applicability: 'Supporting pitch (F0), pause cadence, noise floor, dynamic range, and clipping inspection',
      evidenceStrength: 'Low',
      evidence: [pitchProsodySummary, acousticArtifactsSummary, editSpliceSummary],
      limitations: [
        'Monotone pitch, studio noise gating, conferencing noise suppression (Krisp/Zoom/Teams), and podcast compression occur routinely in real human audio. Simplistic rules such as "flat pitch means AI" are explicitly avoided.',
      ],
      toolUsed: 'Autocorrelation F0 + Segment RMS/ZCR/Splice Analyzer',
      correlationGroup: `${idPrefix}audio_supporting_signals`,
      providerGroup: 'deterministic_signal',
      durationMs: signalDurationMs,
      detectorVersion: '4.0',
    },
    {
      id: `${idPrefix}audio_metadata_inspection`,
      methodName: `${trackPrefix}Supporting Signal: Audio Codec & Container Metadata`,
      category: 'metadata',
      status: 'completed',
      ran: true,
      resultSummary: codecMetadataSummary,
      rawSignalScore: ttsEncoderMatch ? 78 : null,
      signalPolarity: ttsEncoderMatch ? 'ai_indicator' : 'neutral',
      reliability: ttsEncoderMatch ? 'Moderate' : 'Low',
      reliabilityWeight: 0.35,
      calibrationStatus: 'Deterministic Check',
      applicability: 'Audio container headers, sample rate, bitrate, and encoder tags',
      evidenceStrength: ttsEncoderMatch ? 'Moderate' : 'Low',
      evidence: [
        codecMetadataSummary,
        `Decoded duration: ${durationSec.toFixed(2)}s | Original sample rate: ${metadata.sampleRate ?? 'Unknown'} Hz | Channels: ${metadata.audioChannels ?? 'Unknown'}.`,
      ],
      limitations: [
        'Metadata can be stripped or rewritten by messaging apps, editors, or transcoders. Missing tags are strictly neutral.',
      ],
      toolUsed: 'music-metadata + FFprobe Container Inspector',
      correlationGroup: `${idPrefix}metadata_context`,
      providerGroup: 'deterministic_metadata',
      durationMs: signalDurationMs,
      detectorVersion: '4.0',
    },
    ...(geminiExplanation
      ? [
          {
            id: `${idPrefix}audio_grounded_explanation`,
            methodName: `${trackPrefix}Grounded Findings Explanation (Gemini — Non-Scoring)`,
            category: 'ai_reasoning' as const,
            status: 'completed' as const,
            ran: true,
            resultSummary: geminiExplanation.explanationSummary,
            rawSignalScore: null, // Strictly explanation-only: NEVER replaces or scores alongside trained audio detector
            signalPolarity: 'neutral' as const,
            reliability: 'Experimental' as const,
            reliabilityWeight: 0,
            calibrationStatus: 'Uncalibrated (Experimental)' as const,
            applicability: 'Explains actual pretrained detector & supporting signal outputs in plain language without generating independent detection scores',
            evidenceStrength: 'Low' as const,
            evidence: geminiExplanation.keyObservations,
            limitations: [
              'Gemini is used strictly to explain real detector findings and does NOT act as an independent audio deepfake classifier.',
              ...geminiExplanation.caveats,
            ],
            toolUsed: `Gemini Explanation Layer (${geminiModelUsed ?? GEMINI_FORENSIC_MODEL})`,
            correlationGroup: `${idPrefix}explanation_only`,
            providerGroup: 'gemini_multimodal',
            durationMs: gemDurationMs,
            detectorVersion: '4.0',
          },
        ]
      : []),
    {
      id: `${idPrefix}voice_cloning_biometric`,
      methodName: `${trackPrefix}Speaker-Identity Voiceprint Biometric Verification`,
      category: 'signal',
      status: 'unavailable',
      ran: false,
      resultSummary:
        'Not performed — speaker-identity enrollment matching (ECAPA-TDNN / x-vector against a verified reference voice) is not integrated in this runtime.',
      rawSignalScore: null,
      signalPolarity: 'neutral',
      reliability: 'High',
      reliabilityWeight: 0.7,
      calibrationStatus: 'Uncalibrated (Experimental)',
      applicability: 'Comparing an unknown recording against an enrolled reference voiceprint',
      evidenceStrength: 'Low',
      evidence: [
        'Primary anti-spoofing detection ran via Wav2Vec2, but verifying whether the voice matches a specific named individual requires an enrolled biometric reference.',
      ],
      limitations: ['Requires verified reference recordings of the target speaker and speaker-embedding verification.'],
      toolUsed: 'Not integrated (Speaker Biometric Matcher unavailable)',
      correlationGroup: `${idPrefix}voice_biometrics`,
      providerGroup: 'unavailable',
      durationMs: 0,
      detectorVersion: '4.0',
    },
    ...(sourceTrack === 'standalone_audio'
      ? [provenance.c2paDetector, provenance.synthIdDetector]
      : []),
  ];

  const audioProfile: AudioProfile = {
    sourceTrack,
    assessmentCategory,
    primaryDetector: primaryDetectorSummary,
    supportingSignals,
    processingLimitations,
    experimentalMethodsDisclosed,
    waveformPeaks,
    spectralCentroidHz: globalSpectral.spectralCentroidHz,
    highFreqEnergyRatio: globalSpectral.highFreqEnergyRatio,
    silenceRatio,
    dynamicRangeDb,
    clippingRate,
    noiseFloorDb,
    spectralFlatness: globalSpectral.spectralFlatness,
    speechDetected,
    syntheticIndicatorCount: detectedIndicators.length,
    detectedIndicators,
    uncertainSignals,
    naturalCues,
    segments,
  };

  return {
    detectors,
    watermarks: sourceTrack === 'standalone_audio' ? provenance.provenanceWatermarks : [],
    audioProfile,
    geminiStatus: skipGeminiExplanation
      ? 'unavailable'
      : geminiExplanation
        ? 'completed'
        : classifierFailed
          ? 'unavailable'
          : 'failed',
    geminiModelUsed,
    geminiFailureDetail: geminiError,
  };
}
