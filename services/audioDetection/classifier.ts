import os from 'node:os';
import path from 'node:path';
import { IS_SERVERLESS } from '../storage.ts';

export interface SegmentClassificationResult {
  rawFakeScore: number; // 0..1 raw uncalibrated model score for synthetic/spoof class
  rawRealScore: number; // 0..1 raw uncalibrated model score for genuine/human class
  rawLabel: string; // highest-scoring raw label returned by model
  allLabels: Array<{ label: string; score: number }>;
}

export interface AudioClassifierDescriptor {
  backend: 'custom_endpoint' | 'huggingface_inference_api' | 'local_onnx_wav2vec2';
  modelId: string;
  modelBase: string;
  license: string;
  inputSpec: string;
}

const DEFAULT_LOCAL_ONNX_MODEL = 'ai8shiro/deepfake-audio-wav2vec2-ONNX';
const DEFAULT_LOCAL_ONNX_DTYPE = 'q4';
const DEFAULT_HF_API_MODEL = 'garystafford/wav2vec2-deepfake-voice-detector';

const MODEL_METADATA: Record<string, { base: string; license: string }> = {
  'ai8shiro/deepfake-audio-wav2vec2-ONNX': {
    base: 'Vansh180/deepfake-audio-wav2vec2 (Balanced ASVspoof 2021 Wav2Vec2)',
    license: 'MIT',
  },
  'as1605/Deepfake-audio-detection-V2': {
    base: 'MelodyMachine/Deepfake-audio-detection-V2 (Wav2Vec2)',
    license: 'Apache-2.0',
  },
  'garystafford/wav2vec2-deepfake-voice-detector': {
    base: 'Gustking/wav2vec2-large-xlsr-deepfake-audio-classification (Wav2Vec2-XLSR)',
    license: 'Apache-2.0',
  },
};

export function getConfiguredAudioClassifier(): AudioClassifierDescriptor {
  if (process.env.AUDIO_DETECTOR_API_URL?.trim()) {
    const modelId =
      process.env.AUDIO_DETECTOR_MODEL_NAME?.trim() ||
      'Custom Audio Anti-Spoofing Endpoint (AUDIO_DETECTOR_API_URL)';
    const meta = MODEL_METADATA[modelId];
    return {
      backend: 'custom_endpoint',
      modelId,
      modelBase: meta?.base ?? 'Configured server-side audio anti-spoofing model',
      license: meta?.license ?? 'Configured deployment license',
      inputSpec: '16 kHz mono WAV / PCM',
    };
  }

  if (process.env.HF_TOKEN?.trim()) {
    const modelId = process.env.AUDIO_DETECTOR_HF_MODEL?.trim() || DEFAULT_HF_API_MODEL;
    const meta = MODEL_METADATA[modelId];
    return {
      backend: 'huggingface_inference_api',
      modelId,
      modelBase: meta?.base ?? 'Pretrained Wav2Vec2 Audio Classification',
      license: meta?.license ?? 'Verify model card license',
      inputSpec: '16 kHz mono WAV (Hugging Face Inference API)',
    };
  }

  const modelId = process.env.AUDIO_DETECTOR_LOCAL_MODEL?.trim() || DEFAULT_LOCAL_ONNX_MODEL;
  const dtype = process.env.AUDIO_DETECTOR_LOCAL_DTYPE?.trim() || DEFAULT_LOCAL_ONNX_DTYPE;
  const meta = MODEL_METADATA[modelId];
  return {
    backend: 'local_onnx_wav2vec2',
    modelId: `${modelId} (${dtype})`,
    modelBase: meta?.base ?? 'Pretrained Wav2Vec2 ONNX Audio Classification',
    license: meta?.license ?? 'Verify model card license',
    inputSpec: '16 kHz mono Float32 PCM (normalized)',
  };
}

function encodeFloat32ToWavBuffer(samples: Float32Array, sampleRate = 16000): Buffer {
  const numSamples = samples.length;
  const dataBytes = numSamples * 2;
  const buf = Buffer.alloc(44 + dataBytes);

  buf.write('RIFF', 0, 'ascii');
  buf.writeUInt32LE(36 + dataBytes, 4);
  buf.write('WAVE', 8, 'ascii');
  buf.write('fmt ', 12, 'ascii');
  buf.writeUInt32LE(16, 16); // PCM chunk size
  buf.writeUInt16LE(1, 20); // PCM linear
  buf.writeUInt16LE(1, 22); // Mono
  buf.writeUInt32LE(sampleRate, 24);
  buf.writeUInt32LE(sampleRate * 2, 28); // Byte rate
  buf.writeUInt16LE(2, 32); // Block align
  buf.writeUInt16LE(16, 34); // Bits per sample
  buf.write('data', 36, 'ascii');
  buf.writeUInt32LE(dataBytes, 40);

  let offset = 44;
  for (let i = 0; i < numSamples; i++) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    const int16 = clamped < 0 ? Math.round(clamped * 32768) : Math.round(clamped * 32767);
    buf.writeInt16LE(int16, offset);
    offset += 2;
  }
  return buf;
}

function normalizeModelOutputs(
  entries: Array<{ label: string; score: number }>,
): SegmentClassificationResult {
  const cleaned = entries
    .filter((e) => typeof e?.label === 'string' && typeof e?.score === 'number' && Number.isFinite(e.score))
    .map((e) => ({ label: e.label.trim(), score: Math.max(0, Math.min(1, e.score)) }))
    .sort((a, b) => b.score - a.score);

  if (cleaned.length === 0) {
    throw new Error('Audio classifier returned no valid class scores.');
  }

  const fakeRegex = /\b(fake|spoof|synthetic|ai|generated|cloned|deepfake|label_1)\b/i;
  const realRegex = /\b(real|bonafide|bona_fide|genuine|human|authentic|natural|label_0)\b/i;

  let fakeScore: number | null = null;
  let realScore: number | null = null;

  for (const item of cleaned) {
    if (fakeScore === null && fakeRegex.test(item.label)) {
      fakeScore = item.score;
    } else if (realScore === null && realRegex.test(item.label)) {
      realScore = item.score;
    }
  }

  if (fakeScore === null && realScore !== null) {
    fakeScore = Math.max(0, Math.min(1, 1 - realScore));
  } else if (realScore === null && fakeScore !== null) {
    realScore = Math.max(0, Math.min(1, 1 - fakeScore));
  }

  if (fakeScore === null || realScore === null) {
    throw new Error(
      `Audio classifier returned unrecognized label names (${cleaned.map((c) => c.label).join(', ')}).`,
    );
  }

  return {
    rawFakeScore: Number(fakeScore.toFixed(4)),
    rawRealScore: Number(realScore.toFixed(4)),
    rawLabel: cleaned[0].label,
    allLabels: cleaned,
  };
}

async function fetchWithRetry(
  url: string,
  init: RequestInit,
  options: { timeoutMs: number; maxAttempts: number },
): Promise<Response> {
  let lastError: Error | null = null;

  for (let attempt = 1; attempt <= options.maxAttempts; attempt++) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), options.timeoutMs);
    try {
      const res = await fetch(url, { ...init, signal: controller.signal });
      clearTimeout(timer);

      // Retry transient 503 (e.g. HF model cold loading) or 429 rate-limit responses
      if ((res.status === 503 || res.status === 429) && attempt < options.maxAttempts) {
        let waitMs = 2000 * attempt;
        try {
          const body = (await res.json()) as { estimated_time?: number };
          if (typeof body.estimated_time === 'number' && Number.isFinite(body.estimated_time)) {
            waitMs = Math.min(12_000, Math.max(1500, Math.round(body.estimated_time * 1000)));
          }
        } catch {
          // Ignore non-JSON retry hint
        }
        await new Promise((r) => setTimeout(r, waitMs));
        continue;
      }

      return res;
    } catch (err) {
      clearTimeout(timer);
      lastError = err instanceof Error ? err : new Error(String(err));
      if (attempt < options.maxAttempts) {
        await new Promise((r) => setTimeout(r, 1200 * attempt));
      }
    }
  }

  throw lastError ?? new Error('Audio classifier request failed after retries.');
}

async function classifyViaCustomEndpoint(
  samples: Float32Array,
): Promise<SegmentClassificationResult> {
  const url = process.env.AUDIO_DETECTOR_API_URL!.trim();
  const apiKey = process.env.AUDIO_DETECTOR_API_KEY?.trim();
  const wav = encodeFloat32ToWavBuffer(samples, 16000);

  const headers: Record<string, string> = {
    'Content-Type': 'audio/wav',
    Accept: 'application/json',
  };
  if (apiKey) {
    headers.Authorization = `Bearer ${apiKey}`;
  }

  const res = await fetchWithRetry(
    url,
    {
      method: 'POST',
      headers,
      body: new Uint8Array(wav),
    },
    { timeoutMs: 22_000, maxAttempts: 2 },
  );

  if (!res.ok) {
    const snippet = (await res.text().catch(() => '')).slice(0, 180);
    throw new Error(`Custom audio detector endpoint returned HTTP ${res.status}: ${snippet || res.statusText}`);
  }

  const data = (await res.json()) as unknown;
  if (Array.isArray(data)) {
    return normalizeModelOutputs(data as Array<{ label: string; score: number }>);
  }
  if (data && typeof data === 'object' && Array.isArray((data as { results?: unknown }).results)) {
    return normalizeModelOutputs(
      (data as { results: Array<{ label: string; score: number }> }).results,
    );
  }
  throw new Error('Custom audio detector endpoint returned an unexpected JSON schema.');
}

async function classifyViaHuggingFaceApi(
  samples: Float32Array,
  modelId: string,
): Promise<SegmentClassificationResult> {
  const token = process.env.HF_TOKEN!.trim();
  const wav = encodeFloat32ToWavBuffer(samples, 16000);
  const url = `https://router.huggingface.co/hf-inference/models/${modelId}`;

  const res = await fetchWithRetry(
    url,
    {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${token}`,
        'Content-Type': 'audio/wav',
        Accept: 'application/json',
      },
      body: new Uint8Array(wav),
    },
    { timeoutMs: 25_000, maxAttempts: 3 },
  );

  if (!res.ok) {
    const snippet = (await res.text().catch(() => '')).slice(0, 180);
    throw new Error(
      `Hugging Face Inference API (${modelId}) returned HTTP ${res.status}: ${snippet || res.statusText}`,
    );
  }

  const data = (await res.json()) as unknown;
  if (Array.isArray(data)) {
    return normalizeModelOutputs(data as Array<{ label: string; score: number }>);
  }
  throw new Error(`Hugging Face Inference API (${modelId}) returned an unexpected response payload.`);
}

type LocalAudioPipelineFn = (
  input: Float32Array,
  options?: { top_k?: number },
) => Promise<Array<{ label: string; score: number }>>;

let localPipelinePromise: Promise<LocalAudioPipelineFn> | null = null;
let localPipelineModelKey: string | null = null;

async function getLocalOnnxClassifier(): Promise<LocalAudioPipelineFn> {
  const modelId = process.env.AUDIO_DETECTOR_LOCAL_MODEL?.trim() || DEFAULT_LOCAL_ONNX_MODEL;
  const dtype = process.env.AUDIO_DETECTOR_LOCAL_DTYPE?.trim() || DEFAULT_LOCAL_ONNX_DTYPE;
  const key = `${modelId}::${dtype}`;

  if (localPipelinePromise && localPipelineModelKey === key) {
    return localPipelinePromise;
  }

  localPipelineModelKey = key;
  localPipelinePromise = (async () => {
    const transformers = await import('@huggingface/transformers');
    transformers.env.cacheDir = IS_SERVERLESS
      ? path.join(os.tmpdir(), 'truthlens-hf-cache')
      : path.resolve(process.cwd(), 'data', 'hf-cache');
    transformers.env.allowLocalModels = true;

    const clf = await transformers.pipeline('audio-classification', modelId, {
      dtype: dtype as 'q4' | 'q8' | 'fp16' | 'fp32' | 'q4f16' | 'bnb4',
    });
    return clf as unknown as LocalAudioPipelineFn;
  })().catch((err) => {
    localPipelinePromise = null;
    localPipelineModelKey = null;
    throw err;
  });

  return localPipelinePromise;
}

async function classifyViaLocalOnnx(
  samples: Float32Array,
  timeoutMs = 60_000,
): Promise<SegmentClassificationResult> {
  let timer: ReturnType<typeof setTimeout> | null = null;
  try {
    const work = (async () => {
      const clf = await getLocalOnnxClassifier();
      const raw = await clf(samples, { top_k: 5 });
      return normalizeModelOutputs(raw);
    })();

    const timeout = new Promise<never>((_resolve, reject) => {
      timer = setTimeout(
        () => reject(new Error(`Local ONNX Wav2Vec2 inference timed out after ${Math.round(timeoutMs / 1000)}s.`)),
        timeoutMs,
      );
    });

    return await Promise.race([work, timeout]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

export async function classifyAudioSegmentsWithPretrainedModel(options: {
  segments: Array<{ index: number; startSec: number; endSec: number; samples: Float32Array }>;
  onProgress?: (completed: number, total: number, modelId: string) => void;
}): Promise<{
  ok: true;
  descriptor: AudioClassifierDescriptor;
  results: Array<{ index: number; result: SegmentClassificationResult }>;
  durationMs: number;
} | {
  ok: false;
  descriptor: AudioClassifierDescriptor;
  error: string;
  durationMs: number;
}> {
  const descriptor = getConfiguredAudioClassifier();
  const t0 = Date.now();

  try {
    const results: Array<{ index: number; result: SegmentClassificationResult }> = [];
    for (let i = 0; i < options.segments.length; i++) {
      const seg = options.segments[i];
      options.onProgress?.(i, options.segments.length, descriptor.modelId);

      let res: SegmentClassificationResult;
      if (descriptor.backend === 'custom_endpoint') {
        res = await classifyViaCustomEndpoint(seg.samples);
      } else if (descriptor.backend === 'huggingface_inference_api') {
        res = await classifyViaHuggingFaceApi(seg.samples, descriptor.modelId);
      } else {
        // First segment gets extra budget for one-time ONNX weight download/cache warm-up
        res = await classifyViaLocalOnnx(seg.samples, i === 0 ? 75_000 : 25_000);
      }

      results.push({ index: seg.index, result: res });
    }

    return {
      ok: true,
      descriptor,
      results,
      durationMs: Date.now() - t0,
    };
  } catch (err) {
    return {
      ok: false,
      descriptor,
      error: err instanceof Error ? err.message : 'Pretrained audio deepfake classifier failed unexpectedly.',
      durationMs: Date.now() - t0,
    };
  }
}
