import { execFile } from 'node:child_process';
import crypto from 'node:crypto';
import fsSync from 'node:fs';
import fs from 'node:fs/promises';
import path from 'node:path';
import { promisify } from 'node:util';
import { fileTypeFromBuffer } from 'file-type';
import sharp from 'sharp';
import type { MediaType } from '../src/types/forensics.ts';

const execFileAsync = promisify(execFile);

export const IMAGE_EXTENSIONS = new Set(['.jpg', '.jpeg', '.png', '.webp']);

export const VIDEO_EXTENSIONS = new Set([
  '.mp4',
  '.mov',
  '.m4v',
  '.webm',
  '.mkv',
  '.avi',
  '.mpeg',
  '.mpg',
  '.ogv',
  '.3gp',
  '.3g2',
  '.flv',
  '.f4v',
  '.wmv',
  '.asf',
  '.ts',
  '.mts',
  '.m2ts',
  '.vob',
  '.divx',
]);

export const AUDIO_EXTENSIONS = new Set([
  '.wav',
  '.mp3',
  '.m4a',
  '.ogg',
  '.oga',
  '.flac',
  '.aac',
  '.opus',
  '.wma',
  '.amr',
  '.aiff',
  '.aif',
  '.weba',
]);

export const DOCUMENT_EXTENSIONS = new Set(['.pdf', '.docx', '.txt']);

export const ALLOWED_EXTENSIONS = new Set([
  ...IMAGE_EXTENSIONS,
  ...VIDEO_EXTENSIONS,
  ...AUDIO_EXTENSIONS,
  ...DOCUMENT_EXTENSIONS,
]);

export const MAX_IMAGE_PIXELS = 100_000_000; // Decompression bomb guard (100MP)

export interface ValidatedFile {
  originalName: string;
  extension: string;
  declaredMime: string;
  detectedMime: string;
  mediaType: MediaType;
  fileSizeBytes: number;
  sha256: string;
  warnings: string[];
}

export class FileValidationError extends Error {
  code: 'UNSUPPORTED_FORMAT' | 'FILE_TOO_LARGE' | 'CORRUPTED_FILE' | 'SECURITY_BLOCKED';
  statusCode: number;

  constructor(
    message: string,
    code: 'UNSUPPORTED_FORMAT' | 'FILE_TOO_LARGE' | 'CORRUPTED_FILE' | 'SECURITY_BLOCKED',
    statusCode = 400,
  ) {
    super(message);
    this.name = 'FileValidationError';
    this.code = code;
    this.statusCode = statusCode;
  }
}

const VIDEO_MIMES = new Set([
  'video/mp4',
  'video/quicktime',
  'video/webm',
  'video/x-matroska',
  'video/matroska',
  'video/x-msvideo',
  'video/avi',
  'video/msvideo',
  'video/vnd.avi',
  'video/x-m4v',
  'video/mpeg',
  'video/ogg',
  'video/3gpp',
  'video/3gpp2',
  'video/x-flv',
  'video/x-ms-wmv',
  'video/x-ms-asf',
  'video/mp2t',
  'video/MP2T',
  'application/mp4',
  'application/x-matroska',
]);

const AUDIO_MIMES = new Set([
  'audio/wav',
  'audio/x-wav',
  'audio/vnd.wave',
  'audio/wave',
  'audio/mpeg',
  'audio/mp3',
  'audio/mp4',
  'audio/x-m4a',
  'audio/m4a',
  'audio/aac',
  'audio/x-aac',
  'audio/ogg',
  'audio/opus',
  'audio/flac',
  'audio/x-flac',
  'audio/webm',
  'audio/x-ms-wma',
  'audio/amr',
  'audio/aiff',
  'audio/x-aiff',
  'application/ogg',
]);

const EXT_TO_FALLBACK_MIME: Record<string, string> = {
  '.mp4': 'video/mp4',
  '.mov': 'video/quicktime',
  '.m4v': 'video/mp4',
  '.webm': 'video/webm',
  '.mkv': 'video/x-matroska',
  '.avi': 'video/x-msvideo',
  '.mpeg': 'video/mpeg',
  '.mpg': 'video/mpeg',
  '.ogv': 'video/ogg',
  '.3gp': 'video/3gpp',
  '.3g2': 'video/3gpp2',
  '.flv': 'video/x-flv',
  '.f4v': 'video/mp4',
  '.wmv': 'video/x-ms-wmv',
  '.asf': 'video/x-ms-asf',
  '.ts': 'video/mp2t',
  '.mts': 'video/mp2t',
  '.m2ts': 'video/mp2t',
  '.vob': 'video/mpeg',
  '.divx': 'video/x-msvideo',
  '.wav': 'audio/wav',
  '.mp3': 'audio/mpeg',
  '.m4a': 'audio/mp4',
  '.ogg': 'audio/ogg',
  '.oga': 'audio/ogg',
  '.flac': 'audio/flac',
  '.aac': 'audio/aac',
  '.opus': 'audio/opus',
  '.wma': 'audio/x-ms-wma',
  '.amr': 'audio/amr',
  '.aiff': 'audio/aiff',
  '.aif': 'audio/aiff',
  '.weba': 'audio/webm',
};

function classifyByMime(mime: string, ext: string): MediaType | null {
  if (['image/jpeg', 'image/png', 'image/webp'].includes(mime)) {
    return 'image';
  }
  // Audio extensions sharing ISO-BMFF / OGG / WebM / ASF containers must stay audio unless ffprobe proves a real video track
  if (AUDIO_EXTENSIONS.has(ext) && (AUDIO_MIMES.has(mime) || VIDEO_MIMES.has(mime))) {
    return 'audio';
  }
  if (VIDEO_EXTENSIONS.has(ext) && (VIDEO_MIMES.has(mime) || mime === 'audio/mp4' || mime === 'application/ogg')) {
    return 'video';
  }
  if (VIDEO_MIMES.has(mime)) {
    return 'video';
  }
  if (AUDIO_MIMES.has(mime)) {
    return 'audio';
  }
  if (mime === 'application/pdf') {
    return 'pdf';
  }
  if (
    mime === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' ||
    (mime === 'application/zip' && ext === '.docx')
  ) {
    return 'document';
  }
  if (mime === 'text/plain' && ext === '.txt') {
    return 'document';
  }
  return null;
}

function looksLikePlainText(buffer: Buffer): boolean {
  const sample = buffer.subarray(0, Math.min(buffer.length, 4096));
  if (sample.length === 0) return false;
  let suspiciousControlChars = 0;
  for (let i = 0; i < sample.length; i++) {
    const byte = sample[i];
    if (byte === 0) return false;
    if (byte < 9 || (byte > 13 && byte < 32)) {
      suspiciousControlChars++;
    }
  }
  return suspiciousControlChars / sample.length < 0.02;
}

interface ProbeSummary {
  ok: boolean;
  hasVideo: boolean;
  hasAudio: boolean;
  formatName: string | null;
  errorDetail?: string;
}

async function probeMediaStreams(filePath: string): Promise<ProbeSummary> {
  const attempts: string[][] = [
    [
      '-v',
      'quiet',
      '-analyzeduration',
      '20000000',
      '-probesize',
      '20000000',
      '-show_entries',
      'format=format_name,duration:stream=codec_type,codec_name:stream_disposition=attached_pic',
      '-of',
      'json',
      filePath,
    ],
    [
      '-v',
      'error',
      '-show_entries',
      'format=format_name,duration:stream=codec_type,codec_name',
      '-of',
      'json',
      filePath,
    ],
  ];

  let lastErr = '';
  for (const args of attempts) {
    try {
      const { stdout } = await execFileAsync('ffprobe', args, {
        timeout: 25_000,
        maxBuffer: 4 * 1024 * 1024,
      });
      const parsed = JSON.parse(stdout) as {
        streams?: Array<{
          codec_type?: string;
          codec_name?: string;
          disposition?: { attached_pic?: number };
        }>;
        format?: { format_name?: string; duration?: string };
      };
      const streams = parsed.streams ?? [];
      const realVideoStreams = streams.filter(
        (s) => s.codec_type === 'video' && !s.disposition?.attached_pic,
      );
      const anyVideoStreams = streams.filter((s) => s.codec_type === 'video');
      const audioStreams = streams.filter((s) => s.codec_type === 'audio');
      return {
        ok: true,
        hasVideo: realVideoStreams.length > 0 || (anyVideoStreams.length > 0 && audioStreams.length === 0),
        hasAudio: audioStreams.length > 0,
        formatName: parsed.format?.format_name ?? null,
      };
    } catch (err) {
      lastErr = err instanceof Error ? err.message : String(err);
    }
  }

  return {
    ok: false,
    hasVideo: false,
    hasAudio: false,
    formatName: null,
    errorDetail: lastErr,
  };
}

// Stream SHA-256 computation so large video files (up to 500MB) never spike RAM
export function computeFileSha256Streaming(filePath: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const hash = crypto.createHash('sha256');
    const stream = fsSync.createReadStream(filePath, { highWaterMark: 64 * 1024 });
    stream.on('data', (chunk) => hash.update(chunk));
    stream.on('end', () => resolve(hash.digest('hex')));
    stream.on('error', reject);
  });
}

async function readHeadBuffer(filePath: string, maxBytes: number): Promise<Buffer> {
  const handle = await fs.open(filePath, 'r');
  try {
    const stat = await handle.stat();
    const len = Math.min(stat.size, maxBytes);
    const buf = Buffer.alloc(len);
    await handle.read(buf, 0, len, 0);
    return buf;
  } finally {
    await handle.close();
  }
}

function detectContainerMimeFromHeader(headBuffer: Buffer, ext: string): string {
  if (headBuffer.length >= 12) {
    const boxType = headBuffer.subarray(4, 8).toString('latin1');
    if (['ftyp', 'moov', 'wide', 'free', 'mdat', 'skip', 'pnot'].includes(boxType)) {
      if (AUDIO_EXTENSIONS.has(ext)) return 'audio/mp4';
      return ext === '.mov' ? 'video/quicktime' : 'video/mp4';
    }
    const riff = headBuffer.subarray(0, 4).toString('latin1');
    const riffSub = headBuffer.subarray(8, 12).toString('latin1');
    if (riff === 'RIFF' && riffSub === 'AVI ') return 'video/x-msvideo';
    if (riff === 'RIFF' && riffSub === 'WAVE') return 'audio/wav';
    if (riff === 'FORM' && (riffSub === 'AIFF' || riffSub === 'AIFC')) return 'audio/aiff';
  }
  if (
    headBuffer.length >= 4 &&
    headBuffer[0] === 0x1a &&
    headBuffer[1] === 0x45 &&
    headBuffer[2] === 0xdf &&
    headBuffer[3] === 0xa3
  ) {
    if (ext === '.weba') return 'audio/webm';
    return ext === '.mkv' ? 'video/x-matroska' : 'video/webm';
  }
  if (headBuffer.length >= 4 && headBuffer.subarray(0, 4).toString('latin1') === 'OggS') {
    return ext === '.ogv' ? 'video/ogg' : 'audio/ogg';
  }
  if (headBuffer.length >= 4 && headBuffer.subarray(0, 4).toString('latin1') === 'fLaC') {
    return 'audio/flac';
  }
  if (headBuffer.length >= 3 && headBuffer.subarray(0, 3).toString('latin1') === 'FLV') {
    return 'video/x-flv';
  }
  if (
    headBuffer.length >= 4 &&
    headBuffer[0] === 0x30 &&
    headBuffer[1] === 0x26 &&
    headBuffer[2] === 0xb2 &&
    headBuffer[3] === 0x75
  ) {
    return ext === '.wma' ? 'audio/x-ms-wma' : 'video/x-ms-wmv';
  }
  if (
    headBuffer.length >= 4 &&
    headBuffer[0] === 0x00 &&
    headBuffer[1] === 0x00 &&
    headBuffer[2] === 0x01 &&
    (headBuffer[3] === 0xba || headBuffer[3] === 0xb3)
  ) {
    return 'video/mpeg';
  }
  if (headBuffer.length >= 189 && headBuffer[0] === 0x47 && headBuffer[188] === 0x47) {
    return 'video/mp2t';
  }
  if (
    headBuffer.length >= 3 &&
    ((headBuffer[0] === 0x49 && headBuffer[1] === 0x44 && headBuffer[2] === 0x33) ||
      (headBuffer[0] === 0xff && (headBuffer[1] & 0xe0) === 0xe0))
  ) {
    return ext === '.aac' ? 'audio/aac' : 'audio/mpeg';
  }
  return '';
}

export async function validateUploadedFile(
  filePath: string,
  originalName: string,
  declaredMime: string,
  maxSizeMb: number,
): Promise<ValidatedFile> {
  const stat = await fs.stat(filePath);
  const maxBytes = maxSizeMb * 1024 * 1024;

  if (stat.size === 0) {
    throw new FileValidationError(
      'Uploaded file is empty (0 bytes) and cannot be analysed.',
      'CORRUPTED_FILE',
      400,
    );
  }

  if (stat.size > maxBytes) {
    throw new FileValidationError(
      `File exceeds the configured upload limit of ${maxSizeMb} MB.`,
      'FILE_TOO_LARGE',
      413,
    );
  }

  const ext = path.extname(originalName).toLowerCase();
  const declaredLower = (declaredMime || '').toLowerCase();
  const isDeclaredVideo = declaredLower.startsWith('video/');
  const isDeclaredAudio = declaredLower.startsWith('audio/');

  if (!ALLOWED_EXTENSIONS.has(ext) && !isDeclaredVideo && !isDeclaredAudio) {
    throw new FileValidationError(
      `Unsupported file format "${ext || 'unknown'}". Supported formats: Images (JPG, PNG, WEBP), Videos (MP4, MOV, M4V, WEBM, MKV, AVI, MPEG, MPG, OGV, 3GP, FLV, WMV, TS), Audio (WAV, MP3, M4A, OGG, FLAC, AAC, OPUS, WMA, WEBA), Documents (PDF, DOCX, TXT).`,
      'UNSUPPORTED_FORMAT',
      415,
    );
  }

  // Stream SHA-256 and read only the header bytes for magic signature detection
  const [sha256, headBuffer] = await Promise.all([
    computeFileSha256Streaming(filePath),
    readHeadBuffer(filePath, 65_536),
  ]);

  const detected = await fileTypeFromBuffer(headBuffer);
  const warnings: string[] = [];

  // Block executable binaries masquerading under media extensions
  if (
    detected &&
    [
      'application/x-msdownload',
      'application/x-elf',
      'application/x-mach-binary',
      'application/x-sh',
    ].includes(detected.mime)
  ) {
    throw new FileValidationError(
      'File binary signature matches an executable and was rejected for security.',
      'SECURITY_BLOCKED',
      400,
    );
  }

  let detectedMime = detected?.mime ?? '';
  if (!detectedMime) {
    detectedMime = detectContainerMimeFromHeader(headBuffer, ext);
  }

  if (!detectedMime && ext === '.txt') {
    if (!looksLikePlainText(headBuffer)) {
      throw new FileValidationError(
        'File has a .txt extension but contains binary or corrupted non-text bytes.',
        'CORRUPTED_FILE',
        400,
      );
    }
    detectedMime = 'text/plain';
  }

  if (
    ext === '.docx' &&
    (detectedMime === 'application/zip' ||
      detectedMime === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document')
  ) {
    const zipHead = headBuffer.toString('latin1');
    if (!zipHead.includes('word/') && !zipHead.includes('[Content_Types].xml')) {
      throw new FileValidationError(
        'File has a .docx extension but does not contain a valid Office Open XML structure.',
        'CORRUPTED_FILE',
        400,
      );
    }
    detectedMime = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document';
  }

  if (ext === '.pdf' || detectedMime === 'application/pdf') {
    const pdfHead = headBuffer.subarray(0, 1024).toString('latin1');
    if (!pdfHead.includes('%PDF-')) {
      throw new FileValidationError(
        'PDF file is missing a valid %PDF header and appears corrupted.',
        'CORRUPTED_FILE',
        400,
      );
    }
    detectedMime = 'application/pdf';
    if (/\/JavaScript\b|\/JS\b|\/Launch\b/i.test(headBuffer.toString('latin1'))) {
      warnings.push(
        'PDF contains active script or launch dictionary entries; inspected statically without execution.',
      );
    }
  }

  let mediaType = detectedMime ? classifyByMime(detectedMime, ext) : null;

  // Authoritative FFprobe verification for ANY video or audio candidate (resolves non-standard MP4 ftyp brands, MKV, AVI, TS, WMV, FLV, OGG, AAC, etc.)
  let probeResult: ProbeSummary | null = null;
  if (
    mediaType === 'video' ||
    mediaType === 'audio' ||
    VIDEO_EXTENSIONS.has(ext) ||
    AUDIO_EXTENSIONS.has(ext) ||
    isDeclaredVideo ||
    isDeclaredAudio
  ) {
    probeResult = await probeMediaStreams(filePath);
    if (probeResult.ok) {
      if (probeResult.hasVideo && !AUDIO_EXTENSIONS.has(ext)) {
        mediaType = 'video';
        if (!detectedMime || !VIDEO_MIMES.has(detectedMime)) {
          detectedMime = EXT_TO_FALLBACK_MIME[ext] || (isDeclaredVideo ? declaredLower : 'video/mp4');
        }
      } else if (probeResult.hasAudio) {
        mediaType = 'audio';
        if (!detectedMime || !AUDIO_MIMES.has(detectedMime)) {
          detectedMime = EXT_TO_FALLBACK_MIME[ext] || (isDeclaredAudio ? declaredLower : 'audio/mpeg');
        }
      } else if (probeResult.hasVideo) {
        // E.g. an .m4a that actually carries a video stream
        mediaType = 'video';
        detectedMime = detectedMime || 'video/mp4';
      }
    }
  }

  if (!detectedMime && !mediaType) {
    throw new FileValidationError(
      `Unsupported or corrupted format for "${originalName}" — binary container signature could not be verified.`,
      'CORRUPTED_FILE',
      400,
    );
  }

  if (!mediaType) {
    throw new FileValidationError(
      `Unsupported binary format "${detectedMime}" for "${originalName}".`,
      'UNSUPPORTED_FORMAT',
      415,
    );
  }

  // Deep decompression-bomb & container-corruption checks by media type
  if (mediaType === 'image') {
    try {
      const meta = await sharp(filePath, { limitInputPixels: MAX_IMAGE_PIXELS }).metadata();
      const pixels = (meta.width ?? 0) * (meta.height ?? 0);
      if (pixels <= 0) {
        throw new FileValidationError(
          'Image header could not be decoded — file appears corrupted.',
          'CORRUPTED_FILE',
          400,
        );
      }
      if (pixels > MAX_IMAGE_PIXELS) {
        throw new FileValidationError(
          `Image resolution (${meta.width}×${meta.height}) exceeds 100MP decompression bomb safety guard.`,
          'SECURITY_BLOCKED',
          400,
        );
      }
      await sharp(filePath, { limitInputPixels: MAX_IMAGE_PIXELS })
        .resize(16, 16, { fit: 'inside' })
        .raw()
        .toBuffer();
    } catch (err) {
      if (err instanceof FileValidationError) throw err;
      throw new FileValidationError(
        `Image file "${originalName}" failed decode verification and appears corrupted or truncated.`,
        'CORRUPTED_FILE',
        400,
      );
    }
  } else if (mediaType === 'video') {
    const check = probeResult ?? (await probeMediaStreams(filePath));
    if (!check.ok) {
      throw new FileValidationError(
        `Video file "${originalName}" could not be decoded by FFprobe (${check.errorDetail ? check.errorDetail.slice(0, 140) : 'container unreadable'}).`,
        'CORRUPTED_FILE',
        400,
      );
    }
    if (!check.hasVideo) {
      throw new FileValidationError(
        'Video container does not contain a readable video stream or appears corrupted.',
        'CORRUPTED_FILE',
        400,
      );
    }
  } else if (mediaType === 'audio') {
    const check = probeResult ?? (await probeMediaStreams(filePath));
    if (!check.ok) {
      throw new FileValidationError(
        `Audio file "${originalName}" failed FFprobe integrity check (${check.errorDetail ? check.errorDetail.slice(0, 140) : 'container unreadable'}).`,
        'CORRUPTED_FILE',
        400,
      );
    }
    if (!check.hasAudio) {
      throw new FileValidationError(
        'Audio file does not contain a decodable audio stream or appears corrupted.',
        'CORRUPTED_FILE',
        400,
      );
    }
  }

  const expectedByExt: MediaType | null = IMAGE_EXTENSIONS.has(ext)
    ? 'image'
    : VIDEO_EXTENSIONS.has(ext)
      ? 'video'
      : AUDIO_EXTENSIONS.has(ext)
        ? 'audio'
        : ext === '.pdf'
          ? 'pdf'
          : DOCUMENT_EXTENSIONS.has(ext)
            ? 'document'
            : null;

  if (expectedByExt && expectedByExt !== mediaType) {
    warnings.push(
      `Extension "${ext}" disagrees with verified binary signature "${detectedMime}". Routed by verified binary type (${mediaType}).`,
    );
  }

  return {
    originalName,
    extension: ext,
    declaredMime: declaredMime || 'application/octet-stream',
    detectedMime,
    mediaType,
    fileSizeBytes: stat.size,
    sha256,
    warnings,
  };
}
