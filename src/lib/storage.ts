/**
 * S3-compatible storage client for the ElysiaJS gateway.
 *
 * Uses @aws-sdk/client-s3 to communicate with RustFS (or any S3-compatible server).
 * Handles presigned URLs for direct browser uploads and server-side file operations.
 */

import {
  CopyObjectCommand,
  CreateBucketCommand,
  DeleteObjectCommand,
  GetObjectCommand,
  HeadBucketCommand,
  HeadObjectCommand,
  ListObjectsV2Command,
  PutObjectCommand,
  S3Client,
} from '@aws-sdk/client-s3'
import { getSignedUrl } from '@aws-sdk/s3-request-presigner'

const S3_ENDPOINT = process.env.S3_ENDPOINT ?? 'http://localhost:9000'
const S3_ACCESS_KEY = process.env.S3_ACCESS_KEY ?? 'theseus'
const S3_SECRET_KEY = process.env.S3_SECRET_KEY ?? 'theseus-secret'

// Bucket names
export const BUCKET_DATASETS = 'theseus-datasets'
export const BUCKET_MODELS = 'theseus-models'
export const BUCKET_EXPORTS = 'theseus-exports'

export const s3 = new S3Client({
  endpoint: S3_ENDPOINT,
  region: 'us-east-1',
  credentials: {
    accessKeyId: S3_ACCESS_KEY,
    secretAccessKey: S3_SECRET_KEY,
  },
  forcePathStyle: true, // Required for S3-compatible servers
})

/* ------------------------------------------------------------------ */
/*  Initialization                                                    */
/* ------------------------------------------------------------------ */

/**
 * Create required buckets if they don't exist (idempotent).
 * Call once at gateway startup.
 */
export async function ensureBuckets(): Promise<void> {
  for (const bucket of [BUCKET_DATASETS, BUCKET_MODELS, BUCKET_EXPORTS]) {
    try {
      await s3.send(new HeadBucketCommand({ Bucket: bucket }))
    } catch {
      try {
        await s3.send(new CreateBucketCommand({ Bucket: bucket }))
        console.log(`[storage] Bucket '${bucket}' created.`)
      } catch (e) {
        console.warn(`[storage] Could not create bucket '${bucket}':`, e)
      }
    }
  }
  console.log('[storage] All buckets verified.')
}

/* ------------------------------------------------------------------ */
/*  Presigned URLs                                                     */
/* ------------------------------------------------------------------ */

/**
 * Generate a presigned PUT URL for direct browser upload.
 * The browser can upload directly to S3 without proxying through the gateway.
 */
export async function getUploadUrl(
  bucket: string,
  key: string,
  contentType?: string,
  expiresIn = 3600,
): Promise<string> {
  const command = new PutObjectCommand({
    Bucket: bucket,
    Key: key,
    ...(contentType && { ContentType: contentType }),
  })
  return getSignedUrl(s3, command, { expiresIn })
}

/**
 * Generate a presigned GET URL for direct browser download.
 */
export async function getDownloadUrl(bucket: string, key: string, expiresIn = 3600): Promise<string> {
  const command = new GetObjectCommand({ Bucket: bucket, Key: key })
  return getSignedUrl(s3, command, { expiresIn })
}

/* ------------------------------------------------------------------ */
/*  Server-side file operations                                       */
/* ------------------------------------------------------------------ */

/**
 * Upload a buffer/stream to S3 from the server side.
 */
export async function uploadFile(
  bucket: string,
  key: string,
  body: Buffer | Uint8Array | ReadableStream | Blob,
  contentType?: string,
): Promise<void> {
  await s3.send(
    new PutObjectCommand({
      Bucket: bucket,
      Key: key,
      Body: body,
      ...(contentType && { ContentType: contentType }),
    }),
  )
}

/**
 * Copy a file or folder in S3.
 */
export async function copyObject(fromBucket: string, fromKey: string, toBucket: string, toKey: string): Promise<void> {
  await s3.send(
    new CopyObjectCommand({
      Bucket: toBucket,
      CopySource: `${fromBucket}/${fromKey}`,
      Key: toKey,
    }),
  )
}

/**
 * Download a file from S3 as a byte array.
 */
export async function downloadFile(bucket: string, key: string): Promise<Uint8Array> {
  const response = await s3.send(new GetObjectCommand({ Bucket: bucket, Key: key }))
  return new Uint8Array(await response.Body!.transformToByteArray())
}

/**
 * Check if a file exists in S3.
 */
export async function fileExists(bucket: string, key: string): Promise<boolean> {
  try {
    await s3.send(new HeadObjectCommand({ Bucket: bucket, Key: key }))
    return true
  } catch {
    return false
  }
}

/**
 * Delete a single file from S3.
 */
export async function deleteFile(bucket: string, key: string): Promise<void> {
  await s3.send(new DeleteObjectCommand({ Bucket: bucket, Key: key }))
}

/**
 * List all object keys under a prefix.
 */
export async function listKeys(bucket: string, prefix: string): Promise<string[]> {
  const keys: string[] = []
  let continuationToken: string | undefined

  do {
    const response = await s3.send(
      new ListObjectsV2Command({
        Bucket: bucket,
        Prefix: prefix,
        ContinuationToken: continuationToken,
      }),
    )
    for (const obj of response.Contents ?? []) {
      if (obj.Key) keys.push(obj.Key)
    }
    continuationToken = response.NextContinuationToken
  } while (continuationToken)

  return keys
}

/* ------------------------------------------------------------------ */
/*  High-level helpers                                                */
/* ------------------------------------------------------------------ */

/**
 * Copy the project dataset folder to a new folder for the training run.
 */
export async function copyDataset(projectId: string, runId: string): Promise<void> {
  const keys = await listKeys(BUCKET_DATASETS, projectId)
  for (const key of keys) {
    const newKey = `${runId}/${key}`
    await copyObject(BUCKET_DATASETS, key, BUCKET_DATASETS, newKey)
  }
}

/**
 * Upload a dataset image to S3.
 * Key format: {projectId}/{runId}/{split}/{className}/{filename}
 */
export async function uploadDatasetImage(
  projectId: string,
  filename: string,
  data: Buffer | Uint8Array | Blob,
  contentType?: string,
): Promise<string> {
  const key = `${projectId}/${filename}`
  await uploadFile(BUCKET_DATASETS, key, data, contentType)
  return key
}

/**
 * Generate a presigned download URL for a model export.
 */
export async function getExportDownloadUrl(
  runId: string,
  format: string,
  expiresIn = 3600,
): Promise<string> {
  const key = `${runId}/model.${format}`
  return getDownloadUrl(BUCKET_MODELS, key, expiresIn)
}
