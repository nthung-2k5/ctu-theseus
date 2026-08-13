/**
 * S3-compatible storage client for the ElysiaJS gateway.
 *
 * Handles presigned URLs for direct browser uploads and server-side file operations.
 */

import {
  CopyObjectCommand,
  CreateBucketCommand,
  DeleteObjectCommand,
  GetObjectCommand,
  HeadBucketCommand,
  HeadObjectCommand,
  type HeadObjectCommandOutput,
  ListObjectsV2Command,
  PutObjectCommand,
  S3Client,
} from '@aws-sdk/client-s3'
import { getSignedUrl } from '@aws-sdk/s3-request-presigner'
import CONSTANTS from '@schema/constants.json'
import { config } from '@server/lib/config'

export const s3 = new S3Client({
  endpoint: config.s3Endpoint,
  region: 'us-east-1',
  credentials: {
    accessKeyId: config.s3AccessKey,
    secretAccessKey: config.s3SecretKey,
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
  for (const bucket of [CONSTANTS.BUCKET_DATASETS, CONSTANTS.BUCKET_TRAINING, CONSTANTS.BUCKET_MODELS]) {
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
 * Copy a file in S3.
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

export async function copyFolder(
  sourceBucket: string,
  sourcePrefix: string,
  targetBucket: string,
  targetPrefix: string,
): Promise<void> {
  // Ensure prefixes end with a trailing slash
  const src = sourcePrefix.endsWith('/') ? sourcePrefix : `${sourcePrefix}/`
  const dst = targetPrefix.endsWith('/') ? targetPrefix : `${targetPrefix}/`

  let isTruncated: boolean | undefined = true
  let continuationToken: string | undefined

  try {
    while (isTruncated) {
      const listCommand: ListObjectsV2Command = new ListObjectsV2Command({
        Bucket: sourceBucket,
        Prefix: src,
        ContinuationToken: continuationToken,
      })

      const listResponse = await s3.send(listCommand)

      if (!listResponse.Contents || listResponse.Contents.length === 0) {
        console.log('No files found in the source directory.')
        break
      }

      const copyPromises = listResponse.Contents.map((object) => {
        const sourceKey = object.Key
        // Generate the new target key by replacing the source prefix with target prefix
        const targetKey = sourceKey?.replace(src, dst)

        const copyCommand = new CopyObjectCommand({
          Bucket: targetBucket,
          Key: targetKey,
          // CopySource format must be: /bucket-name/path/to/object
          CopySource: encodeURIComponent(`/${sourceBucket}/${sourceKey}`),
        })

        console.log(`Copying: ${sourceKey} -> ${targetKey}`)
        return s3.send(copyCommand)
      })

      // Execute current batch of copies in parallel
      await Promise.all(copyPromises)

      // Check if more files remain (S3 lists up to 1000 items per request)
      isTruncated = listResponse.IsTruncated
      continuationToken = listResponse.NextContinuationToken
    }

    console.log('Folder copy completed successfully!')
  } catch (error) {
    console.error('Error copying folder:', error)
    throw error
  }
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
export async function fileExists(bucket: string, key: string): Promise<HeadObjectCommandOutput | null> {
  try {
    const response = await s3.send(new HeadObjectCommand({ Bucket: bucket, Key: key }))
    return response
  } catch {
    return null
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
/*  Canonical S3 key builders                                         */
/*                                                                     */
/*  theseus-datasets/                                                 */
/*    pool/{projectId}/{hash[0:2]}/{hash}{ext}   content-addressed,   */
/*                                                mutable, deduped    */
/*    snapshots/{versionId}/dataset.parquet      immutable            */
/*    snapshots/{versionId}/manifest.json                             */
/*                                                                     */
/*  theseus-training/                                                 */
/*    {runId}/config.yaml                        compiled Ludwig cfg  */
/*    {runId}/results/                           Ludwig output dir    */
/*    {runId}/logs/train.log                                          */
/*                                                                     */
/*  theseus-models/                                                   */
/*    {runId}/model.{onnx|pt2|safetensors}       exports only         */
/* ------------------------------------------------------------------ */

export function poolKey(projectId: string, hash: string, ext: string): string {
  return `pool/${projectId}/${hash.slice(0, 2)}/${hash}${ext}`
}

export function snapshotParquetKey(versionId: string): string {
  return `snapshots/${versionId}/${CONSTANTS.DATASET_VERSION_FILENAME}`
}

export function snapshotManifestKey(versionId: string): string {
  return `snapshots/${versionId}/manifest.json`
}

export function trainingConfigKey(runId: string): string {
  return `${runId}/${CONSTANTS.TRAINING_CONFIG_FILENAME}`
}

export function trainingResultsPrefix(runId: string): string {
  return `${runId}/results/`
}

export function trainingLogsKey(runId: string): string {
  return `${runId}/logs/train.log`
}

export function exportKey(runId: string, format: string): string {
  return `${runId}/model.${format}`
}

/**
 * Upload bytes to the project's content-addressed pool. Hashes the content
 * first and skips the upload entirely if that hash already exists for this
 * project — this is what makes the pool deduplicated.
 */
export async function uploadToPool(
  projectId: string,
  data: Uint8Array,
  ext: string,
  contentType?: string,
): Promise<{ key: string; hash: string; byteSize: number; isDuplicate: boolean }> {
  const hash = new Bun.CryptoHasher('sha256').update(data).digest('hex')
  const key = poolKey(projectId, hash, ext)

  const existing = await fileExists(CONSTANTS.BUCKET_DATASETS, key)
  if (!existing) {
    await uploadFile(CONSTANTS.BUCKET_DATASETS, key, data, contentType)
  }

  return { key, hash, byteSize: data.byteLength, isDuplicate: existing !== null }
}

/**
 * Generate a presigned download URL for a model export.
 */
export async function getExportDownloadUrl(runId: string, format: string, expiresIn = 3600): Promise<string> {
  return getDownloadUrl(CONSTANTS.BUCKET_MODELS, exportKey(runId, format), expiresIn)
}
