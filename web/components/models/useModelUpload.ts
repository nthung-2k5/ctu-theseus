import { CUSTOM_MODELS_KEY, type ModelsApi } from '@public/lib/customModels'
import { putFile } from '@public/lib/upload'
import { useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'

/**
 * Upload a model's weights: ask the API for a presigned URL, PUT the file straight to object storage
 * (never through the API process, since these files are gigabytes), then tell the API it is there so
 * validation starts. `progress` is 0..1 while a file is in flight, otherwise null.
 */
export function useModelUpload(api: ModelsApi) {
  const queryClient = useQueryClient()
  const [progress, setProgress] = useState<number | null>(null)

  const upload = async (modelId: string, file: File) => {
    setProgress(0)
    try {
      const target = await api.uploadUrl(modelId, { filename: file.name, sizeBytes: file.size })
      await putFile(target.url, file, target.headers, setProgress)
      await api.finalize(modelId)
    } finally {
      setProgress(null)
      queryClient.invalidateQueries({ queryKey: CUSTOM_MODELS_KEY })
    }
  }

  return { upload, progress }
}
