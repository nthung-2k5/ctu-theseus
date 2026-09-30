/**
 * PUT a file straight to a presigned object-storage URL, reporting progress.
 *
 * Deliberately NOT the app's axios instance: that one sends the session cookie and refreshes tokens on a
 * 401, neither of which belongs on a request to another origin whose URL already carries its own signature.
 * XMLHttpRequest rather than fetch because only XHR reports upload progress.
 */
export function putFile(
  url: string,
  file: File,
  headers: Record<string, string>,
  onProgress?: (fraction: number) => void,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('PUT', url)
    for (const [name, value] of Object.entries(headers)) xhr.setRequestHeader(name, value)
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress?.(event.loaded / event.total)
    }
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        onProgress?.(1)
        resolve()
      } else {
        reject(new Error(`The upload was rejected by storage (HTTP ${xhr.status})`))
      }
    }
    xhr.onerror = () => reject(new Error('The upload failed. Check your connection and try again.'))
    xhr.onabort = () => reject(new Error('The upload was cancelled'))
    xhr.send(file)
  })
}
