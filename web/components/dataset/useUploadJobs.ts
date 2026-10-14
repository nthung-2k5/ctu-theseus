import { getJobs, subscribeJobs, type UploadJobView } from '@public/lib/upload/jobs'
import { useSyncExternalStore } from 'react'

/** The upload jobs still on screen: running ones, and finished ones whose result has not been dismissed. */
export function useUploadJobs(): UploadJobView[] {
  return useSyncExternalStore(subscribeJobs, getJobs, getJobs)
}
