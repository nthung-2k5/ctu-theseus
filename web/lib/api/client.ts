import Axios, { type AxiosError, type AxiosRequestConfig } from 'axios'

/**
 * The single axios instance behind every Orval-generated call.
 *
 * Auth is two HttpOnly cookies the browser attaches itself (`withCredentials`),
 * so there is no token handling here — only the refresh dance: an access token
 * lives 15 minutes, and a 401 on any non-auth request triggers exactly one
 * `POST /api/auth/refresh` (shared by every request that 401s at the same time)
 * before replaying the original request.
 */
export const axios = Axios.create({ withCredentials: true })

// A 401 from these is a real answer (bad credentials, expired refresh), not an expired access token.
const AUTH_ENDPOINTS = ['/api/auth/login', '/api/auth/register', '/api/auth/refresh', '/api/auth/logout']

let refreshing: Promise<void> | null = null

function refreshSession(): Promise<void> {
  refreshing ??= axios
    .post('/api/auth/refresh')
    .then(() => undefined)
    .finally(() => {
      refreshing = null
    })
  return refreshing
}

type Retryable = AxiosRequestConfig & { _retried?: boolean }

axios.interceptors.response.use(undefined, async (error: AxiosError) => {
  const config = error.config as Retryable | undefined
  const isAuthCall = AUTH_ENDPOINTS.some((path) => config?.url === path)
  if (error.response?.status !== 401 || !config || config._retried || isAuthCall) throw error
  config._retried = true
  await refreshSession()
  return axios.request(config)
})

/** Orval mutator: the generated code calls this with a fully-formed request config. */
export const customInstance = <T>(config: AxiosRequestConfig, options?: AxiosRequestConfig): Promise<T> =>
  axios.request<T>({ ...config, ...options }).then(({ data }) => data)

export type ErrorType<E = unknown> = AxiosError<E>
export type BodyType<B> = B

/** The gateway's error envelope is `{ error: { code, message } }`. */
export function apiErrorMessage(error: unknown, fallback = 'Something went wrong'): string {
  const data = (error as AxiosError<{ error?: { message?: string } }> | undefined)?.response?.data
  return data?.error?.message ?? (error instanceof Error ? error.message : fallback)
}
