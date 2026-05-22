import type { Treaty } from '@elysia/eden'
import {
  type QueryKey,
  type UseMutationOptions,
  type UseQueryOptions,
  useMutation,
  useQuery,
} from '@tanstack/react-query'

/**
 * Wrapper for `useQuery`
 */
export function useEdenQuery<T extends Record<number, unknown> = Record<number, unknown>>(
  queryKey: QueryKey,
  treatyFn: () => Promise<Treaty.TreatyResponse<T>>,
  options?: Omit<
    UseQueryOptions<Treaty.Data<Treaty.TreatyResponse<T>>, Treaty.Error<Treaty.TreatyResponse<T>>>,
    'queryKey' | 'queryFn'
  >,
) {
  return useQuery<Treaty.Data<Treaty.TreatyResponse<T>>, Treaty.Error<Treaty.TreatyResponse<T>>>({
    queryKey,
    queryFn: async () => {
      const response = await treatyFn()

      if (response.error) {
        throw response.error
      }

      return response.data as Treaty.Data<Treaty.TreatyResponse<T>>
    },
    ...options,
  })
}

/**
 * Wrapper for `useMutation`
 */
export function useEdenMutation<T extends Record<number, unknown> = Record<number, unknown>, TVariables = void>(
  treatyFn?: (variables: TVariables) => Promise<Treaty.TreatyResponse<T>>,
  options?: Omit<
    UseMutationOptions<Treaty.Data<Treaty.TreatyResponse<T>>, Treaty.Error<Treaty.TreatyResponse<T>>, TVariables>,
    'mutationFn'
  >,
) {
  return useMutation<Treaty.Data<Treaty.TreatyResponse<T>>, Treaty.Error<Treaty.TreatyResponse<T>>, TVariables>({
    mutationFn: treatyFn ? (async (variables: TVariables) => {
      const response = await treatyFn(variables)

      if (response.error) {
        throw response.error
      }

      return response.data as Treaty.Data<Treaty.TreatyResponse<T>>
    }) : undefined,
    ...options,
  })
}
