import type { Treaty } from '@elysia/eden'
import {
  type QueryFunction,
  type QueryKey,
  type UseMutationOptions,
  type UseQueryOptions,
  useMutation,
  useQuery,
  useQueryClient,
} from '@tanstack/react-query'

/**
 * Wrapper for `useQuery`
 */
// export function useEdenQuery<T extends Record<number, unknown> = Record<number, unknown>>(
//   queryKey: QueryKey,
//   treatyFn: () => Promise<Treaty.TreatyResponse<T>>,
//   options?: Omit<
//     UseQueryOptions<Treaty.Data<Treaty.TreatyResponse<T>>, Treaty.Error<Treaty.TreatyResponse<T>>>,
//     'queryKey' | 'queryFn'
//   >,
// ) {
//   return useQuery<Treaty.Data<Treaty.TreatyResponse<T>>, Treaty.Error<Treaty.TreatyResponse<T>>>({
//     queryKey,
//     queryFn: async () => {
//       const response = await treatyFn()

//       if (response.error) {
//         throw response.error
//       }

//       return response.data as Treaty.Data<Treaty.TreatyResponse<T>>
//     },
//     ...options,
//   })
// }

export function wrapEdenFn<T extends Record<number, unknown> = Record<number, unknown>>(
  queryFn: () => Promise<Treaty.TreatyResponse<T>>,
): QueryFunction<Treaty.Data<Treaty.TreatyResponse<T>>, QueryKey> {
  return async () => {
    const response = await queryFn()

    if (response.error) {
      throw response.error
    }

    return response.data as Treaty.Data<Treaty.TreatyResponse<T>>
  }
}

export function useEdenQuery<T extends Record<number, unknown> = Record<number, unknown>>(
  options: Omit<
    UseQueryOptions<Treaty.Data<Treaty.TreatyResponse<T>>, Treaty.Error<Treaty.TreatyResponse<T>>>,
    'queryFn'
  > & {
    queryFn: () => Promise<Treaty.TreatyResponse<T>>
  },
) {
  return useQuery<Treaty.Data<Treaty.TreatyResponse<T>>, Treaty.Error<Treaty.TreatyResponse<T>>>({
    ...options,
    queryFn: wrapEdenFn(options.queryFn),
  })
}

/**
 * Wrapper for `useMutation`
 */
export function useEdenMutation<T extends Record<number, unknown> = Record<number, unknown>, TVariables = void>(
  treatyFn: (variables: TVariables) => Promise<Treaty.TreatyResponse<T>>,
  invalidateKeys?: QueryKey[],
  options?: Omit<
    UseMutationOptions<Treaty.Data<Treaty.TreatyResponse<T>>, Treaty.Error<Treaty.TreatyResponse<T>>, TVariables>,
    'mutationFn'
  >,
) {
  const queryClient = useQueryClient()

  return useMutation<Treaty.Data<Treaty.TreatyResponse<T>>, Treaty.Error<Treaty.TreatyResponse<T>>, TVariables>({
    ...options,
    mutationFn: async (variables: TVariables) => {
      const response = await treatyFn(variables)

      if (response.error) {
        throw response.error
      }

      return response.data as Treaty.Data<Treaty.TreatyResponse<T>>
    },
    onSuccess(data, variables, context, mutation) {
      if (invalidateKeys) {
        invalidateKeys.forEach((queryKey) => {
          queryClient.invalidateQueries({ queryKey })
        })
      }

      options?.onSuccess?.(data, variables, context, mutation)
    },
  })
}
