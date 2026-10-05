import { Box, Card, SimpleGrid, Skeleton, Stack } from '@mantine/core'

/**
 * Placeholder shaped like the list it stands in for, so the page doesn't jump when items arrive.
 * `role="status"` with a label announces the load to a screen reader once instead of per skeleton.
 */
export const ItemsSkeleton = ({ variant }: { variant: 'grid' | 'table' }) => (
  <Box role="status" aria-label="Loading items" aria-busy="true">
    {variant === 'grid' ? (
      <SimpleGrid cols={{ base: 2, sm: 3, md: 4, lg: 5, xl: 6 }} spacing="sm">
        {Array.from({ length: 12 }, (_, i) => (
          // biome-ignore lint/suspicious/noArrayIndexKey: a fixed-length placeholder list, never reordered
          <Card key={i} withBorder p={0} radius="md">
            <Skeleton h={110} radius={0} />
            <Box p={6}>
              <Skeleton h={10} w="70%" />
              <Skeleton h={14} w="40%" mt={8} />
            </Box>
          </Card>
        ))}
      </SimpleGrid>
    ) : (
      <Stack gap={6}>
        {Array.from({ length: 8 }, (_, i) => (
          // biome-ignore lint/suspicious/noArrayIndexKey: a fixed-length placeholder list, never reordered
          <Skeleton key={i} h={30} />
        ))}
      </Stack>
    )}
  </Box>
)
