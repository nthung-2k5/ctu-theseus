import YAML from 'yaml'
import type { LudwigConfig } from './schema'

/** Serialize a compiled Ludwig config to YAML for config.yaml. */
export function serializeLudwigConfig(config: LudwigConfig): string {
  return YAML.stringify(config)
}
