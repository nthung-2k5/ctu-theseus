import { writeFile } from 'node:fs/promises'
import { $ } from 'bun'
import { TrainingSchema } from './src/lib/schema'

const compileSchema = async <T>(schema: T, name: string) => {
  await writeFile(`./schema/${name}.json`, JSON.stringify(schema, null, 2))
  await $`datamodel-codegen --input schema/${name}.json --output ai_service/schema/${name}.py \
    --input-file-type jsonschema \
    --output-model-type pydantic_v2.BaseModel \
    --use-type-alias \
    --enum-field-as-literal all \
    --target-python-version 3.12 \
    --use-annotated \
    --formatters builtin \
    --naming-strategy parent-prefixed \
    --snake-case-field`
}

compileSchema(TrainingSchema, 'training')
