import { ImportedModelsSection } from '@public/components/models/ImportedModelsSection'
import { PageHeader } from '@public/components/ui'
import { userModelsApi } from '@public/lib/importedModels'

/**
 * "Imported models": base models the user brought themselves, from the Hugging Face Hub or as uploaded weights.
 * They are private to this account and show up in the model picker of its training runs, next to the built-in
 * base models and the ones an administrator shares with everyone.
 */
export function ImportedModelsPage() {
  return (
    <div className="flex flex-col gap-3 p-3">
      <PageHeader
        title="Imported models"
        description="Import a base model and fine-tune it on your data. Only you can use the models listed here."
      />
      <ImportedModelsSection api={userModelsApi} />
    </div>
  )
}
