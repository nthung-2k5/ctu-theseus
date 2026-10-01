import { CustomModelsSection } from '@public/components/models/CustomModelsSection'
import { PageHeader } from '@public/components/ui'
import { userModelsApi } from '@public/lib/customModels'

/**
 * "My models": models the user brought themselves, from the Hugging Face Hub or as uploaded weights. They are
 * private to this account and show up in the model picker of its experiments, next to the built-in models and
 * the ones an administrator shares with everyone.
 */
export function ModelsPage() {
  return (
    <div className="flex flex-col gap-3 p-3">
      <PageHeader
        title="My models"
        description="Bring your own model and train on it. Only you can use the models listed here."
      />
      <CustomModelsSection api={userModelsApi} />
    </div>
  )
}
