import { create } from 'zustand'
import type { ProjectDetail } from './types'

interface ProjectState {
  activeProject: ProjectDetail | null
  setActiveProject: (project: ProjectDetail | null) => void
}

export const useProjectStore = create<ProjectState>((set) => ({
  activeProject: null,
  setActiveProject: (project) => set({ activeProject: project }),
}))
