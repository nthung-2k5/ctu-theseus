CREATE TYPE "sweep_status" AS ENUM('running', 'completed', 'canceled');--> statement-breakpoint
CREATE TYPE "sweep_strategy" AS ENUM('grid', 'random');--> statement-breakpoint
CREATE TABLE "sweeps" (
	"id" uuid PRIMARY KEY DEFAULT uuidv7(),
	"project_id" uuid NOT NULL,
	"dataset_version_id" uuid NOT NULL,
	"name" varchar(255) NOT NULL,
	"search_space" jsonb NOT NULL,
	"strategy" "sweep_strategy" NOT NULL,
	"max_trials" integer NOT NULL,
	"status" "sweep_status" DEFAULT 'running'::"sweep_status" NOT NULL,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	"updated_at" timestamp with time zone NOT NULL
);
--> statement-breakpoint
ALTER TABLE "training_runs" ADD COLUMN "sweep_id" uuid;--> statement-breakpoint
ALTER TABLE "training_runs" ADD COLUMN "trial_index" integer;--> statement-breakpoint
CREATE INDEX "sweeps_projectId_idx" ON "sweeps" ("project_id");--> statement-breakpoint
CREATE INDEX "trainingRuns_sweepId_idx" ON "training_runs" ("sweep_id");--> statement-breakpoint
ALTER TABLE "sweeps" ADD CONSTRAINT "sweeps_project_id_projects_id_fkey" FOREIGN KEY ("project_id") REFERENCES "projects"("id") ON DELETE CASCADE;--> statement-breakpoint
ALTER TABLE "sweeps" ADD CONSTRAINT "sweeps_dataset_version_id_dataset_versions_id_fkey" FOREIGN KEY ("dataset_version_id") REFERENCES "dataset_versions"("id") ON DELETE CASCADE;--> statement-breakpoint
ALTER TABLE "training_runs" ADD CONSTRAINT "training_runs_sweep_id_sweeps_id_fkey" FOREIGN KEY ("sweep_id") REFERENCES "sweeps"("id") ON DELETE CASCADE;