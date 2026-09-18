CREATE TYPE "inference_job_status" AS ENUM('pending', 'success', 'failed');--> statement-breakpoint
CREATE TABLE "inference_jobs" (
	"id" uuid PRIMARY KEY,
	"run_id" uuid NOT NULL,
	"status" "inference_job_status" DEFAULT 'pending'::"inference_job_status" NOT NULL,
	"output" jsonb,
	"error" text,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	"completed_at" timestamp with time zone
);
--> statement-breakpoint
CREATE INDEX "inferenceJobs_runId_idx" ON "inference_jobs" ("run_id");--> statement-breakpoint
CREATE INDEX "inferenceJobs_status_idx" ON "inference_jobs" ("status");--> statement-breakpoint
ALTER TABLE "inference_jobs" ADD CONSTRAINT "inference_jobs_run_id_training_runs_id_fkey" FOREIGN KEY ("run_id") REFERENCES "training_runs"("id") ON DELETE CASCADE;