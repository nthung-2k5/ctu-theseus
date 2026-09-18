CREATE TYPE "evaluation_split" AS ENUM('train', 'test', 'validation', 'full');--> statement-breakpoint
CREATE TYPE "evaluation_status" AS ENUM('success', 'failed');--> statement-breakpoint
CREATE TABLE "run_evaluations" (
	"run_id" uuid PRIMARY KEY,
	"status" "evaluation_status" NOT NULL,
	"split" "evaluation_split",
	"report_key" text,
	"predictions_key" text,
	"report" jsonb,
	"accuracy" real,
	"macro_f1" real,
	"failed_message" text,
	"evaluated_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
ALTER TABLE "run_evaluations" ADD CONSTRAINT "run_evaluations_run_id_training_runs_id_fkey" FOREIGN KEY ("run_id") REFERENCES "training_runs"("id") ON DELETE CASCADE;