CREATE TYPE "export_format" AS ENUM('onnx', 'torchscript');--> statement-breakpoint
CREATE TYPE "export_lang" AS ENUM('python', 'typescript');--> statement-breakpoint
CREATE TYPE "export_status" AS ENUM('pending', 'converting', 'assembling', 'ready', 'failed');--> statement-breakpoint
CREATE TYPE "export_tier" AS ENUM('model', 'devkit', 'app');--> statement-breakpoint
CREATE TABLE "exports" (
	"id" uuid PRIMARY KEY DEFAULT uuidv7(),
	"run_id" uuid NOT NULL,
	"user_id" uuid NOT NULL,
	"tier" "export_tier" NOT NULL,
	"format" "export_format" NOT NULL,
	"lang" "export_lang",
	"status" "export_status" DEFAULT 'pending'::"export_status" NOT NULL,
	"conversion_job_id" uuid,
	"bundle_key" text,
	"byte_size" integer,
	"checksum" char(64),
	"failed_message" text,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	"ready_at" timestamp with time zone,
	"updated_at" timestamp with time zone NOT NULL
);
--> statement-breakpoint
CREATE INDEX "exports_runId_idx" ON "exports" ("run_id");--> statement-breakpoint
CREATE INDEX "exports_conversionJobId_idx" ON "exports" ("conversion_job_id");--> statement-breakpoint
CREATE INDEX "exports_status_idx" ON "exports" ("status");--> statement-breakpoint
ALTER TABLE "exports" ADD CONSTRAINT "exports_run_id_training_runs_id_fkey" FOREIGN KEY ("run_id") REFERENCES "training_runs"("id") ON DELETE CASCADE;--> statement-breakpoint
ALTER TABLE "exports" ADD CONSTRAINT "exports_user_id_users_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE;