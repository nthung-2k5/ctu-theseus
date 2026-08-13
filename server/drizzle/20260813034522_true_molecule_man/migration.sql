CREATE TYPE "annotation_type" AS ENUM('classification', 'bounding_box', 'segmentation_mask', 'text_sequence', 'token_tags', 'preference_rank');--> statement-breakpoint
CREATE TYPE "audio_codec" AS ENUM('wav', 'mp3', 'flac', 'ogg');--> statement-breakpoint
CREATE TYPE "dataset_version_status" AS ENUM('draft', 'building', 'ready', 'failed');--> statement-breakpoint
CREATE TYPE "image_format" AS ENUM('jpeg', 'png');--> statement-breakpoint
CREATE TYPE "modality" AS ENUM('text', 'vision', 'audio', 'tabular');--> statement-breakpoint
CREATE TYPE "project_task" AS ENUM('text_classification', 'token_classification', 'text_generation', 'question_answering', 'summarization', 'sequence_to_sequence', 'text_embedding', 'image_classification', 'object_detection', 'image_segmentation', 'image_captioning', 'audio_classification', 'automatic_speech_recognition', 'audio_segmentation', 'audio_captioning', 'tabular_regression', 'tabular_classification', 'tabular_clustering', 'tabular_anomaly_detection');--> statement-breakpoint
CREATE TYPE "split_type" AS ENUM('train', 'test', 'validation');--> statement-breakpoint
CREATE TYPE "training_status" AS ENUM('queued', 'running', 'succeeded', 'failed', 'canceled');--> statement-breakpoint
CREATE TABLE "accounts" (
	"id" uuid PRIMARY KEY DEFAULT uuidv7(),
	"account_id" text NOT NULL,
	"provider_id" text NOT NULL,
	"user_id" uuid NOT NULL,
	"access_token" text,
	"refresh_token" text,
	"id_token" text,
	"access_token_expires_at" timestamp,
	"refresh_token_expires_at" timestamp,
	"scope" text,
	"password" text,
	"created_at" timestamp NOT NULL,
	"updated_at" timestamp NOT NULL
);
--> statement-breakpoint
CREATE TABLE "annotations" (
	"id" uuid PRIMARY KEY DEFAULT uuidv7(),
	"item_id" uuid NOT NULL,
	"annotator_id" varchar(100),
	"annotation_type" "annotation_type" NOT NULL,
	"class_id" uuid,
	"label_text_sequence" text,
	"label_structured" jsonb,
	"confidence_score" numeric(4,3),
	"created_at" timestamp with time zone DEFAULT now(),
	CONSTRAINT "confidence_bounds" CHECK ("confidence_score" BETWEEN 0.0 AND 1.0)
);
--> statement-breakpoint
CREATE TABLE "audio_features" (
	"item_id" uuid PRIMARY KEY,
	"duration_seconds" numeric(8,3) NOT NULL,
	"sample_rate_hz" integer NOT NULL,
	"channels" integer DEFAULT 1,
	"audio_codec" "audio_codec"
);
--> statement-breakpoint
CREATE TABLE "dataset_items" (
	"id" uuid PRIMARY KEY DEFAULT uuidv7(),
	"dataset_id" uuid NOT NULL,
	"external_id" varchar(255),
	"storage_url" text,
	"content_hash" char(64),
	"byte_size" integer,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	CONSTRAINT "dataset_items_datasetId_contentHash_key" UNIQUE("dataset_id","content_hash")
);
--> statement-breakpoint
CREATE TABLE "dataset_version_items" (
	"version_id" uuid,
	"item_id" uuid,
	"split_type" "split_type" NOT NULL,
	CONSTRAINT "dataset_version_items_pkey" PRIMARY KEY("version_id","item_id")
);
--> statement-breakpoint
CREATE TABLE "dataset_versions" (
	"id" uuid PRIMARY KEY DEFAULT uuidv7(),
	"dataset_id" uuid NOT NULL,
	"augmentation_config" jsonb,
	"version_tag" varchar(50),
	"status" "dataset_version_status" DEFAULT 'draft'::"dataset_version_status" NOT NULL,
	"item_count" integer,
	"class_count" integer,
	"parquet_key" text,
	"failed_message" text,
	"built_at" timestamp with time zone,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	CONSTRAINT "datasetVersions_datasetId_versionTag_key" UNIQUE("dataset_id","version_tag")
);
--> statement-breakpoint
CREATE TABLE "datasets" (
	"project_id" uuid PRIMARY KEY,
	"modality" "modality" NOT NULL,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	"updated_at" timestamp with time zone NOT NULL
);
--> statement-breakpoint
CREATE TABLE "label_classes" (
	"class_id" uuid PRIMARY KEY DEFAULT uuidv7(),
	"dataset_id" uuid NOT NULL,
	"name" varchar(100) NOT NULL,
	"description" text,
	"ui_color_hex" varchar(7) DEFAULT '#FFFFFF',
	"is_active" boolean DEFAULT true NOT NULL,
	"created_at" timestamp with time zone DEFAULT now(),
	CONSTRAINT "labelClasses_datasetId_name_key" UNIQUE("dataset_id","name")
);
--> statement-breakpoint
CREATE TABLE "projects" (
	"id" uuid PRIMARY KEY DEFAULT uuidv7(),
	"user_id" uuid NOT NULL,
	"name" text NOT NULL,
	"description" text DEFAULT '',
	"task" "project_task" NOT NULL,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	"updated_at" timestamp with time zone NOT NULL
);
--> statement-breakpoint
CREATE TABLE "sessions" (
	"id" uuid PRIMARY KEY DEFAULT uuidv7(),
	"expires_at" timestamp NOT NULL,
	"token" text NOT NULL UNIQUE,
	"created_at" timestamp NOT NULL,
	"updated_at" timestamp NOT NULL,
	"ip_address" text,
	"user_agent" text,
	"user_id" uuid NOT NULL,
	"impersonated_by" text
);
--> statement-breakpoint
CREATE TABLE "tabular_features" (
	"item_id" uuid PRIMARY KEY,
	"features_json" jsonb NOT NULL
);
--> statement-breakpoint
CREATE TABLE "text_features" (
	"item_id" uuid PRIMARY KEY,
	"raw_text" text NOT NULL,
	"token_count" integer,
	"language_code" varchar(10),
	"meta_json" jsonb
);
--> statement-breakpoint
CREATE TABLE "training_metrics" (
	"training_run_id" uuid,
	"epoch" integer,
	"split" "split_type",
	"metric_name" varchar(64),
	"metric_value" real NOT NULL,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	CONSTRAINT "training_metrics_pkey" PRIMARY KEY("training_run_id","epoch","split","metric_name")
);
--> statement-breakpoint
CREATE TABLE "training_runs" (
	"id" uuid PRIMARY KEY DEFAULT uuidv7(),
	"name" varchar(255) NOT NULL,
	"status" "training_status" DEFAULT 'queued'::"training_status" NOT NULL,
	"project_id" uuid NOT NULL,
	"dataset_version_id" uuid NOT NULL,
	"hyperparameters" jsonb NOT NULL,
	"ludwig_config" jsonb,
	"config_key" text,
	"best_epoch" integer,
	"heartbeat_at" timestamp with time zone,
	"started_at" timestamp with time zone,
	"completed_at" timestamp with time zone,
	"failed_message" text,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	"updated_at" timestamp with time zone NOT NULL
);
--> statement-breakpoint
CREATE TABLE "users" (
	"id" uuid PRIMARY KEY DEFAULT uuidv7(),
	"name" text NOT NULL,
	"email" text NOT NULL UNIQUE,
	"email_verified" boolean DEFAULT false NOT NULL,
	"image" text,
	"created_at" timestamp NOT NULL,
	"updated_at" timestamp NOT NULL,
	"role" text,
	"banned" boolean DEFAULT false,
	"ban_reason" text,
	"ban_expires" timestamp
);
--> statement-breakpoint
CREATE TABLE "verifications" (
	"id" uuid PRIMARY KEY DEFAULT uuidv7(),
	"identifier" text NOT NULL,
	"value" text NOT NULL,
	"expires_at" timestamp NOT NULL,
	"created_at" timestamp NOT NULL,
	"updated_at" timestamp NOT NULL
);
--> statement-breakpoint
CREATE TABLE "vision_features" (
	"item_id" uuid PRIMARY KEY,
	"width" integer NOT NULL,
	"height" integer NOT NULL,
	"channels" integer DEFAULT 3,
	"image_format" "image_format",
	"exif_data" jsonb
);
--> statement-breakpoint
CREATE INDEX "accounts_userId_idx" ON "accounts" ("user_id");--> statement-breakpoint
CREATE INDEX "idx_annotations_item" ON "annotations" ("item_id");--> statement-breakpoint
CREATE INDEX "idx_annotations_type" ON "annotations" ("annotation_type");--> statement-breakpoint
CREATE INDEX "dataset_items_datasetId_idx" ON "dataset_items" ("dataset_id");--> statement-breakpoint
CREATE INDEX "datasetVersionItems_versionId_splitType_idx" ON "dataset_version_items" ("version_id","split_type");--> statement-breakpoint
CREATE INDEX "datasetVersions_datasetId_idx" ON "dataset_versions" ("dataset_id");--> statement-breakpoint
CREATE INDEX "idx_label_classes_dataset" ON "label_classes" ("dataset_id");--> statement-breakpoint
CREATE INDEX "projects_userId_idx" ON "projects" ("user_id");--> statement-breakpoint
CREATE INDEX "sessions_userId_idx" ON "sessions" ("user_id");--> statement-breakpoint
CREATE INDEX "idx_tabular_features_gin" ON "tabular_features" USING gin ("features_json");--> statement-breakpoint
CREATE INDEX "idx_text_features_lang" ON "text_features" ("language_code");--> statement-breakpoint
CREATE INDEX "trainingRuns_projectId_idx" ON "training_runs" ("project_id");--> statement-breakpoint
CREATE INDEX "verifications_identifier_idx" ON "verifications" ("identifier");--> statement-breakpoint
ALTER TABLE "accounts" ADD CONSTRAINT "accounts_user_id_users_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE;--> statement-breakpoint
ALTER TABLE "annotations" ADD CONSTRAINT "annotations_item_id_dataset_items_id_fkey" FOREIGN KEY ("item_id") REFERENCES "dataset_items"("id") ON DELETE CASCADE;--> statement-breakpoint
ALTER TABLE "annotations" ADD CONSTRAINT "annotations_class_id_label_classes_class_id_fkey" FOREIGN KEY ("class_id") REFERENCES "label_classes"("class_id") ON DELETE RESTRICT;--> statement-breakpoint
ALTER TABLE "audio_features" ADD CONSTRAINT "audio_features_item_id_dataset_items_id_fkey" FOREIGN KEY ("item_id") REFERENCES "dataset_items"("id") ON DELETE CASCADE;--> statement-breakpoint
ALTER TABLE "dataset_items" ADD CONSTRAINT "dataset_items_dataset_id_datasets_project_id_fkey" FOREIGN KEY ("dataset_id") REFERENCES "datasets"("project_id") ON DELETE CASCADE;--> statement-breakpoint
ALTER TABLE "dataset_version_items" ADD CONSTRAINT "dataset_version_items_version_id_dataset_versions_id_fkey" FOREIGN KEY ("version_id") REFERENCES "dataset_versions"("id") ON DELETE CASCADE;--> statement-breakpoint
ALTER TABLE "dataset_version_items" ADD CONSTRAINT "dataset_version_items_item_id_dataset_items_id_fkey" FOREIGN KEY ("item_id") REFERENCES "dataset_items"("id") ON DELETE RESTRICT;--> statement-breakpoint
ALTER TABLE "dataset_versions" ADD CONSTRAINT "dataset_versions_dataset_id_datasets_project_id_fkey" FOREIGN KEY ("dataset_id") REFERENCES "datasets"("project_id") ON DELETE CASCADE;--> statement-breakpoint
ALTER TABLE "datasets" ADD CONSTRAINT "datasets_project_id_projects_id_fkey" FOREIGN KEY ("project_id") REFERENCES "projects"("id") ON DELETE CASCADE;--> statement-breakpoint
ALTER TABLE "label_classes" ADD CONSTRAINT "label_classes_dataset_id_datasets_project_id_fkey" FOREIGN KEY ("dataset_id") REFERENCES "datasets"("project_id") ON DELETE CASCADE;--> statement-breakpoint
ALTER TABLE "projects" ADD CONSTRAINT "projects_user_id_users_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE;--> statement-breakpoint
ALTER TABLE "sessions" ADD CONSTRAINT "sessions_user_id_users_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE;--> statement-breakpoint
ALTER TABLE "tabular_features" ADD CONSTRAINT "tabular_features_item_id_dataset_items_id_fkey" FOREIGN KEY ("item_id") REFERENCES "dataset_items"("id") ON DELETE CASCADE;--> statement-breakpoint
ALTER TABLE "text_features" ADD CONSTRAINT "text_features_item_id_dataset_items_id_fkey" FOREIGN KEY ("item_id") REFERENCES "dataset_items"("id") ON DELETE CASCADE;--> statement-breakpoint
ALTER TABLE "training_metrics" ADD CONSTRAINT "training_metrics_training_run_id_training_runs_id_fkey" FOREIGN KEY ("training_run_id") REFERENCES "training_runs"("id") ON DELETE CASCADE;--> statement-breakpoint
ALTER TABLE "training_runs" ADD CONSTRAINT "training_runs_project_id_projects_id_fkey" FOREIGN KEY ("project_id") REFERENCES "projects"("id") ON DELETE CASCADE;--> statement-breakpoint
ALTER TABLE "training_runs" ADD CONSTRAINT "training_runs_dataset_version_id_dataset_versions_id_fkey" FOREIGN KEY ("dataset_version_id") REFERENCES "dataset_versions"("id") ON DELETE CASCADE;--> statement-breakpoint
ALTER TABLE "vision_features" ADD CONSTRAINT "vision_features_item_id_dataset_items_id_fkey" FOREIGN KEY ("item_id") REFERENCES "dataset_items"("id") ON DELETE CASCADE;