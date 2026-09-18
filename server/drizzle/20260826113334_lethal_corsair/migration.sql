CREATE INDEX "idx_annotations_class" ON "annotations" ("class_id","annotation_type");--> statement-breakpoint
-- Nothing previously stopped an item from acquiring two classification
-- annotations (the classify route did a read-then-write with no constraint to
-- conflict on), and buildSnapshot then picked one arbitrarily. The unique index
-- below cannot be created while any duplicates remain, so collapse them first,
-- keeping the most recent label per item — the one the UI was already showing.
DELETE FROM "annotations" a
USING "annotations" b
WHERE a."annotation_type" = 'classification'
  AND b."annotation_type" = 'classification'
  AND a."item_id" = b."item_id"
  AND (a."created_at", a."id") < (b."created_at", b."id");--> statement-breakpoint
CREATE UNIQUE INDEX "annotations_item_classification_key" ON "annotations" ("item_id") WHERE "annotation_type" = 'classification';
