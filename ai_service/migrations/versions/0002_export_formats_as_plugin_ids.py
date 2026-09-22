"""export formats as plugin ids

Revision ID: 0002
Revises: 0001

exports.(tier, format, lang) collapse into one free-text `format` holding an export format plugin
id, so adding a format is a Python class and never another enum migration.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("exports", sa.Column("format_id", sa.String(length=64), nullable=True))
    op.execute(
        """
        UPDATE exports SET format_id = CASE
            WHEN tier = 'model' AND format = 'onnx' THEN 'onnx'
            WHEN tier = 'model' THEN 'torch_export'
            WHEN tier = 'devkit' THEN lang::text || '_devkit'
            ELSE lang::text || '_app'
        END
        """
    )
    op.alter_column("exports", "format_id", nullable=False)
    op.drop_column("exports", "format")
    op.drop_column("exports", "tier")
    op.drop_column("exports", "lang")
    op.alter_column("exports", "format_id", new_column_name="format")
    op.execute("DROP TYPE export_format")
    op.execute("DROP TYPE export_tier")
    op.execute("DROP TYPE export_lang")


def downgrade() -> None:
    # Formats that did not exist before the plugin refactor have no old-schema equivalent.
    op.execute(
        """
        DELETE FROM exports WHERE format NOT IN (
            'onnx', 'torch_export', 'python_devkit', 'typescript_devkit', 'csharp_devkit', 'java_devkit',
            'pwa_app', 'flutter_app'
        )
        """
    )
    export_tier = postgresql.ENUM("model", "devkit", "app", name="export_tier")
    export_format = postgresql.ENUM("onnx", "torchscript", name="export_format")
    export_lang = postgresql.ENUM("python", "typescript", "csharp", "java", "pwa", "flutter", name="export_lang")
    for enum in (export_tier, export_format, export_lang):
        enum.create(op.get_bind(), checkfirst=False)
    op.alter_column("exports", "format", new_column_name="format_id")
    op.add_column("exports", sa.Column("tier", export_tier, nullable=True))
    op.add_column("exports", sa.Column("format", export_format, nullable=True))
    op.add_column("exports", sa.Column("lang", export_lang, nullable=True))
    op.execute(
        """
        UPDATE exports SET
            tier = CASE
                WHEN format_id LIKE '%\\_devkit' THEN 'devkit'
                WHEN format_id LIKE '%\\_app' THEN 'app'
                ELSE 'model'
            END::export_tier,
            format = CASE WHEN format_id = 'torch_export' THEN 'torchscript' ELSE 'onnx' END::export_format,
            lang = CASE
                WHEN format_id LIKE '%\\_devkit' THEN replace(format_id, '_devkit', '')
                WHEN format_id LIKE '%\\_app' THEN replace(format_id, '_app', '')
            END::export_lang
        """
    )
    op.alter_column("exports", "tier", nullable=False)
    op.alter_column("exports", "format", nullable=False)
    op.drop_column("exports", "format_id")
