"""add_historical_import_and_audit_schema

Revision ID: 8f853c4d65c8
Revises: 74771e86d71b
Create Date: 2026-10-29 16:31:05.124200

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '8f853c4d65c8'
down_revision: Union[str, Sequence[str], None] = '74771e86d71b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('dataset_import_logs',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('import_batch_id', sa.String(length=64), nullable=False),
        sa.Column('source_file', sa.String(length=255), nullable=False),
        sa.Column('file_sha256', sa.String(length=64), nullable=False),
        sa.Column('file_size_bytes', sa.BigInteger(), nullable=False),
        sa.Column('mode', sa.String(length=30), nullable=False),
        sa.Column('status', sa.String(length=30), nullable=False),
        sa.Column('total_rows_read', sa.Integer(), nullable=False),
        sa.Column('rows_accepted', sa.Integer(), nullable=False),
        sa.Column('rows_inserted', sa.Integer(), nullable=False),
        sa.Column('rows_skipped_duplicate', sa.Integer(), nullable=False),
        sa.Column('rows_rejected', sa.Integer(), nullable=False),
        sa.Column('validation_summary', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('error_details', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("mode IN ('dry_run', 'sample', 'full_import')", name='ck_dataset_import_logs_mode'),
        sa.CheckConstraint("status IN ('started', 'completed', 'failed', 'completed_with_warnings')", name='ck_dataset_import_logs_status'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_dataset_import_logs_import_batch_id'), 'dataset_import_logs', ['import_batch_id'], unique=True)
    
    op.add_column('incidents', sa.Column('root_cause_description', sa.Text(), nullable=True))
    op.add_column('incidents', sa.Column('source_row_hash', sa.String(length=64), nullable=True))
    
    op.create_index('ix_incidents_equipment_type', 'incidents', ['equipment_type'], unique=False)
    op.create_index('ix_incidents_incident_type', 'incidents', ['incident_type'], unique=False)
    op.create_index('ix_incidents_source_date', 'incidents', ['record_source', 'incident_date'], unique=False)
    op.create_index(op.f('ix_incidents_source_row_hash'), 'incidents', ['source_row_hash'], unique=False)
    op.create_index('uq_incidents_historical_row_hash', 'incidents', ['source_row_hash'], unique=True, postgresql_where=sa.text("record_source = 'historical_dataset'"))
    
    # Partition views for convenient and segregated querying
    op.execute("""
        CREATE OR REPLACE VIEW v_historical_incidents AS
        SELECT * FROM incidents WHERE record_source = 'historical_dataset';
    """)
    op.execute("""
        CREATE OR REPLACE VIEW v_user_incidents AS
        SELECT * FROM incidents WHERE record_source = 'user_submission';
    """)


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP VIEW IF EXISTS v_historical_incidents;")
    op.execute("DROP VIEW IF EXISTS v_user_incidents;")
    
    op.drop_index('uq_incidents_historical_row_hash', table_name='incidents', postgresql_where=sa.text("record_source = 'historical_dataset'"))
    op.drop_index(op.f('ix_incidents_source_row_hash'), table_name='incidents')
    op.drop_index('ix_incidents_source_date', table_name='incidents')
    op.drop_index('ix_incidents_incident_type', table_name='incidents')
    op.drop_index('ix_incidents_equipment_type', table_name='incidents')
    op.drop_column('incidents', 'source_row_hash')
    op.drop_column('incidents', 'root_cause_description')
    op.drop_index(op.f('ix_dataset_import_logs_import_batch_id'), table_name='dataset_import_logs')
    op.drop_table('dataset_import_logs')
