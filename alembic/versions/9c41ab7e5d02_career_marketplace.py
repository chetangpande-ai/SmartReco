"""career marketplace: course skills, enrollments, career profiles and plans

Revision ID: 9c41ab7e5d02
Revises: 20bba0bd2bda
Create Date: 2026-08-08 15:10:00.000000

Every create here is guarded by an existence check, which is not normal alembic practice
and is correct for this application. `main.py` calls `init_db()` — `metadata.create_all`
— on startup, so anyone who pulls this change and runs the app before migrating already
has the four new *tables* (create_all makes them) but none of the new products *columns*
(create_all never alters an existing table). Without the guards the migration then dies
on "table course_skills already exists" and leaves a half-built `_alembic_tmp_products`
behind, after which every later attempt fails with a different, more confusing error.
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = '9c41ab7e5d02'
down_revision: str | None = '20bba0bd2bda'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Marketplace metadata added to the existing catalogue. Every one is NOT NULL with a
# server_default so the backfill of existing rows happens in the same statement — an
# added-nullable-then-populated column would leave the catalogue rendering blanks in
# between, and SQLite cannot add a NOT NULL column without a default at all.
_PRODUCT_COLUMNS = (
    sa.Column('subcategory', sa.String(80), nullable=False, server_default=''),
    sa.Column('format', sa.String(40), nullable=False, server_default='Self-Paced Course'),
    sa.Column('duration_hours', sa.Integer(), nullable=False, server_default='0'),
    sa.Column('language', sa.String(24), nullable=False, server_default='English'),
    sa.Column('delivery_mode', sa.String(24), nullable=False, server_default='Self-Paced'),
    sa.Column('certificate', sa.Boolean(), nullable=False, server_default=sa.false()),
    sa.Column('instructor', sa.String(120), nullable=False, server_default=''),
    sa.Column('learners', sa.Integer(), nullable=False, server_default='0'),
    sa.Column('objectives', sa.JSON(), nullable=False, server_default='[]'),
    sa.Column('curriculum', sa.JSON(), nullable=False, server_default='[]'),
    sa.Column('projects', sa.JSON(), nullable=False, server_default='[]'),
    sa.Column('labs', sa.Integer(), nullable=False, server_default='0'),
    sa.Column('assessments', sa.Integer(), nullable=False, server_default='0'),
)


def _tables() -> set[str]:
    return set(sa.inspect(op.get_bind()).get_table_names())


def _columns(table: str) -> set[str]:
    return {c['name'] for c in sa.inspect(op.get_bind()).get_columns(table)}


def upgrade() -> None:
    # A stale temp table is the fingerprint of an earlier run that died partway through
    # a SQLite batch alter. Left in place it shadows every later attempt, and it holds
    # nothing that is not already in `products`.
    if '_alembic_tmp_products' in _tables():
        op.drop_table('_alembic_tmp_products')

    missing = [c for c in _PRODUCT_COLUMNS if c.name not in _columns('products')]
    if missing:
        with op.batch_alter_table('products', schema=None) as batch:
            for column in missing:
                batch.add_column(column)
            batch.create_index('ix_products_subcategory', ['subcategory'])
            batch.create_index('ix_products_format', ['format'])

    tables = _tables()

    if 'course_skills' not in tables:
        op.create_table(
            'course_skills',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('product_id', sa.Integer(), nullable=False),
            sa.Column('skill', sa.String(80), nullable=False),
            sa.Column('kind', sa.String(16), nullable=False, server_default='teaches'),
            sa.Column('position', sa.Integer(), nullable=False, server_default='0'),
            sa.ForeignKeyConstraint(['product_id'], ['products.id'], ondelete='CASCADE'),
            sa.PrimaryKeyConstraint('id'),
            sa.UniqueConstraint('product_id', 'skill', 'kind', name='uq_course_skill'),
        )
        op.create_index('ix_course_skills_product_id', 'course_skills', ['product_id'])
        op.create_index('ix_course_skills_skill', 'course_skills', ['skill', 'kind'])

    if 'enrollments' not in tables:
        op.create_table(
            'enrollments',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('user_id', sa.Integer(), nullable=False),
            sa.Column('product_id', sa.Integer(), nullable=False),
            sa.Column('status', sa.String(16), nullable=False, server_default='active'),
            sa.Column('progress_pct', sa.Integer(), nullable=False, server_default='0'),
            sa.Column('certificate_code', sa.String(32), nullable=False, server_default=''),
            sa.Column('started_at', sa.DateTime(timezone=True), nullable=False),
            sa.Column('last_activity_at', sa.DateTime(timezone=True), nullable=False),
            sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
            sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
            sa.ForeignKeyConstraint(['product_id'], ['products.id'], ondelete='CASCADE'),
            sa.PrimaryKeyConstraint('id'),
            sa.UniqueConstraint('user_id', 'product_id', name='uq_enrollment'),
        )
        op.create_index('ix_enrollments_user_id', 'enrollments', ['user_id'])

    if 'career_profiles' not in tables:
        op.create_table(
            'career_profiles',
            sa.Column('user_id', sa.Integer(), nullable=False),
            sa.Column('current_role', sa.String(120), nullable=False, server_default=''),
            sa.Column('target_role', sa.String(80), nullable=False, server_default=''),
            sa.Column('years_experience', sa.Integer(), nullable=False, server_default='0'),
            sa.Column('skills', sa.JSON(), nullable=False, server_default='[]'),
            sa.Column('extra_skills', sa.JSON(), nullable=False, server_default='[]'),
            sa.Column('interests', sa.String(300), nullable=False, server_default=''),
            sa.Column('weekly_hours', sa.Integer(), nullable=False, server_default='0'),
            sa.Column('level', sa.String(24), nullable=False, server_default=''),
            sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
            sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
            sa.PrimaryKeyConstraint('user_id'),
        )

    if 'career_plans' not in tables:
        op.create_table(
            'career_plans',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('user_id', sa.Integer(), nullable=False),
            sa.Column('target_role', sa.String(80), nullable=False, server_default=''),
            sa.Column('path_slug', sa.String(80), nullable=False, server_default=''),
            sa.Column('headline', sa.String(200), nullable=False, server_default=''),
            sa.Column('narrative', sa.Text(), nullable=False, server_default=''),
            sa.Column('have', sa.JSON(), nullable=False, server_default='[]'),
            sa.Column('gaps', sa.JSON(), nullable=False, server_default='[]'),
            sa.Column('stages', sa.JSON(), nullable=False, server_default='[]'),
            sa.Column('strategy', sa.String(24), nullable=False, server_default='agentic'),
            sa.Column('model', sa.String(80), nullable=False, server_default=''),
            sa.Column('readiness', sa.Float(), nullable=False, server_default='0'),
            sa.Column('is_current', sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
            sa.PrimaryKeyConstraint('id'),
        )
        op.create_index('ix_career_plans_user_id', 'career_plans', ['user_id'])
        op.create_index('ix_career_plans_created_at', 'career_plans', ['created_at'])
        op.create_index('ix_plans_user_current', 'career_plans', ['user_id', 'is_current'])


def downgrade() -> None:
    for table in ('career_plans', 'career_profiles', 'enrollments', 'course_skills'):
        if table in _tables():
            op.drop_table(table)

    present = _columns('products')
    with op.batch_alter_table('products', schema=None) as batch:
        batch.drop_index('ix_products_format')
        batch.drop_index('ix_products_subcategory')
        for column in reversed(_PRODUCT_COLUMNS):
            if column.name in present:
                batch.drop_column(column.name)
