"""0002 projects documents

Revision ID: 0002_projects_documents
Revises: 15e3da35bd32
Create Date: 2026-09-16 00:00:00.000000

Xem plan6.md phần B1. Cây RAPTOR vẫn ở `.pkl`; chỉ metadata tài liệu/project
vào Postgres. Viết bằng `op.execute` với đúng SQL trong plan — enum 7 giá trị,
composite FK đích cho pgvector sau này, và `session_document_scope` cho việc
đổi tập nguồn giữa chừng.
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002_projects_documents"
down_revision: str | Sequence[str] | None = "15e3da35bd32"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "CREATE TYPE index_status AS ENUM "
        "('queued','parsing','summarizing','embedding','ready','failed','superseded')"
    )

    op.execute(
        """
        CREATE TABLE users (
          id uuid PRIMARY KEY DEFAULT uuidv7(),
          display_name text NOT NULL,
          created_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        "INSERT INTO users (id, display_name) "
        "VALUES ('00000000-0000-0000-0000-000000000001','Local') ON CONFLICT DO NOTHING"
    )

    op.execute(
        """
        CREATE TABLE projects (
          id uuid PRIMARY KEY DEFAULT uuidv7(),
          owner_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
          name text NOT NULL, language text NOT NULL DEFAULT 'vi',
          created_at timestamptz NOT NULL DEFAULT now(),
          updated_at timestamptz NOT NULL DEFAULT now(),
          deleted_at timestamptz
        )
        """
    )
    op.execute(
        "CREATE UNIQUE INDEX projects_owner_name_uniq "
        "ON projects (owner_id, name) WHERE deleted_at IS NULL"
    )

    op.execute(
        """
        CREATE TABLE documents (
          id uuid PRIMARY KEY DEFAULT uuidv7(),
          project_id uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
          ref_no int NOT NULL,
          doc_slug text NOT NULL,
          title text NOT NULL,
          source_sha1 char(40) NOT NULL,
          source_path text, source_bytes int,
          status index_status NOT NULL DEFAULT 'queued',
          active_index_id uuid,
          doc_summary text,
          outline text[], total_nodes int, num_layers smallint, total_tokens int,
          created_at timestamptz NOT NULL DEFAULT now(),
          updated_at timestamptz NOT NULL DEFAULT now(),
          deleted_at timestamptz,
          CONSTRAINT documents_pid_id_uniq  UNIQUE (project_id, id),
          CONSTRAINT documents_pid_ref_uniq UNIQUE (project_id, ref_no)
        )
        """
    )
    op.execute(
        "CREATE UNIQUE INDEX documents_project_sha1_uniq "
        "ON documents (project_id, source_sha1) WHERE deleted_at IS NULL"
    )

    op.execute(
        """
        CREATE TABLE document_indexes (
          id uuid PRIMARY KEY DEFAULT uuidv7(),
          document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
          attempt_no int NOT NULL DEFAULT 1,
          status index_status NOT NULL DEFAULT 'queued',
          source_sha1 char(40) NOT NULL,
          artifact_path text, artifact_sha1 char(40),
          total_nodes int, num_layers smallint, total_tokens int,
          build_stats jsonb,
          error_message text,
          started_at timestamptz, finished_at timestamptz,
          created_at timestamptz NOT NULL DEFAULT now(),
          CONSTRAINT document_indexes_attempt_uniq UNIQUE (document_id, attempt_no)
        )
        """
    )
    op.execute(
        "ALTER TABLE documents ADD CONSTRAINT documents_active_index_fk "
        "FOREIGN KEY (active_index_id) REFERENCES document_indexes(id) ON DELETE SET NULL"
    )

    op.execute(
        "INSERT INTO projects (id, owner_id, name) "
        "VALUES ('00000000-0000-0000-0000-000000000002',"
        "        '00000000-0000-0000-0000-000000000001','Inbox') ON CONFLICT DO NOTHING"
    )
    op.execute(
        "UPDATE sessions SET user_id='00000000-0000-0000-0000-000000000001' "
        "WHERE user_id IS NULL"
    )
    op.execute(
        "UPDATE sessions SET project_id='00000000-0000-0000-0000-000000000002' "
        "WHERE project_id IS NULL"
    )

    op.execute(
        "ALTER TABLE sessions ADD COLUMN document_id uuid "
        "REFERENCES documents(id) ON DELETE SET NULL"
    )
    op.execute(
        "ALTER TABLE sessions ADD CONSTRAINT sessions_project_fk "
        "FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE NOT VALID"
    )
    op.execute("ALTER TABLE sessions VALIDATE CONSTRAINT sessions_project_fk")
    op.execute(
        "ALTER TABLE message_citations ADD CONSTRAINT message_citations_document_fk "
        "FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE SET NULL NOT VALID"
    )
    op.execute(
        "ALTER TABLE message_citations VALIDATE CONSTRAINT message_citations_document_fk"
    )

    op.execute(
        """
        CREATE TABLE session_document_scope (
          session_id  uuid NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
          document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
          PRIMARY KEY (session_id, document_id)
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS session_document_scope")
    op.execute(
        "ALTER TABLE message_citations DROP CONSTRAINT IF EXISTS message_citations_document_fk"
    )
    op.execute("ALTER TABLE sessions DROP CONSTRAINT IF EXISTS sessions_project_fk")
    op.execute("ALTER TABLE sessions DROP COLUMN IF EXISTS document_id")
    op.execute(
        "ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_active_index_fk"
    )
    op.execute("DROP TABLE IF EXISTS document_indexes")
    op.execute("DROP INDEX IF EXISTS documents_project_sha1_uniq")
    op.execute("DROP TABLE IF EXISTS documents")
    op.execute("DROP INDEX IF EXISTS projects_owner_name_uniq")
    op.execute("DROP TABLE IF EXISTS projects")
    op.execute("DROP TABLE IF EXISTS users")
    op.execute("DROP TYPE IF EXISTS index_status")
