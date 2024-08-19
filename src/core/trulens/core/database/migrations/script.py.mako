"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}
"""

from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

# revision identifiers, used by Alembic.
revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}

def upgrade(config) -> None:
    prefix = config.get_main_option("trulens.table_prefix")

    if prefix is None:
        raise RuntimeError("trulens.table_prefix is not set")

    # TODO: The automatically generated code below likely references
    #       tables such as "trulens_feedback_defs" or "trulens_records".
    #       However, the common prefix for these tables "trulens_" is
    #       actually configurable and so replace it with the variable
    #       prefix.
    #       e.g. replace "trulens_records" with prefix + "records".
    ${upgrades if upgrades else "pass"}


def downgrade(config) -> None:
    prefix = config.get_main_option("trulens.table_prefix")

    if prefix is None:
        raise RuntimeError("trulens.table_prefix is not set")

    # TODO: The automatically generated code below likely references
    #       tables such as "trulens_feedback_defs" or "trulens_records".
    #       However, the common prefix for these tables "trulens_" is
    #       actually configurable and so replace it with the variable
    #       prefix.
    #       e.g. replace "trulens_records" with prefix + "records".
    ${downgrades if downgrades else "pass"}
