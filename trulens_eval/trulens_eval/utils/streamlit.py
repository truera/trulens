import argparse
import sys

from trulens_eval.database import base as mod_db
from trulens_eval.tru import Tru


def init_from_args():
    """Parse command line arguments and initialize Tru with them.
    
    As Tru is a singleton, further Tru() uses will get the same configuration.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url", default=None)
    parser.add_argument(
        "--database-prefix", default=mod_db.DEFAULT_DATABASE_PREFIX
    )

    try:
        args = parser.parse_args()
    except SystemExit as e:
        print(e)

        # This exception will be raised if --help or invalid command line arguments
        # are used. Currently, streamlit prevents the program from exiting normally,
        # so we have to do a hard exit.
        sys.exit(e.code)

    Tru(database_url=args.database_url, database_prefix=args.database_prefix)
