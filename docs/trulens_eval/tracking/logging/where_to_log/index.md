# Where to Log

By default, all data is logged to the current working directory to `default.sqlite` (`sqlite:///default.sqlite`).
Data can be logged to a SQLAlchemy-compatible referred to by `database_url` in the format `dialect+driver://username:password@host:port/database`.

See [this article](https://docs.sqlalchemy.org/en/20/core/engines.html#database-urls) for more details on SQLAlchemy database URLs.

For example, for Postgres database `trulens` running on `localhost` with username `trulensuser` and password `password` set up a connection like so.

```
from trulens_eval import Tru
tru = Tru(database_url="postgresql://trulensuser:password@localhost/trulens")
```

After which you should receive the following message:

```
🦑 Tru initialized with db url postgresql://trulensuser:password@localhost/trulens.
```
