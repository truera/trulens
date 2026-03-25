# trulens-connectors-snowflake

Snowflake connector for [TruLens](https://www.trulens.org/). Logs traces and evaluations to your Snowflake account.

## Install

```bash
pip install trulens-connectors-snowflake
```

## Quick Start

```python
from trulens.connectors.snowflake import SnowflakeConnector
from trulens.core import TruSession

conn = SnowflakeConnector(snowpark_session=my_existing_session)
session = TruSession(connector=conn)
```

## Authentication Methods

No password required for most auth methods. `SnowflakeConnector` supports any auth that Snowpark supports.

| Method | Key Param | Password Required? |
|---|---|---|
| Existing Snowpark Session | `snowpark_session=` | No |
| Browser SSO | `authenticator="externalbrowser"` | No |
| Key-pair auth | `private_key_file="rsa_key.p8"` | No |
| OAuth token | `authenticator="oauth", token=...` | No |
| Username / password | `password="..."` | Yes |

### Browser-based SSO

```python
conn = SnowflakeConnector(
    account="myaccount",
    user="myuser",
    authenticator="externalbrowser",
    database="mydb",
    schema="myschema",
    warehouse="mywh",
    role="myrole",
)
```

### Key-pair auth

```python
conn = SnowflakeConnector(
    account="myaccount",
    user="myuser",
    private_key_file="/path/to/rsa_key.p8",
    database="mydb",
    schema="myschema",
    warehouse="mywh",
    role="myrole",
)
```

### OAuth access token

```python
conn = SnowflakeConnector(
    account="myaccount",
    user="myuser",
    authenticator="oauth",
    token="<token>",
    database="mydb",
    schema="myschema",
    warehouse="mywh",
    role="myrole",
)
```

## Full Documentation

See the [Logging in Snowflake](https://www.trulens.org/component_guides/logging/where_to_log/log_in_snowflake/) guide for complete examples including session refresh for long-running apps and troubleshooting.
