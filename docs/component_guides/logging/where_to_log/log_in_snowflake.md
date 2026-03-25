# ![Snowflake](../../../assets/images/logos/snowflake_logo.svg){ width="30" } Logging in Snowflake

TruLens can log traces and evaluations to a Snowflake database. This page covers every supported authentication method.

!!! tip "You don't need a password"

    `SnowflakeConnector` supports SSO, key-pair auth, OAuth tokens, and existing Snowpark sessions -- **no password required**. Password-based auth is just one option.

## Install

!!! example "Install using pip"

    ```bash
    pip install trulens-connectors-snowflake
    ```

## Auth Methods at a Glance

| Method | Key Parameter | Password Required? |
|---|---|---|
| [Browser-based SSO](#browser-based-sso) (recommended) | `authenticator="externalbrowser"` | No |
| [Key-pair authentication](#key-pair-authentication) | `private_key_file="rsa_key.p8"` | No |
| [OAuth access token](#oauth-access-token) | `authenticator="oauth", token=...` | No |
| [Existing Snowpark Session](#existing-snowpark-session) | `snowpark_session=` | No |
| [Username / password](#username-and-password) | `password="..."` | Yes |

Every method below results in a `TruSession` connected to Snowflake. Once connected, all traces and evaluations are logged automatically.

---

## Browser-Based SSO

The easiest way to get started. Pass your connection details directly to `SnowflakeConnector` -- it creates the Snowpark session for you. Your browser opens for SSO via your identity provider (Okta, Azure AD, etc.), and you're done.

!!! example "Connect with browser-based SSO"

    ```python
    from trulens.connectors.snowflake import SnowflakeConnector
    from trulens.core import TruSession

    conn = SnowflakeConnector(
        account="<account>",
        user="<user>",
        database="<database>",
        schema="<schema>",
        warehouse="<warehouse>",
        role="<role>",
        authenticator="externalbrowser",
    )
    session = TruSession(connector=conn)
    ```

---

## Key-Pair Authentication

Uses an RSA private key instead of a password. See the [Snowflake key-pair auth docs](https://docs.snowflake.com/en/user-guide/key-pair-auth) for setup instructions.

!!! example "Connect with key-pair auth"

    ```python
    from trulens.connectors.snowflake import SnowflakeConnector
    from trulens.core import TruSession

    conn = SnowflakeConnector(
        account="<account>",
        user="<user>",
        database="<database>",
        schema="<schema>",
        warehouse="<warehouse>",
        role="<role>",
        private_key_file="/path/to/rsa_key.p8",
    )
    session = TruSession(connector=conn)
    ```

??? example "Generating a key pair"

    ```bash
    # Generate an encrypted private key
    openssl genrsa 2048 | openssl pkcs8 -topk8 -inform PEM -out rsa_key.p8 -nocrypt

    # Generate the public key
    openssl rsa -in rsa_key.p8 -pubout -out rsa_key.pub

    # Assign the public key to your Snowflake user
    # In Snowflake:
    # ALTER USER <user> SET RSA_PUBLIC_KEY='<contents of rsa_key.pub without header/footer>';
    ```

---

## OAuth Access Token

Use an existing OAuth token from a token endpoint, service account flow, or Snowpark Container Services (SPCS).

!!! example "Connect with an OAuth token"

    ```python
    from trulens.connectors.snowflake import SnowflakeConnector
    from trulens.core import TruSession

    conn = SnowflakeConnector(
        account="<account>",
        user="<user>",
        database="<database>",
        schema="<schema>",
        warehouse="<warehouse>",
        role="<role>",
        authenticator="oauth",
        token="<your-oauth-token>",
    )
    session = TruSession(connector=conn)
    ```

---

## Existing Snowpark Session

If your app already has a [Snowpark session](https://docs.snowflake.com/en/developer-guide/snowpark/reference/python/latest/snowpark/api/snowflake.snowpark.Session), pass it directly. Works with **any** auth method that Snowpark supports.

!!! example "Pass an existing Snowpark session"

    ```python
    from trulens.connectors.snowflake import SnowflakeConnector
    from trulens.core import TruSession

    conn = SnowflakeConnector(snowpark_session=snowpark_session)
    session = TruSession(connector=conn)
    ```

---

## Username and Password

The traditional approach. Works but consider SSO or key-pair auth for better security.

!!! example "Connect with username and password"

    ```python
    from trulens.connectors.snowflake import SnowflakeConnector
    from trulens.core import TruSession

    conn = SnowflakeConnector(
        account="<account>",
        user="<user>",
        password="<password>",
        database="<database>",
        schema="<schema>",
        warehouse="<warehouse>",
        role="<role>",
    )
    session = TruSession(connector=conn)
    ```

---

## Troubleshooting

??? question "I see a warning about `password` being required"

    If you're using non-password auth (SSO, key-pair, OAuth), you can safely ignore this warning. Use the **Snowsight AI Observability** page for dashboards.

??? question "`paramstyle` error: pyformat vs qmark"

    The Snowpark session must use `paramstyle='qmark'`. If you created the Snowflake connection manually, pass `paramstyle='qmark'` to `snowflake.connector.connect()`.
