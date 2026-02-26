/**
 * SnowflakeConnector — holds a Snowflake connection for use by the span
 * exporter.
 *
 * Mirrors the Python SnowflakeConnector from
 * trulens.connectors.snowflake.connector.
 */

import snowflake from "snowflake-sdk";

export type SnowflakeAuthenticator =
  | "SNOWFLAKE"
  | "OAUTH"
  | "OAUTH_AUTHORIZATION_CODE"
  | "OAUTH_CLIENT_CREDENTIALS"
  | "SNOWFLAKE_JWT"
  | "EXTERNALBROWSER";

export interface SnowflakeConnectionOptions {
  account: string;
  username: string;
  database: string;
  schema: string;
  warehouse?: string;
  role?: string;

  /** Authentication method. Defaults to password-based when omitted. */
  authenticator?: SnowflakeAuthenticator;

  // --- Password auth ---
  password?: string;

  // --- Key-pair auth (SNOWFLAKE_JWT) ---
  /** Mutually exclusive with password. */
  privateKey?: string;
  privateKeyPath?: string;
  /** Passphrase for an encrypted private key file. Maps to snowflake-sdk's `privateKeyPass`. */
  privateKeyPass?: string;

  // --- OAuth (OAUTH) ---
  /** Pre-obtained OAuth access token (for authenticator: "OAUTH"). */
  token?: string;

  // --- OAuth Authorization Code / Client Credentials ---
  /** OAuth client ID from Snowflake security integration. */
  oauthClientId?: string;
  /** OAuth client secret from Snowflake security integration. */
  oauthClientSecret?: string;
  /** IdP authorization endpoint (auto-derived when Snowflake is the IdP). */
  oauthAuthorizationUrl?: string;
  /** IdP token endpoint (auto-derived when Snowflake is the IdP). */
  oauthTokenRequestUrl?: string;
  /** OAuth scope (defaults to role-derived scope). */
  oauthScope?: string;
  /** Redirect URI (defaults to http://127.0.0.1:{randomPort}). */
  oauthRedirectUri?: string;
}

export class SnowflakeConnector {
  readonly options: SnowflakeConnectionOptions;
  private _connection: snowflake.Connection | null = null;
  private _connecting: Promise<snowflake.Connection> | null = null;

  constructor(options: SnowflakeConnectionOptions) {
    this.options = options;
    snowflake.configure({ logLevel: "WARN" });
  }

  /** Lazily create and return the Snowflake connection. */
  async getConnection(): Promise<snowflake.Connection> {
    if (this._connection) return this._connection;
    if (this._connecting) return this._connecting;

    this._connecting = this._connect();
    try {
      const conn = await this._connecting;
      this._connection = conn;
      return conn;
    } finally {
      this._connecting = null;
    }
  }

  private async _connect(): Promise<snowflake.Connection> {
    const o = this.options;

    const raw: Record<string, unknown> = {
      account: o.account,
      username: o.username,
      database: o.database,
      schema: o.schema,
    };

    if (o.authenticator !== undefined) raw.authenticator = o.authenticator;
    if (o.password !== undefined) raw.password = o.password;
    if (o.privateKey !== undefined) raw.privateKey = o.privateKey;
    if (o.privateKeyPath !== undefined)
      raw.privateKeyPath = o.privateKeyPath;
    if (o.privateKeyPass !== undefined)
      raw.privateKeyPass = o.privateKeyPass;
    if (o.warehouse !== undefined) raw.warehouse = o.warehouse;
    if (o.role !== undefined) raw.role = o.role;
    if (o.token !== undefined) raw.token = o.token;
    if (o.oauthClientId !== undefined) raw.oauthClientId = o.oauthClientId;
    if (o.oauthClientSecret !== undefined)
      raw.oauthClientSecret = o.oauthClientSecret;
    if (o.oauthAuthorizationUrl !== undefined)
      raw.oauthAuthorizationUrl = o.oauthAuthorizationUrl;
    if (o.oauthTokenRequestUrl !== undefined)
      raw.oauthTokenRequestUrl = o.oauthTokenRequestUrl;
    if (o.oauthScope !== undefined) raw.oauthScope = o.oauthScope;
    if (o.oauthRedirectUri !== undefined)
      raw.oauthRedirectUri = o.oauthRedirectUri;

    const needsAsync =
      o.authenticator === "OAUTH_AUTHORIZATION_CODE" ||
      o.authenticator === "EXTERNALBROWSER";

    // For browser-based SSO, use the direct console login URL rather
    // than querying Snowflake for an SSO URL (which can fail with a
    // proofKey error when no IdP is configured server-side).
    if (needsAsync && raw.disableConsoleLogin === undefined) {
      raw.disableConsoleLogin = false;
    }

    const conn = snowflake.createConnection(
      raw as unknown as snowflake.ConnectionOptions,
    );

    if (needsAsync) {
      await new Promise<void>((resolve, reject) => {
        (conn as any).connectAsync((err: Error | undefined) => {
          if (err) reject(err);
          else resolve();
        });
      });
    } else {
      await new Promise<void>((resolve, reject) => {
        conn.connect((err) => {
          if (err) reject(err);
          else resolve();
        });
      });
    }

    return conn;
  }

  /** Execute a SQL statement and return rows. */
  async execute(
    sql: string,
    binds?: unknown[]
  ): Promise<unknown[]> {
    const conn = await this.getConnection();
    return new Promise((resolve, reject) => {
      conn.execute({
        sqlText: sql,
        binds: binds as snowflake.Binds,
        complete: (err, _stmt, rows) => {
          if (err) reject(err);
          else resolve(rows ?? []);
        },
      });
    });
  }

  /** Close the underlying connection. */
  async close(): Promise<void> {
    if (!this._connection) return;
    const conn = this._connection;
    this._connection = null;
    await new Promise<void>((resolve, reject) => {
      conn.destroy((err) => {
        if (err) reject(err);
        else resolve();
      });
    });
  }
}
