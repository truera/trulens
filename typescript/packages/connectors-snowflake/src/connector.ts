/**
 * SnowflakeConnector — holds a Snowflake connection for use by the span
 * exporter.
 *
 * Mirrors the Python SnowflakeConnector from
 * trulens.connectors.snowflake.connector.
 */

import snowflake from "snowflake-sdk";

export interface SnowflakeConnectionOptions {
  account: string;
  username: string;
  password?: string;
  /** Mutually exclusive with password. */
  privateKey?: string;
  privateKeyPath?: string;
  /** Passphrase for an encrypted private key file. Maps to snowflake-sdk's `privateKeyPass`. */
  privateKeyPass?: string;
  database: string;
  schema: string;
  warehouse?: string;
  role?: string;
}

export class SnowflakeConnector {
  readonly options: SnowflakeConnectionOptions;
  private _connection: snowflake.Connection | null = null;

  constructor(options: SnowflakeConnectionOptions) {
    this.options = options;
  }

  /** Lazily create and return the Snowflake connection. */
  async getConnection(): Promise<snowflake.Connection> {
    if (this._connection) return this._connection;

    const opts: snowflake.ConnectionOptions = {
      account: this.options.account,
      username: this.options.username,
      database: this.options.database,
      schema: this.options.schema,
    };
    if (this.options.password !== undefined) opts.password = this.options.password;
    if (this.options.privateKey !== undefined) opts.privateKey = this.options.privateKey;
    if (this.options.privateKeyPath !== undefined) opts.privateKeyPath = this.options.privateKeyPath;
    if (this.options.privateKeyPass !== undefined) opts.privateKeyPass = this.options.privateKeyPass;
    if (this.options.warehouse !== undefined) opts.warehouse = this.options.warehouse;
    if (this.options.role !== undefined) opts.role = this.options.role;

    const conn = snowflake.createConnection(opts);

    await new Promise<void>((resolve, reject) => {
      conn.connect((err) => {
        if (err) reject(err);
        else resolve();
      });
    });

    this._connection = conn;
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
