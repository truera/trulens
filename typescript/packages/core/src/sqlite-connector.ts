/**
 * SQLiteConnector — default DBConnector backed by better-sqlite3.
 *
 * Creates the same tables as the Python TruLens ORM so that the
 * Streamlit dashboard and Python-based evaluations can read from the
 * same `.sqlite` file without migration.
 */

import Database from "better-sqlite3";

import type { AppDefinition, DBConnector, EventRecord } from "./db-connector.js";

const DEFAULT_DB_PATH = "default.sqlite";
const DEFAULT_TABLE_PREFIX = "trulens_";
const ALEMBIC_HEAD = "10";

export interface SQLiteConnectorOptions {
  /** Path to the SQLite file.  Defaults to `"default.sqlite"`. */
  dbPath?: string;
  /** Table name prefix.  Defaults to `"trulens_"`. */
  tablePrefix?: string;
}

export class SQLiteConnector implements DBConnector {
  private readonly db: Database.Database;
  private readonly prefix: string;

  private readonly insertAppStmt: Database.Statement;
  private readonly insertEventStmt: Database.Statement;

  constructor(options: SQLiteConnectorOptions = {}) {
    const dbPath = options.dbPath ?? DEFAULT_DB_PATH;
    this.prefix = options.tablePrefix ?? DEFAULT_TABLE_PREFIX;

    this.db = new Database(dbPath);
    this.db.pragma("journal_mode = WAL");

    this._createTables();

    this.insertAppStmt = this.db.prepare(`
      INSERT OR REPLACE INTO ${this.prefix}apps
        (app_id, app_name, app_version, app_json)
      VALUES (?, ?, ?, ?)
    `);

    this.insertEventStmt = this.db.prepare(`
      INSERT OR REPLACE INTO ${this.prefix}events
        (event_id, record, record_attributes, record_type,
         resource_attributes, start_timestamp, timestamp, trace)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `);
  }

  addApp(app: AppDefinition): string {
    this.insertAppStmt.run(
      app.appId,
      app.appName,
      app.appVersion,
      JSON.stringify(app.appJson),
    );
    return app.appId;
  }

  addEvents(events: EventRecord[]): string[] {
    const ids: string[] = [];
    const insertMany = this.db.transaction((evts: EventRecord[]) => {
      for (const e of evts) {
        this.insertEventStmt.run(
          e.eventId,
          JSON.stringify(e.record),
          JSON.stringify(e.recordAttributes),
          e.recordType,
          JSON.stringify(e.resourceAttributes),
          e.startTimestamp.toISOString(),
          e.timestamp.toISOString(),
          JSON.stringify(e.trace),
        );
        ids.push(e.eventId);
      }
    });
    insertMany(events);
    return ids;
  }

  close(): void {
    this.db.close();
  }

  // ------------------------------------------------------------------
  // Schema bootstrap
  // ------------------------------------------------------------------

  private _createTables(): void {
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS ${this.prefix}apps (
        app_id    VARCHAR(256) NOT NULL PRIMARY KEY,
        app_name  VARCHAR(256) NOT NULL,
        app_version VARCHAR(256) NOT NULL,
        app_json  JSON NOT NULL,
        UNIQUE(app_name, app_version)
      );
    `);

    this.db.exec(`
      CREATE TABLE IF NOT EXISTS ${this.prefix}events (
        event_id           VARCHAR(256) NOT NULL PRIMARY KEY,
        record             JSON NOT NULL,
        record_attributes  JSON NOT NULL,
        record_type        VARCHAR(256) NOT NULL,
        resource_attributes JSON NOT NULL,
        start_timestamp    TIMESTAMP NOT NULL,
        timestamp          TIMESTAMP NOT NULL,
        trace              JSON NOT NULL
      );
    `);

    this.db.exec(`
      CREATE TABLE IF NOT EXISTS ${this.prefix}records (
        record_id  VARCHAR(256) NOT NULL PRIMARY KEY,
        app_id     VARCHAR(256) NOT NULL,
        input      TEXT,
        output     TEXT,
        record_json TEXT NOT NULL,
        tags       TEXT NOT NULL,
        ts         FLOAT NOT NULL,
        cost_json  TEXT NOT NULL,
        perf_json  TEXT NOT NULL
      );
    `);

    this.db.exec(`
      CREATE TABLE IF NOT EXISTS ${this.prefix}feedback_defs (
        feedback_definition_id VARCHAR(256) NOT NULL PRIMARY KEY,
        run_location           TEXT,
        feedback_json          TEXT NOT NULL
      );
    `);

    this.db.exec(`
      CREATE TABLE IF NOT EXISTS ${this.prefix}feedbacks (
        feedback_result_id    VARCHAR(256) NOT NULL PRIMARY KEY,
        record_id             VARCHAR(256) NOT NULL,
        feedback_definition_id VARCHAR(256),
        last_ts               FLOAT NOT NULL,
        status                TEXT NOT NULL,
        error                 TEXT,
        calls_json            TEXT NOT NULL,
        result                FLOAT,
        name                  TEXT NOT NULL,
        cost_json             TEXT NOT NULL,
        multi_result          TEXT
      );
    `);

    // Stamp the alembic version so Python's TruSession won't try to
    // re-migrate when it opens the same database file.
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS ${this.prefix}alembic_version (
        version_num VARCHAR(32) NOT NULL
      );
    `);
    const row = this.db
      .prepare(
        `SELECT version_num FROM ${this.prefix}alembic_version LIMIT 1`,
      )
      .get() as { version_num: string } | undefined;
    if (!row) {
      this.db
        .prepare(
          `INSERT INTO ${this.prefix}alembic_version (version_num) VALUES (?)`,
        )
        .run(ALEMBIC_HEAD);
    }
  }
}
