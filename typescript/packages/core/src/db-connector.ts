/**
 * DBConnector — pluggable database interface for TruLens TypeScript.
 *
 * Mirrors the Python `DBConnector` from
 * `trulens.core.database.connector.base` so that the receiver and session
 * are DB-agnostic.  The default implementation is `SQLiteConnector`;
 * alternative backends (Postgres, etc.) can implement this interface.
 */

export interface AppDefinition {
  appId: string;
  appName: string;
  appVersion: string;
  appJson: Record<string, unknown>;
}

/**
 * A single span event to be persisted.  Field names map 1:1 to the
 * `trulens_events` table columns created by Python migration 10.
 */
export interface EventRecord {
  eventId: string;
  record: Record<string, unknown>;
  recordAttributes: Record<string, unknown>;
  recordType: "SPAN";
  resourceAttributes: Record<string, unknown>;
  startTimestamp: Date;
  timestamp: Date;
  trace: Record<string, unknown>;
}

/**
 * Minimal DB connector interface used by `TruLensReceiver` and `TruSession`.
 *
 * Intentionally narrow — only the methods the TypeScript tracing path
 * needs.  Python's dashboard / eval layer reads from the same DB via
 * its own full `DBConnector`.
 */
export interface DBConnector {
  /** Insert or update an app definition.  Returns the app ID. */
  addApp(app: AppDefinition): string;

  /** Batch-insert span events.  Returns the list of event IDs. */
  addEvents(events: EventRecord[]): string[];

  /** Close the underlying connection / release resources. */
  close(): void;
}
