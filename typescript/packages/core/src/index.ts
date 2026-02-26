export { TruSession } from "./session.js";
export type { TruSessionOptions, OtelInstrumentation } from "./session.js";
export { computeAppId } from "./app-id.js";
export { createTruApp } from "./app.js";
export type { TruAppOptions } from "./app.js";

export {
  instrument,
  instrumentDecorator,
  withRecord,
} from "./instrument.js";
export type {
  AttributeResolver,
  InstrumentOptions,
  WithRecordOptions,
} from "./instrument.js";

// DB connector interface + implementations
export type { DBConnector, AppDefinition, EventRecord } from "./db-connector.js";
export { SQLiteConnector } from "./sqlite-connector.js";
export type { SQLiteConnectorOptions } from "./sqlite-connector.js";

// OTLP receiver
export { TruLensReceiver } from "./receiver.js";
export type { TruLensReceiverOptions } from "./receiver.js";

// Re-export semconv for convenience so users only need @trulens/core
export { ResourceAttributes, SpanAttributes, SpanType } from "@trulens/semconv";
