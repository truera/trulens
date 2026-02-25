export { SnowflakeConnector } from "./connector.js";
export type {
  SnowflakeConnectionOptions,
  SnowflakeAuthenticator,
} from "./connector.js";

export { TruLensSnowflakeSpanExporter } from "./exporter.js";
export type { SnowflakeSpanExporterOptions } from "./exporter.js";

export { SnowflakeRunManager, RunStatus } from "./run-manager.js";
export type {
  RunManagerOptions,
  WaitForMetricsOptions,
} from "./run-manager.js";
