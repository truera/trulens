/**
 * TruSession — the main entry point to TruLens instrumentation.
 *
 * Sets up the OTEL TracerProvider and stamps every span with app_name /
 * app_version resource attributes. Mirrors the Python TruSession singleton.
 */

import { trace, type Tracer } from "@opentelemetry/api";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-http";
import { Resource } from "@opentelemetry/resources";
import {
  BatchSpanProcessor,
  NodeTracerProvider,
  type SpanExporter,
} from "@opentelemetry/sdk-trace-node";
import { ResourceAttributes } from "@trulens/semconv";

import { computeAppId } from "./app-id.js";
import type { DBConnector } from "./db-connector.js";
import { TruLensReceiver } from "./receiver.js";

/**
 * Minimal interface for OTEL instrumentations (avoids hard dep on the
 * instrumentation package from core).
 */
export interface OtelInstrumentation {
  enable(): void | Promise<void>;
  disable(): void;
  setTracerProvider(provider: NodeTracerProvider): void;
}

export interface TruSessionOptions {
  /** Name of the app being traced (shown in the TruLens dashboard). */
  appName: string;
  /** Version of the app being traced. */
  appVersion: string;
  /**
   * The span exporter to use.  Mutually exclusive with `connector`.
   *
   * - For Snowflake direct: use `new TruLensSnowflakeSpanExporter(...)` from
   *   `@trulens/connectors-snowflake`.
   * - For a custom OTLP endpoint: use `new OTLPTraceExporter(...)`.
   */
  exporter?: SpanExporter;
  /**
   * Base URL of the TruLens OTLP receiver (Python side).
   * Used for app registration before tracing begins.
   * Ignored when `connector` is provided (registration is done directly).
   * Defaults to `http://localhost:4318`.
   */
  endpoint?: string;
  /**
   * A `DBConnector` for local storage.  Mutually exclusive with `exporter`.
   *
   * When provided, `TruSession.init()` automatically starts a
   * `TruLensReceiver` on `receiverPort` and creates an OTLP exporter
   * pointing at it.  The app is registered directly via the connector.
   *
   * Example:
   * ```ts
   * import { TruSession, SQLiteConnector } from "@trulens/core";
   * const session = await TruSession.init({
   *   appName: "my-app",
   *   appVersion: "v1",
   *   connector: new SQLiteConnector(),
   * });
   * ```
   */
  connector?: DBConnector;
  /** Port for the built-in receiver (used with `connector`). Defaults to 4318. */
  receiverPort?: number;
  /** Optional app ID override. Defaults to the deterministic hash
   *  matching Python's `AppDefinition._compute_app_id`. */
  appId?: string;
  /**
   * OTEL instrumentations to register with this session's TracerProvider.
   *
   * Example:
   * ```ts
   * import { OpenAIInstrumentation } from "@trulens/instrumentation-openai";
   * await TruSession.init({
   *   ...,
   *   instrumentations: [new OpenAIInstrumentation()],
   * });
   * ```
   */
  instrumentations?: OtelInstrumentation[];
  /**
   * Name of the Snowflake Run associated with this session.
   * When set, every RECORD_ROOT span is stamped with
   * `ai.observability.run.name`.
   */
  runName?: string;
  /**
   * Called after the TracerProvider is created but before instrumentations
   * are enabled. Use this to set up external resources (e.g. create a
   * Snowflake External Agent + Run).
   */
  onInit?: () => Promise<void>;
  /**
   * Called at the beginning of `shutdown()`, after flushing spans.
   * Receives the number of records traced during this session.
   */
  onShutdown?: (inputRecordsCount: number) => Promise<void>;
}

let _instance: TruSession | null = null;

export class TruSession {
  readonly appName: string;
  readonly appVersion: string;
  readonly appId: string;
  readonly runName: string | undefined;
  private readonly provider: NodeTracerProvider;
  private readonly instrumentations: OtelInstrumentation[];
  private readonly _onShutdown:
    | ((inputRecordsCount: number) => Promise<void>)
    | undefined;
  private readonly _receiver: TruLensReceiver | null;
  private readonly _connector: DBConnector | null;

  /** Number of RECORD_ROOT spans created during this session. */
  inputRecordsCount = 0;

  private constructor(
    options: TruSessionOptions & {
      resolvedAppId: string;
      resolvedExporter: SpanExporter;
      receiver: TruLensReceiver | null;
    },
  ) {
    this.appName = options.appName;
    this.appVersion = options.appVersion;
    this.appId = options.resolvedAppId;
    this.runName = options.runName;
    this.instrumentations = options.instrumentations ?? [];
    this._onShutdown = options.onShutdown;
    this._receiver = options.receiver;
    this._connector = options.connector ?? null;

    this.provider = new NodeTracerProvider({
      resource: new Resource({
        [ResourceAttributes.APP_NAME]: this.appName,
        [ResourceAttributes.APP_VERSION]: this.appVersion,
        [ResourceAttributes.APP_ID]: this.appId,
      }),
    });

    this.provider.addSpanProcessor(
      new BatchSpanProcessor(options.resolvedExporter),
    );
    this.provider.register();
  }

  /**
   * Initialise the TruSession singleton.
   *
   * **Two modes:**
   *
   * 1. **`connector` mode** (recommended for local dev): pass a
   *    `DBConnector` (e.g. `new SQLiteConnector()`) and the session
   *    auto-starts an embedded OTLP receiver.  No Python needed for
   *    tracing.
   *
   * 2. **`exporter` mode**: pass a custom `SpanExporter` (e.g. for
   *    Snowflake direct export) and optionally an `endpoint` for app
   *    registration.
   *
   * `connector` and `exporter` are mutually exclusive.
   */
  static async init(options: TruSessionOptions): Promise<TruSession> {
    if (options.connector && options.exporter) {
      throw new Error(
        "TruSession: `connector` and `exporter` are mutually exclusive. " +
          "Provide one or the other.",
      );
    }
    if (!options.connector && !options.exporter) {
      throw new Error(
        "TruSession: either `connector` or `exporter` must be provided.",
      );
    }

    if (_instance) {
      await _instance.shutdown();
    }

    const resolvedAppId =
      options.appId ?? computeAppId(options.appName, options.appVersion);

    let resolvedExporter: SpanExporter;
    let receiver: TruLensReceiver | null = null;

    if (options.connector) {
      // ---- connector mode: start embedded receiver ----
      const port = options.receiverPort ?? 4318;
      receiver = new TruLensReceiver({
        connector: options.connector,
        port,
        host: "127.0.0.1",
      });
      await receiver.start();

      resolvedExporter = new OTLPTraceExporter({
        url: `http://127.0.0.1:${port}/v1/traces`,
      });

      // Register the app directly via the connector
      options.connector.addApp({
        appId: resolvedAppId,
        appName: options.appName,
        appVersion: options.appVersion,
        appJson: {
          app_id: resolvedAppId,
          app_name: options.appName,
          app_version: options.appVersion,
          metadata: {},
          tags: "-",
          record_ingest_mode: "immediate",
        },
      });
      console.log(
        `TruSession: registered app_id=${resolvedAppId} (via connector)`,
      );
    } else {
      // ---- exporter mode: use the provided exporter ----
      resolvedExporter = options.exporter!;

      const endpoint = options.endpoint;
      if (endpoint !== undefined) {
        await TruSession._register(
          endpoint,
          options.appName,
          options.appVersion,
        );
      }
    }

    _instance = new TruSession({
      ...options,
      resolvedAppId,
      resolvedExporter,
      receiver,
    });

    if (options.onInit) {
      await options.onInit();
    }

    for (const instr of _instance.instrumentations) {
      instr.setTracerProvider(_instance.provider);
      await instr.enable();
    }

    return _instance;
  }

  /** Return the current singleton, or throw if not yet initialised. */
  static getInstance(): TruSession {
    if (!_instance) {
      throw new Error(
        "TruSession has not been initialised. Call TruSession.init() first.",
      );
    }
    return _instance;
  }

  /** Get a tracer scoped to this session. */
  getTracer(name = "@trulens/core"): Tracer {
    return trace.getTracer(name);
  }

  /**
   * Flush pending spans, invoke onShutdown hook, shut down the
   * provider, stop the receiver (if any), and close the connector.
   */
  async shutdown(): Promise<void> {
    for (const instr of this.instrumentations) {
      instr.disable();
    }
    await this.provider.forceFlush();

    if (this._onShutdown) {
      await this._onShutdown(this.inputRecordsCount);
    }

    await this.provider.shutdown();

    if (this._receiver) {
      await this._receiver.stop();
    }
    if (this._connector) {
      this._connector.close();
    }

    if (_instance === this) {
      _instance = null;
    }
  }

  /**
   * Register the app with a remote OTLP receiver so it appears in
   * the TruLens dashboard.
   */
  private static async _register(
    endpoint: string,
    appName: string,
    appVersion: string,
  ): Promise<void> {
    const url = `${endpoint}/v1/register`;
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          app_name: appName,
          app_version: appVersion,
        }),
      });
      if (!res.ok) {
        console.warn(
          `TruSession: app registration returned ${res.status} — ` +
            "the app may not appear in the dashboard.",
        );
      } else {
        const data = (await res.json()) as { app_id?: string };
        console.log(
          `TruSession: registered app_id=${data.app_id ?? "(unknown)"}`,
        );
      }
    } catch (err) {
      console.warn(
        "TruSession: could not register app with receiver at " +
          `${url} — ${err instanceof Error ? err.message : String(err)}. ` +
          "Tracing will continue but the app may not appear in the dashboard.",
      );
    }
  }
}
