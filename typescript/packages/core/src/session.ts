/**
 * TruSession — the main entry point to TruLens instrumentation.
 *
 * Sets up the OTEL TracerProvider and stamps every span with app_name /
 * app_version resource attributes. Mirrors the Python TruSession singleton.
 */

import { trace, type Tracer } from "@opentelemetry/api";
import { Resource } from "@opentelemetry/resources";
import {
  BatchSpanProcessor,
  NodeTracerProvider,
  type SpanExporter,
} from "@opentelemetry/sdk-trace-node";
import { ResourceAttributes } from "@trulens/semconv";

import { computeAppId } from "./app-id.js";

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
   * The span exporter to use.
   *
   * - For SQLite/Postgres/etc: use `new OTLPTraceExporter(...)` from
   *   `@opentelemetry/exporter-trace-otlp-http` pointing at a running
   *   Python `TruSession` OTLP receiver.
   * - For Snowflake direct: use `new TruLensSnowflakeSpanExporter(...)` from
   *   `@trulens/connectors-snowflake`.
   */
  exporter: SpanExporter;
  /**
   * Base URL of the TruLens OTLP receiver (Python side).
   * Used for app registration before tracing begins.
   * Defaults to `http://localhost:4318`.
   * Set to `undefined` or omit to skip registration (e.g. in tests).
   */
  endpoint?: string;
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
}

let _instance: TruSession | null = null;

export class TruSession {
  readonly appName: string;
  readonly appVersion: string;
  readonly appId: string;
  private readonly provider: NodeTracerProvider;

  private readonly instrumentations: OtelInstrumentation[];

  private constructor(
    options: TruSessionOptions & { resolvedAppId: string }
  ) {
    this.appName = options.appName;
    this.appVersion = options.appVersion;
    this.appId = options.resolvedAppId;
    this.instrumentations = options.instrumentations ?? [];

    this.provider = new NodeTracerProvider({
      resource: new Resource({
        [ResourceAttributes.APP_NAME]: this.appName,
        [ResourceAttributes.APP_VERSION]: this.appVersion,
        [ResourceAttributes.APP_ID]: this.appId,
      }),
    });

    this.provider.addSpanProcessor(
      new BatchSpanProcessor(options.exporter)
    );
    this.provider.register();
  }

  /**
   * Initialise the TruSession singleton.
   *
   * Registers the app with the Python TruLens receiver (if `endpoint`
   * is provided) before setting up the OTEL tracer, so the dashboard
   * can discover the app.
   *
   * Calling `init()` again replaces the existing session.
   */
  static async init(options: TruSessionOptions): Promise<TruSession> {
    if (_instance) {
      await _instance.shutdown();
    }

    const resolvedAppId =
      options.appId ?? computeAppId(options.appName, options.appVersion);

    const endpoint = options.endpoint;
    if (endpoint !== undefined) {
      await TruSession._register(
        endpoint,
        options.appName,
        options.appVersion
      );
    }

    _instance = new TruSession({ ...options, resolvedAppId });

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
        "TruSession has not been initialised. Call TruSession.init() first."
      );
    }
    return _instance;
  }

  /** Get a tracer scoped to this session. */
  getTracer(name = "@trulens/core"): Tracer {
    return trace.getTracer(name);
  }

  /** Flush pending spans and shut down the provider. */
  async shutdown(): Promise<void> {
    for (const instr of this.instrumentations) {
      instr.disable();
    }
    await this.provider.shutdown();
    if (_instance === this) {
      _instance = null;
    }
  }

  /**
   * Register the app with the Python OTLP receiver so it appears in
   * the TruLens dashboard.
   */
  private static async _register(
    endpoint: string,
    appName: string,
    appVersion: string
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
            "the app may not appear in the dashboard."
        );
      } else {
        const data = (await res.json()) as { app_id?: string };
        console.log(
          `TruSession: registered app_id=${data.app_id ?? "(unknown)"}`
        );
      }
    } catch (err) {
      console.warn(
        "TruSession: could not register app with receiver at " +
          `${url} — ${err instanceof Error ? err.message : String(err)}. ` +
          "Tracing will continue but the app may not appear in the dashboard."
      );
    }
  }
}
