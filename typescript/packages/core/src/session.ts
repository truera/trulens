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
  /** Optional app ID override. Defaults to `${appName}@${appVersion}`. */
  appId?: string;
}

let _instance: TruSession | null = null;

export class TruSession {
  readonly appName: string;
  readonly appVersion: string;
  readonly appId: string;
  private readonly provider: NodeTracerProvider;

  private constructor(options: TruSessionOptions) {
    this.appName = options.appName;
    this.appVersion = options.appVersion;
    this.appId =
      options.appId ?? `${options.appName}@${options.appVersion}`;

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
   * Calling `init()` again replaces the existing session.
   */
  static init(options: TruSessionOptions): TruSession {
    if (_instance) {
      _instance.shutdown();
    }
    _instance = new TruSession(options);
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
    await this.provider.shutdown();
    if (_instance === this) {
      _instance = null;
    }
  }
}
