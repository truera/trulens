/**
 * TruLensSnowflakeSpanExporter — exports OTEL spans directly to Snowflake's
 * AI Observability stage via a stored procedure call.
 *
 * Mirrors the Python TruLensSnowflakeSpanExporter from
 * trulens.connectors.snowflake.otel_exporter.
 *
 * Flow:
 *   1. Serialise spans to OTLP protobuf (length-delimited records).
 *   2. Upload the .pb file to a Snowflake temp stage.
 *   3. Call SYSTEM$EXECUTE_AI_OBSERVABILITY_RUN to ingest.
 */

import * as fs from "fs";
import * as os from "os";
import * as path from "path";

import { ExportResultCode, type ExportResult } from "@opentelemetry/core";
import type { ReadableSpan, SpanExporter } from "@opentelemetry/sdk-trace-node";
import { ResourceAttributes, SpanAttributes } from "@trulens/semconv";

import type { SnowflakeConnector } from "./connector.js";

const STAGE_NAME = "trulens_spans";

export interface SnowflakeSpanExporterOptions {
  connector: SnowflakeConnector;
  /**
   * When true, performs the full export pipeline but skips the final
   * SYSTEM$EXECUTE_AI_OBSERVABILITY_RUN call. Useful for testing.
   */
  dryRun?: boolean;
}

export class TruLensSnowflakeSpanExporter implements SpanExporter {
  private readonly connector: SnowflakeConnector;
  private readonly dryRun: boolean;
  private enabled = true;
  /** Persistent cache so RECORD_ID / RUN_NAME propagate across batches. */
  private readonly _traceAttrCache = new Map<
    string,
    { recordId?: string; runName?: string }
  >();

  constructor(options: SnowflakeSpanExporterOptions) {
    this.connector = options.connector;
    this.dryRun = options.dryRun ?? false;
  }

  disable(): void {
    this.enabled = false;
  }

  async export(
    spans: ReadableSpan[],
    resultCallback: (result: ExportResult) => void
  ): Promise<void> {
    if (!this.enabled) {
      resultCallback({ code: ExportResultCode.SUCCESS });
      return;
    }

    const trulensSpans = this.dryRun ? spans : spans.filter(isTruLensSpan);
    const enriched = propagateTraceAttrs(trulensSpans, this._traceAttrCache);

    try {
      await this._exportToSnowflake(enriched);
      resultCallback({ code: ExportResultCode.SUCCESS });
    } catch (err) {
      resultCallback({
        code: ExportResultCode.FAILED,
        error: err instanceof Error ? err : new Error(String(err)),
      });
    }
  }

  async shutdown(): Promise<void> {
    this.enabled = false;
  }

  // ---------------------------------------------------------------------------
  // Private helpers
  // ---------------------------------------------------------------------------

  private async _exportToSnowflake(spans: ReadableSpan[]): Promise<void> {
    if (spans.length === 0) return;

    // Group by (app_name, app_version, run_name, input_records_count).
    const groups = new Map<
      string,
      {
        appName: string;
        appVersion: string;
        runName: string | undefined;
        inputRecordsCount: number | undefined;
        spans: ReadableSpan[];
      }
    >();

    for (const span of spans) {
      const appName = String(
        span.resource.attributes[ResourceAttributes.APP_NAME] ?? ""
      );
      const appVersion = String(
        span.resource.attributes[ResourceAttributes.APP_VERSION] ?? ""
      );
      const runName = span.attributes[SpanAttributes.RUN_NAME] as
        | string
        | undefined;
      const inputRecordsCount = span.attributes[
        SpanAttributes.INPUT_RECORDS_COUNT
      ] as number | undefined;

      const key = JSON.stringify([
        appName,
        appVersion,
        runName,
        inputRecordsCount,
      ]);
      if (!groups.has(key)) {
        groups.set(key, {
          appName,
          appVersion,
          runName,
          inputRecordsCount,
          spans: [],
        });
      }
      groups.get(key)!.spans.push(span);
    }

    for (const group of groups.values()) {
      await this._exportGroup(group);
    }
  }

  private async _exportGroup(group: {
    appName: string;
    appVersion: string;
    runName: string | undefined;
    inputRecordsCount: number | undefined;
    spans: ReadableSpan[];
  }): Promise<void> {
    const tmpPath = await this._writeSpansToTempFile(group.spans);
    const tmpBasename = path.basename(tmpPath);

    try {
      await this._uploadToStage(tmpPath);

      if (!this.dryRun) {
        await this._callIngestSproc(
          tmpBasename,
          group.appName,
          group.appVersion,
          group.runName,
          group.inputRecordsCount
        );
      }
    } finally {
      try {
        fs.unlinkSync(tmpPath);
      } catch {
        // Best-effort cleanup; don't fail the export.
      }
    }
  }

  private async _writeSpansToTempFile(
    spans: ReadableSpan[]
  ): Promise<string> {
    const tmpPath = path.join(os.tmpdir(), `trulens_spans_${Date.now()}.pb`);

    // Mirrors the Python serialisation: each span is individually
    // encoded as a protobuf `opentelemetry.proto.trace.v1.Span` message
    // preceded by a varint length delimiter.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const traceInternal: any = await import(
      "@opentelemetry/otlp-transformer/build/esm/trace/internal.js"
    );
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const commonUtils: any = await import(
      "@opentelemetry/otlp-transformer/build/esm/common/utils.js"
    );
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const rootMod: any = await import(
      "@opentelemetry/otlp-transformer/build/esm/generated/root.js"
    );

    const sdkSpanToOtlpSpan = traceInternal.sdkSpanToOtlpSpan;
    const getOtlpEncoder = commonUtils.getOtlpEncoder;
    const rootObj = rootMod.default ?? rootMod;
    const SpanProto = rootObj.opentelemetry.proto.trace.v1.Span;
    const encoder = getOtlpEncoder();

    const buffers: Buffer[] = [];
    for (const span of spans) {
      const otlpSpan = sdkSpanToOtlpSpan(span, encoder);
      const encoded = SpanProto.encode(
        SpanProto.fromObject(otlpSpan)
      ).finish();
      buffers.push(encodeVarint(encoded.byteLength));
      buffers.push(Buffer.from(encoded));
    }

    fs.writeFileSync(tmpPath, Buffer.concat(buffers));
    return tmpPath;
  }

  private async _uploadToStage(tmpPath: string): Promise<void> {
    await this.connector.execute(
      `CREATE TEMP STAGE IF NOT EXISTS ${STAGE_NAME}`
    );
    await this.connector.execute(
      `PUT file://${tmpPath} @${STAGE_NAME}`
    );
  }

  private async _callIngestSproc(
    tmpBasename: string,
    appName: string,
    appVersion: string,
    runName: string | undefined,
    inputRecordsCount: number | undefined
  ): Promise<void> {
    // Mirrors the Python SYSTEM$EXECUTE_AI_OBSERVABILITY_RUN call.
    const rows = await this.connector.execute(
      `SELECT CURRENT_DATABASE(), CURRENT_SCHEMA()`
    ) as Array<Record<string, string>>;

    const database = rows[0]?.["CURRENT_DATABASE()"] ?? "";
    const schema = rows[0]?.["CURRENT_SCHEMA()"] ?? "";

    await this.connector.execute(
      `
      CALL SYSTEM$EXECUTE_AI_OBSERVABILITY_RUN(
        OBJECT_CONSTRUCT(
          'object_name', ?,
          'object_type', 'External Agent',
          'object_version', ?
        ),
        OBJECT_CONSTRUCT(
          'run_name', ?
        ),
        OBJECT_CONSTRUCT(
          'type', 'stage_file',
          'stage_file_path', ?,
          'input_record_count', ?
        ),
        ARRAY_CONSTRUCT(),
        ARRAY_CONSTRUCT('INGESTION_MULTIPLE_BATCHES')
      )
      `,
      [
        `${database}.${schema}.${appName.toUpperCase()}`,
        appVersion,
        runName ?? "",
        `@${database}.${schema}.${STAGE_NAME}/${tmpBasename}.gz`,
        inputRecordsCount ?? 0,
      ]
    );
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Propagate RECORD_ID and RUN_NAME from the RECORD_ROOT span to all other
 * spans in the same trace. LangChain callbacks may run outside the OTEL
 * async context, so child spans can lose baggage-propagated values.
 *
 * The persistent `cache` maps traceId → {recordId, runName} across export
 * batches so child spans arriving after the RECORD_ROOT still get the
 * correct values.
 */
/** @internal Exported for testing. */
export function propagateTraceAttrs(
  spans: ReadableSpan[],
  cache: Map<string, { recordId?: string; runName?: string }>
): ReadableSpan[] {
  for (const span of spans) {
    const traceId = span.spanContext().traceId;
    const rid = span.attributes[SpanAttributes.RECORD_ID];
    const rn = span.attributes[SpanAttributes.RUN_NAME];
    if (rid || rn) {
      const existing = cache.get(traceId) ?? {};
      if (rid) existing.recordId = String(rid);
      if (rn) existing.runName = String(rn);
      cache.set(traceId, existing);
    }
  }

  return spans.map((span) => {
    const cached = cache.get(span.spanContext().traceId);
    if (!cached) return span;
    const needsRecordId =
      !span.attributes[SpanAttributes.RECORD_ID] && cached.recordId;
    const needsRunName =
      !span.attributes[SpanAttributes.RUN_NAME] && cached.runName;
    if (!needsRecordId && !needsRunName) return span;
    const extra: Record<string, string> = {};
    if (needsRecordId) extra[SpanAttributes.RECORD_ID] = cached.recordId!;
    if (needsRunName) extra[SpanAttributes.RUN_NAME] = cached.runName!;
    return Object.create(span, {
      attributes: {
        value: { ...span.attributes, ...extra },
        enumerable: true,
      },
    }) as ReadableSpan;
  });
}

/** @internal Exported for testing. */
export function isTruLensSpan(span: ReadableSpan): boolean {
  return (
    ResourceAttributes.APP_NAME in span.resource.attributes ||
    SpanAttributes.SPAN_TYPE in span.attributes
  );
}

/**
 * Encode a non-negative integer as a protobuf base-128 varint.
 * @internal Exported for testing.
 */
export function encodeVarint(value: number): Buffer {
  const bytes: number[] = [];
  while (value > 0x7f) {
    bytes.push((value & 0x7f) | 0x80);
    value >>>= 7;
  }
  bytes.push(value & 0x7f);
  return Buffer.from(bytes);
}
