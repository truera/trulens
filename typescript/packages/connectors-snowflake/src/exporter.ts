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

// Snowflake stage used for span uploads (temp, auto-cleaned by Snowflake).
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

    try {
      await this._exportToSnowflake(trulensSpans);
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
    await this.connector.close();
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
    const tmpPath = this._writeSpansToTempFile(group.spans);
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

  private _writeSpansToTempFile(spans: ReadableSpan[]): string {
    const tmpPath = path.join(os.tmpdir(), `trulens_spans_${Date.now()}.pb`);

    // Serialise using the standard OTLP protobuf format.
    // ProtobufTraceSerializer is the stable API in @opentelemetry/otlp-transformer.
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { ProtobufTraceSerializer } = require("@opentelemetry/otlp-transformer") as typeof import("@opentelemetry/otlp-transformer");
    const serialised = ProtobufTraceSerializer.serializeRequest(spans);

    const buffers: Buffer[] = [];
    if (serialised) {
      // Length-delimited: write varint(len) then bytes (mirrors Python).
      buffers.push(encodeVarint(serialised.byteLength));
      buffers.push(Buffer.from(serialised));
    }
    fs.writeFileSync(tmpPath, Buffer.concat(buffers));
    return tmpPath;
  }

  private async _uploadToStage(tmpPath: string): Promise<void> {
    // Create the stage if needed, then PUT the file.
    await this.connector.execute(
      `CREATE TEMP STAGE IF NOT EXISTS ${STAGE_NAME}`
    );

    const tmpBasename = path.basename(tmpPath);
    // Snowflake's PUT command uploads local files to a stage.
    await this.connector.execute(
      `PUT file://${tmpPath} @${STAGE_NAME}/${tmpBasename}`
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
    ) as Array<{ CURRENT_DATABASE: string; CURRENT_SCHEMA: string }>;

    const database = rows[0]?.CURRENT_DATABASE ?? "";
    const schema = rows[0]?.CURRENT_SCHEMA ?? "";

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
        ARRAY_CONSTRUCT('ingestion_multiple_batches')
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

function isTruLensSpan(span: ReadableSpan): boolean {
  return (
    ResourceAttributes.APP_NAME in span.resource.attributes ||
    SpanAttributes.SPAN_TYPE in span.attributes
  );
}

/** Encode a non-negative integer as a protobuf base-128 varint. */
function encodeVarint(value: number): Buffer {
  const bytes: number[] = [];
  while (value > 0x7f) {
    bytes.push((value & 0x7f) | 0x80);
    value >>>= 7;
  }
  bytes.push(value & 0x7f);
  return Buffer.from(bytes);
}
