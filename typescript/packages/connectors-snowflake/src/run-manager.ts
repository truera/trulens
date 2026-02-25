/**
 * SnowflakeRunManager — manages the lifecycle of Snowflake AI Observability
 * Runs from TypeScript, without needing Python.
 *
 * Ports the key SQL operations from:
 *   - trulens.connectors.snowflake.dao.external_agent.ExternalAgentDao
 *   - trulens.connectors.snowflake.dao.run.RunDao
 */

import type { SnowflakeConnector } from "./connector.js";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export enum RunStatus {
  CREATED = "CREATED",
  INVOCATION_IN_PROGRESS = "INVOCATION_IN_PROGRESS",
  INVOCATION_COMPLETED = "INVOCATION_COMPLETED",
  INVOCATION_PARTIALLY_COMPLETED = "INVOCATION_PARTIALLY_COMPLETED",
  COMPUTATION_IN_PROGRESS = "COMPUTATION_IN_PROGRESS",
  COMPLETED = "COMPLETED",
  PARTIALLY_COMPLETED = "PARTIALLY_COMPLETED",
  FAILED = "FAILED",
  CANCELLED = "CANCELLED",
  UNKNOWN = "UNKNOWN",
}

export interface RunManagerOptions {
  connector: SnowflakeConnector;
}

export interface CreateRunOptions {
  objectName: string;
  appVersion: string;
  runName: string;
  datasetName?: string;
  sourceType?: string;
  datasetSpec?: Record<string, string>;
  mode?: "APP_INVOCATION" | "LOG_INGESTION";
  description?: string;
  label?: string;
  llmJudgeName?: string;
}

export interface RunIdentifier {
  objectName: string;
  objectType?: string;
  appVersion?: string;
  runName: string;
}

export interface FinalizeRunOptions extends RunIdentifier {
  appVersion: string;
  inputRecordsCount: number;
}

export interface ComputeMetricsOptions extends RunIdentifier {
  appVersion: string;
  metrics: string[];
}

export interface WaitForIngestionOptions extends RunIdentifier {
  pollIntervalMs?: number;
  timeoutMs?: number;
}

export interface WaitForMetricsOptions extends RunIdentifier {
  pollIntervalMs?: number;
  timeoutMs?: number;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DEFAULT_LLM_JUDGE_NAME = "llama3.1-70b";
const DEFAULT_OBJECT_TYPE = "External Agent";

// ---------------------------------------------------------------------------
// SnowflakeRunManager
// ---------------------------------------------------------------------------

export class SnowflakeRunManager {
  private readonly connector: SnowflakeConnector;

  constructor(options: RunManagerOptions) {
    this.connector = options.connector;
  }

  // -----------------------------------------------------------------------
  // External Agent
  // -----------------------------------------------------------------------

  /**
   * Create the External Agent and version if they don't already exist.
   * Returns the fully-qualified object name (DB.SCHEMA.AGENT_NAME).
   */
  async ensureExternalAgent(
    appName: string,
    appVersion: string
  ): Promise<string> {
    const resolvedName = appName.toUpperCase();
    const exists = await this._agentExists(resolvedName);

    if (!exists) {
      await this.connector.execute(
        `CREATE EXTERNAL AGENT "${resolvedName}" WITH VERSION "${appVersion}"`
      );
    } else {
      const versions = await this._listAgentVersions(resolvedName);
      if (!versions.includes(appVersion)) {
        await this.connector.execute(
          `ALTER EXTERNAL AGENT IF EXISTS "${resolvedName}" ADD VERSION "${appVersion}"`
        );
      }
    }

    const { database, schema } = await this._currentDbSchema();
    return `${database}.${schema}.${resolvedName}`;
  }

  // -----------------------------------------------------------------------
  // Run CRUD
  // -----------------------------------------------------------------------

  /** Create a new Run entity via SYSTEM$AIML_RUN_OPERATION('CREATE'). */
  async createRun(opts: CreateRunOptions): Promise<void> {
    const { database, schema } = await this._currentDbSchema();
    const fqName = `${database}.${schema}.${opts.objectName.toUpperCase()}`;

    const payload: Record<string, unknown> = {
      object_name: fqName,
      object_type: DEFAULT_OBJECT_TYPE,
      run_name: opts.runName,
      description: opts.description ?? "",
      run_metadata: {
        labels: opts.label ? [opts.label] : [],
        llm_judge_name: opts.llmJudgeName ?? DEFAULT_LLM_JUDGE_NAME,
        mode: opts.mode ?? "APP_INVOCATION",
      },
      source_info: {
        name: opts.datasetName ?? opts.runName,
        source_type: opts.sourceType ?? "DATAFRAME",
        column_spec: opts.datasetSpec ?? {},
      },
    };

    if (opts.appVersion) {
      payload.object_version = opts.appVersion;
    }

    await this.connector.execute(
      `SELECT SYSTEM$AIML_RUN_OPERATION('CREATE', ?)`,
      [JSON.stringify(payload)]
    );
  }

  /**
   * Get the current status of a run by querying
   * SYSTEM$AIML_RUN_OPERATION('GET').
   */
  async getRunStatus(opts: RunIdentifier): Promise<RunStatus> {
    const { database, schema } = await this._currentDbSchema();
    const fqName = `${database}.${schema}.${opts.objectName.toUpperCase()}`;

    const payload: Record<string, unknown> = {
      object_name: fqName,
      object_type: opts.objectType ?? DEFAULT_OBJECT_TYPE,
      run_name: opts.runName,
    };
    if (opts.appVersion) {
      payload.object_version = opts.appVersion;
    }

    const rows = await this.connector.execute(
      `SELECT SYSTEM$AIML_RUN_OPERATION('GET', ?)`,
      [JSON.stringify(payload)]
    );

    if (!rows || rows.length === 0) return RunStatus.UNKNOWN;

    const raw = Object.values(rows[0] as Record<string, unknown>)[0];
    let meta: Record<string, unknown>;
    try {
      meta =
        typeof raw === "string"
          ? JSON.parse(raw)
          : (raw as Record<string, unknown>);
    } catch {
      return RunStatus.UNKNOWN;
    }

    return this._parseRunStatus(meta);
  }

  /**
   * Poll getRunStatus until ingestion is complete (INVOCATION_COMPLETED) or
   * the timeout is reached.
   */
  async waitForIngestion(opts: WaitForIngestionOptions): Promise<void> {
    const pollMs = opts.pollIntervalMs ?? 3_000;
    const timeoutMs = opts.timeoutMs ?? 300_000;
    const start = Date.now();

    while (Date.now() - start < timeoutMs) {
      const status = await this.getRunStatus(opts);

      if (
        status === RunStatus.INVOCATION_COMPLETED ||
        status === RunStatus.COMPLETED ||
        status === RunStatus.PARTIALLY_COMPLETED ||
        status === RunStatus.COMPUTATION_IN_PROGRESS
      ) {
        return;
      }

      if (
        status === RunStatus.FAILED ||
        status === RunStatus.CANCELLED
      ) {
        throw new Error(
          `Run "${opts.runName}" entered terminal status: ${status}`
        );
      }

      await sleep(pollMs);
    }

    throw new Error(
      `Timed out waiting for ingestion of run "${opts.runName}" ` +
        `after ${timeoutMs}ms`
    );
  }

  /**
   * Poll getRunStatus until metric computation is complete
   * (COMPLETED / PARTIALLY_COMPLETED) or the timeout is reached.
   * Returns the terminal RunStatus so callers can distinguish full vs partial.
   */
  async waitForMetrics(opts: WaitForMetricsOptions): Promise<RunStatus> {
    const pollMs = opts.pollIntervalMs ?? 5_000;
    const timeoutMs = opts.timeoutMs ?? 600_000;
    const start = Date.now();

    while (Date.now() - start < timeoutMs) {
      const status = await this.getRunStatus(opts);

      if (
        status === RunStatus.COMPLETED ||
        status === RunStatus.PARTIALLY_COMPLETED
      ) {
        return status;
      }

      if (
        status === RunStatus.FAILED ||
        status === RunStatus.CANCELLED
      ) {
        throw new Error(
          `Run "${opts.runName}" entered terminal status: ${status}`
        );
      }

      await sleep(pollMs);
    }

    throw new Error(
      `Timed out waiting for metrics of run "${opts.runName}" ` +
        `after ${timeoutMs}ms`
    );
  }

  // -----------------------------------------------------------------------
  // Finalize & Compute
  // -----------------------------------------------------------------------

  /** Finalize the run: trigger START_INGESTION phase. */
  async finalizeRun(opts: FinalizeRunOptions): Promise<void> {
    const { database, schema } = await this._currentDbSchema();
    const fqName = `${database}.${schema}.${opts.objectName.toUpperCase()}`;
    const objectType = opts.objectType ?? DEFAULT_OBJECT_TYPE;

    await this.connector.execute(
      `
      CALL SYSTEM$EXECUTE_AI_OBSERVABILITY_RUN(
        OBJECT_CONSTRUCT(
          'object_name', ?,
          'object_type', ?,
          'object_version', ?
        ),
        OBJECT_CONSTRUCT(
          'run_name', ?
        ),
        OBJECT_CONSTRUCT(
          'type', 'stage_file',
          'input_record_count', ?
        ),
        ARRAY_CONSTRUCT(),
        ARRAY_CONSTRUCT('START_INGESTION')
      )
      `,
      [
        fqName,
        objectType,
        opts.appVersion,
        opts.runName,
        opts.inputRecordsCount,
      ]
    );
  }

  /**
   * Trigger server-side metric computation (e.g. "answer_relevance").
   * Metric computation is asynchronous on Snowflake's side — this call
   * returns once the tasks are set up.
   */
  async computeMetrics(opts: ComputeMetricsOptions): Promise<void> {
    if (opts.metrics.length === 0) {
      throw new Error("At least one metric name must be provided.");
    }

    const { database, schema } = await this._currentDbSchema();
    const fqName = `${database}.${schema}.${opts.objectName.toUpperCase()}`;
    const objectType = opts.objectType ?? DEFAULT_OBJECT_TYPE;

    const metricPlaceholders = opts.metrics.map(() => "?").join(", ");

    await this.connector.execute(
      `
      CALL SYSTEM$EXECUTE_AI_OBSERVABILITY_RUN(
        OBJECT_CONSTRUCT(
          'object_name', ?,
          'object_type', ?,
          'object_version', ?
        ),
        OBJECT_CONSTRUCT(
          'run_name', ?
        ),
        OBJECT_CONSTRUCT('type', 'stage_file'),
        ARRAY_CONSTRUCT(${metricPlaceholders}),
        ARRAY_CONSTRUCT('COMPUTE_METRICS')
      )
      `,
      [fqName, objectType, opts.appVersion, opts.runName, ...opts.metrics]
    );
  }

  // -----------------------------------------------------------------------
  // Private helpers
  // -----------------------------------------------------------------------

  private _dbSchemaCache: { database: string; schema: string } | null = null;

  private async _currentDbSchema(): Promise<{
    database: string;
    schema: string;
  }> {
    if (this._dbSchemaCache) return this._dbSchemaCache;

    const rows = (await this.connector.execute(
      `SELECT CURRENT_DATABASE(), CURRENT_SCHEMA()`
    )) as Array<{
      "CURRENT_DATABASE()": string;
      "CURRENT_SCHEMA()": string;
    }>;

    this._dbSchemaCache = {
      database: rows[0]?.["CURRENT_DATABASE()"] ?? "",
      schema: rows[0]?.["CURRENT_SCHEMA()"] ?? "",
    };
    return this._dbSchemaCache;
  }

  private async _agentExists(resolvedName: string): Promise<boolean> {
    const rows = (await this.connector.execute(
      `SHOW EXTERNAL AGENTS`
    )) as Array<Record<string, unknown>>;

    return rows.some(
      (r) => String(r.name ?? r.NAME ?? "").toUpperCase() === resolvedName
    );
  }

  private async _listAgentVersions(
    resolvedName: string
  ): Promise<string[]> {
    const rows = (await this.connector.execute(
      `SHOW VERSIONS IN EXTERNAL AGENT "${resolvedName}"`
    )) as Array<Record<string, unknown>>;

    return rows.map((r) => String(r.name ?? r.NAME ?? ""));
  }

  private _parseRunStatus(meta: Record<string, unknown>): RunStatus {
    const runStatus = meta.run_status as string | undefined;
    if (runStatus === "CANCELLED") return RunStatus.CANCELLED;

    const runMetadata = meta.run_metadata as Record<string, unknown> | undefined;
    if (!runMetadata) return RunStatus.CREATED;

    const invocations = runMetadata.invocations as
      | Record<string, Record<string, unknown>>
      | undefined;

    const hasInvocations =
      invocations != null && Object.keys(invocations).length > 0;

    const metrics = runMetadata.metrics as
      | Record<string, Record<string, unknown>>
      | undefined;

    const hasMetrics = metrics != null && Object.keys(metrics).length > 0;

    if (!hasInvocations) return RunStatus.CREATED;

    if (hasMetrics) {
      return this._overallComputationStatus(invocations!, metrics!);
    }

    return this._latestInvocationStatus(invocations!);
  }

  private _latestInvocationStatus(
    invocations: Record<string, Record<string, unknown>>
  ): RunStatus {
    const entries = Object.values(invocations);
    const latest = entries.reduce((a, b) =>
      ((a.start_time_ms as number) ?? 0) >= ((b.start_time_ms as number) ?? 0)
        ? a
        : b
    );

    const cs = latest.completion_status as
      | Record<string, unknown>
      | undefined;
    const status = cs?.status as string | undefined;

    switch (status) {
      case "COMPLETED":
        return RunStatus.INVOCATION_COMPLETED;
      case "PARTIALLY_COMPLETED":
        return RunStatus.INVOCATION_PARTIALLY_COMPLETED;
      case "STARTED":
        return RunStatus.INVOCATION_IN_PROGRESS;
      case "FAILED":
        return RunStatus.FAILED;
      default:
        return RunStatus.UNKNOWN;
    }
  }

  private _overallComputationStatus(
    invocations: Record<string, Record<string, unknown>>,
    metrics: Record<string, Record<string, unknown>>
  ): RunStatus {
    const metricEntries = Object.values(metrics);
    const allCompleted = metricEntries.every(
      (m) =>
        (m.completion_status as Record<string, unknown> | undefined)
          ?.status === "COMPLETED"
    );
    const allFailed = metricEntries.every(
      (m) =>
        (m.completion_status as Record<string, unknown> | undefined)
          ?.status === "FAILED"
    );

    const entries = Object.values(invocations);
    const latest = entries.reduce((a, b) =>
      ((a.start_time_ms as number) ?? 0) >= ((b.start_time_ms as number) ?? 0)
        ? a
        : b
    );
    const invocationStatus = (
      latest.completion_status as Record<string, unknown> | undefined
    )?.status as string | undefined;

    if (allCompleted) {
      return invocationStatus === "COMPLETED"
        ? RunStatus.COMPLETED
        : RunStatus.PARTIALLY_COMPLETED;
    }

    if (allFailed) return RunStatus.FAILED;

    return RunStatus.COMPUTATION_IN_PROGRESS;
  }
}

// ---------------------------------------------------------------------------
// Util
// ---------------------------------------------------------------------------

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
