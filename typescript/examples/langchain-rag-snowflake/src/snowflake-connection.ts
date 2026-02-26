import "dotenv/config";
import { SnowflakeConnector } from "@trulens/connectors-snowflake";

export const sfConnector = new SnowflakeConnector({
  account: process.env.SNOWFLAKE_ACCOUNT!,
  username: process.env.SNOWFLAKE_USER!,
  password: process.env.SNOWFLAKE_PASSWORD,
  database: process.env.SNOWFLAKE_DATABASE!,
  schema: process.env.SNOWFLAKE_SCHEMA!,
  warehouse: process.env.SNOWFLAKE_WAREHOUSE!,
});

/**
 * Connect and verify with a simple query.
 */
export async function connectAndVerify(): Promise<void> {
  console.log("Connecting to Snowflake...");

  const rows = (await sfConnector.execute(
    'SELECT CURRENT_USER() AS "user", CURRENT_DATABASE() AS "db", CURRENT_SCHEMA() AS "schema"',
  )) as Array<Record<string, string>>;

  const r = rows[0] ?? {};
  console.log(
    `Connected as ${r.user} | ${r.db}.${r.schema}\n`,
  );
}
