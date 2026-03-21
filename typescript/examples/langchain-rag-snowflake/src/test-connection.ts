import { sfConnector, connectAndVerify } from "./snowflake-connection.js";

async function main() {
  try {
    await connectAndVerify();
    console.log("Snowflake connection is working.");
  } catch (err) {
    console.error("Connection failed:", err);
  } finally {
    await sfConnector.close();
  }
}

main();
