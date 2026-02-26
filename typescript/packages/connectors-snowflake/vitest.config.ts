import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    // No unit tests for the Snowflake connector — integration tests
    // require live Snowflake credentials and are run separately.
    passWithNoTests: true,
  },
});
