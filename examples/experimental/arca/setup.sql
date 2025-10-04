-- ============================================================================
-- ARCA Snowflake Setup - Complete SQL Commands
-- ============================================================================
-- 
-- This file contains all SQL commands needed to set up the Snowflake
-- environment for ARCA (Agent Response and Configuration Automation).
-- 
-- Prerequisites:
-- - ACCOUNTADMIN role (or equivalent permissions)
-- - CORTEX enabled in your Snowflake account
-- 
-- ============================================================================

-- Switch to ACCOUNTADMIN role (required for most setup operations)
USE ROLE ACCOUNTADMIN;

-- ============================================================================
-- 1. CREATE DATABASE
-- ============================================================================

CREATE DATABASE IF NOT EXISTS NV_AGENT_RCA_DB;
USE DATABASE NV_AGENT_RCA_DB;

-- ============================================================================
-- 2. CREATE SCHEMA
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS NV_AGENT_RCA_SCHEMA;
USE SCHEMA NV_AGENT_RCA_SCHEMA;

-- ============================================================================
-- 3. CREATE WAREHOUSE
-- ============================================================================

CREATE WAREHOUSE IF NOT EXISTS NV_AGENT_RCA_WH 
WITH 
    WAREHOUSE_SIZE = 'MEDIUM';

USE WAREHOUSE NV_AGENT_RCA_WH;

-- ============================================================================
-- 4. CREATE STAGE
-- ============================================================================

CREATE STAGE IF NOT EXISTS NV_AGENT_RCA_STAGE;

-- ============================================================================
-- 5. CREATE EXAMPLE UDF (for testing)
-- ============================================================================

CREATE OR REPLACE FUNCTION calculate_revenue(
    region VARCHAR, 
    start_date DATE, 
    end_date DATE
)
RETURNS FLOAT
LANGUAGE PYTHON
RUNTIME_VERSION='3.10'
HANDLER='calculate_revenue'
AS $$
def calculate_revenue(region, start_date, end_date):
    # Mock implementation - returns fake revenue
    import random
    random.seed(hash(region + str(start_date) + str(end_date)))
    return random.uniform(1000000, 10000000)
$$;

-- Test the UDF
-- SELECT calculate_revenue('North America', '2024-01-01'::DATE, '2024-03-31'::DATE);

-- ============================================================================
-- 6. OPTIONAL: CREATE CUSTOM ROLE (if not using ACCOUNTADMIN)
-- ============================================================================

-- Uncomment if you want to create a dedicated role for ARCA:

-- CREATE ROLE IF NOT EXISTS NVYTLA_ARCA_AGENT_ROLE;
-- GRANT ROLE NVYTLA_ARCA_AGENT_ROLE TO USER your_username;

-- Grant privileges to the custom role:
-- GRANT ALL ON DATABASE NV_AGENT_RCA_DB TO ROLE NV_AGENT_RCA_AGENT_ROLE;
-- GRANT ALL ON SCHEMA NV_AGENT_RCA_DB.NV_AGENT_RCA_SCHEMA TO ROLE NV_AGENT_RCA_AGENT_ROLE;
-- GRANT ALL ON WAREHOUSE NV_AGENT_RCA_WH TO ROLE NV_AGENT_RCA_AGENT_ROLE;
-- GRANT ALL ON ALL TABLES IN SCHEMA NV_AGENT_RCA_DB.NV_AGENT_RCA_SCHEMA TO ROLE NV_AGENT_RCA_AGENT_ROLE;
-- GRANT ALL ON FUTURE TABLES IN SCHEMA NV_AGENT_RCA_DB.NV_AGENT_RCA_SCHEMA TO ROLE NV_AGENT_RCA_AGENT_ROLE;

-- ============================================================================
-- VERIFICATION QUERIES
-- ============================================================================

-- Check current context
SELECT CURRENT_ROLE(), CURRENT_DATABASE(), CURRENT_SCHEMA(), CURRENT_WAREHOUSE();

-- List databases
SHOW DATABASES LIKE 'NV_AGENT_RCA%';

-- List schemas
SHOW SCHEMAS IN DATABASE NV_AGENT_RCA_DB;

-- List warehouses
SHOW WAREHOUSES LIKE 'NV_AGENT_RCA%';

-- List stages
SHOW STAGES IN SCHEMA NV_AGENT_RCA_DB.NV_AGENT_RCA_SCHEMA;

-- List functions
SHOW FUNCTIONS IN SCHEMA NV_AGENT_RCA_DB.NV_AGENT_RCA_SCHEMA;

-- Check grants
SHOW GRANTS TO ROLE ACCOUNTADMIN;

-- ============================================================================
-- CLEANUP (if needed)
-- ============================================================================

-- WARNING: These commands will DELETE all objects!
-- Uncomment only if you need to start fresh:

-- DROP FUNCTION IF EXISTS NV_AGENT_RCA_DB.NV_AGENT_RCA_SCHEMA.calculate_revenue(VARCHAR, DATE, DATE);
-- DROP STAGE IF EXISTS NV_AGENT_RCA_DB.NV_AGENT_RCA_SCHEMA.NV_AGENT_RCA_STAGE;
-- DROP WAREHOUSE IF EXISTS NV_AGENT_RCA_WH;
-- DROP SCHEMA IF EXISTS NV_AGENT_RCA_DB.NV_AGENT_RCA_SCHEMA;
-- DROP DATABASE IF EXISTS NV_AGENT_RCA_DB;
-- DROP ROLE IF EXISTS NV_AGENT_RCA_AGENT_ROLE;

-- ============================================================================
-- END OF SETUP
-- ============================================================================

