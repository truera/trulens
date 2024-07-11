-- Run this script before starting.

-- If using a new role, create role and grant the role the necessary permissions.
-- For MLPLATFORMTEST, you can just get the permissons on your account.
CREATE ROLE <role>;
GRANT ROLE <role> TO USER <user_name>;

GRANT CREATE INTEGRATION ON ACCOUNT TO ROLE <role>;
GRANT CREATE WAREHOUSE ON ACCOUNT TO ROLE <role>;
GRANT CREATE DATABASE ON ACCOUNT TO ROLE <role>;
GRANT CREATE APPLICATION PACKAGE ON ACCOUNT TO ROLE <role>;
GRANT CREATE APPLICATION ON ACCOUNT TO ROLE <role>;
GRANT CREATE COMPUTE POOL ON ACCOUNT TO ROLE <role> WITH GRANT OPTION;
GRANT BIND SERVICE ENDPOINT ON ACCOUNT TO ROLE <role> WITH GRANT OPTION;


-- Create the warehouse/image repository.
USE ROLE <role>;

CREATE OR REPLACE WAREHOUSE tutorial_warehouse WITH
  WAREHOUSE_SIZE = 'X-SMALL'
  AUTO_SUSPEND = 180
  AUTO_RESUME = true
  INITIALLY_SUSPENDED = false;

CREATE DATABASE IF NOT EXISTS <database>;
CREATE SCHEMA IF NOT EXISTS TRULENS_DEMO;
USE SCHEMA TRULENS_DEMO;
CREATE IMAGE REPOSITORY IF NOT EXISTS TRULENS_IMAGE_REPO;

-- Create the schema for the network rule/integration.
USE DATABASE <database>;

CREATE OR REPLACE NETWORK RULE core.allow_all_network_rule
    TYPE = 'HOST_PORT'
    MODE = 'EGRESS'
    VALUE_LIST = ('0.0.0.0:443','0.0.0.0:80');

CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION trulens_demo_access_integration
    ALLOWED_NETWORK_RULES = (core.allow_all_network_rule)
    ENABLED = true;
