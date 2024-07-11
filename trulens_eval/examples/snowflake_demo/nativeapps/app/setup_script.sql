-- This is the setup script that runs while installing a Snowflake Native App in a consumer account.
-- To write this script, you can familiarize yourself with some of the following concepts:
-- Application Roles
-- Versioned Schemas
-- UDFs/Procs
-- Extension Code
-- Refer to https://docs.snowflake.com/en/developer-guide/native-apps/creating-setup-script for a detailed understanding of this file. 
CREATE APPLICATION ROLE IF NOT EXISTS app_user;

CREATE SCHEMA IF NOT EXISTS core;
GRANT USAGE ON SCHEMA core TO APPLICATION ROLE app_user;

CREATE OR ALTER VERSIONED SCHEMA app_public;
GRANT USAGE ON SCHEMA app_public TO APPLICATION ROLE app_user;

CREATE OR REPLACE PROCEDURE app_public.start_app(
  snowflake_account STRING, 
  snowflake_database STRING, 
  snowflake_schema STRING, 
  snowflake_warehouse STRING,
  snowflake_role STRING,
  snowflake_user STRING,
  snowflake_password STRING,
  cortex_search STRING,
  replicate_token STRING,
  pinecone_key STRING,
  external_access_integration STRING
)
   RETURNS string
   LANGUAGE sql
   AS
$$
BEGIN
   -- account-level compute pool object prefixed with app name to prevent clashes
   LET pool_name := (SELECT CURRENT_DATABASE()) || '_compute_pool';

   CREATE COMPUTE POOL IF NOT EXISTS IDENTIFIER(:pool_name)
      MIN_NODES = 1
      MAX_NODES = 1
      INSTANCE_FAMILY = CPU_X64_L
      AUTO_RESUME = true;

   LET dashboard_query := 'CREATE SERVICE IF NOT EXISTS core.trulens_demo_dashboard
      IN COMPUTE POOL ' || pool_name || '
      EXTERNAL_ACCESS_INTEGRATIONS = ('|| external_access_integration ||')
      FROM SPECIFICATION \$\$
spec:\n
  containers:\n
    - name: trulens-demo-dashboard-container\n
      image: /gtok/trulens_demo/trulens_image_repo/trulens_demo:latest\n
      env:\n
        # Env variables here cannot start with SNOWFLAKE_ otherwise they clash with vars set in the running process.\n
        TRULENS_SNOWFLAKE_DATABASE: ' || snowflake_database || '\n
        TRULENS_SNOWFLAKE_SCHEMA: ' || snowflake_schema || '\n
        TRULENS_SNOWFLAKE_WAREHOUSE: ' || snowflake_warehouse || '\n
        TRULENS_SNOWFLAKE_ROLE: ' || snowflake_role || '\n
        TRULENS_SNOWFLAKE_CORTEX_SEARCH_SERVICE: ' || cortex_search || '\n
        RUN_DASHBOARD: "1"\n
        TRULENS_SNOWFLAKE_USER: ' || snowflake_user || '\n
        TRULENS_SNOWFLAKE_USER_PASSWORD: ' || snowflake_password || '\n
        REPLICATE_API_TOKEN: ' || replicate_token || '\n
        PINECONE_API_KEY: ' || pinecone_key || '\n
        TRULENS_SNOWFLAKE_ACCOUNT: ' || snowflake_account || '\n
  endpoints:\n
    - name: trulens-demo-dashboard-endpoint\n
      port: 8484\n
      public: true\n
\$\$;
';
   EXECUTE IMMEDIATE :dashboard_query;

   LET app_query := 'CREATE SERVICE IF NOT EXISTS core.trulens_demo_app
      IN COMPUTE POOL ' || pool_name || '
      EXTERNAL_ACCESS_INTEGRATIONS = ('|| external_access_integration ||')
      FROM SPECIFICATION \$\$
spec:\n
  containers:\n
    - name: trulens-demo-app-container\n
      image: /gtok/trulens_demo/trulens_image_repo/trulens_demo:latest\n
      env:\n
        # Env variables here cannot start with SNOWFLAKE_ otherwise they clash with vars set in the running process.\n
        TRULENS_SNOWFLAKE_DATABASE: ' || snowflake_database || '\n
        TRULENS_SNOWFLAKE_SCHEMA: ' || snowflake_schema || '\n
        TRULENS_SNOWFLAKE_WAREHOUSE: ' || snowflake_warehouse || '\n
        TRULENS_SNOWFLAKE_ROLE: ' || snowflake_role || '\n
        TRULENS_SNOWFLAKE_CORTEX_SEARCH_SERVICE: ' || cortex_search || '\n
        RUN_APP: "1"\n
        TRULENS_SNOWFLAKE_USER: ' || snowflake_user || '\n
        TRULENS_SNOWFLAKE_USER_PASSWORD: ' || snowflake_password || '\n
        REPLICATE_API_TOKEN: ' || replicate_token || '\n
        PINECONE_API_KEY: ' || pinecone_key || '\n
        TRULENS_SNOWFLAKE_ACCOUNT: ' || snowflake_account || '\n
  endpoints:\n
    - name: trulens-demo-app-endpoint\n
      port: 8501\n
      public: true\n
\$\$;
';
   EXECUTE IMMEDIATE :app_query;

    GRANT USAGE ON SERVICE core.trulens_demo_dashboard TO APPLICATION ROLE app_user;
    GRANT USAGE ON SERVICE core.trulens_demo_app TO APPLICATION ROLE app_user;
    GRANT SERVICE ROLE core.trulens_demo_dashboard!ALL_ENDPOINTS_USAGE TO APPLICATION ROLE app_user;
    GRANT SERVICE ROLE core.trulens_demo_app!ALL_ENDPOINTS_USAGE TO APPLICATION ROLE app_user;

   RETURN 'Starting app...'::VARCHAR;
END;
$$;

GRANT USAGE ON PROCEDURE app_public.start_app(STRING,STRING,STRING,STRING,STRING,STRING,STRING,STRING,STRING,STRING,STRING) TO APPLICATION ROLE app_user;

CREATE OR REPLACE PROCEDURE app_public.register_reference(ref_name STRING, operation STRING, ref_or_alias STRING)
  RETURNS STRING
  LANGUAGE SQL
AS $$
  BEGIN
    CASE (operation)
      WHEN 'ADD' THEN
        SELECT SYSTEM$SET_REFERENCE(:ref_name, :ref_or_alias);
      WHEN 'REMOVE' THEN
        SELECT SYSTEM$REMOVE_REFERENCE(:ref_name, :ref_or_alias);
      WHEN 'CLEAR' THEN
        SELECT SYSTEM$REMOVE_ALL_REFERENCES(:ref_name);
    ELSE
      RETURN 'unknown operation: ' || operation;
    END CASE;
    RETURN NULL;
  END;
$$;

grant usage on procedure app_public.register_reference(string, string, string) to application role app_user;

-- Explain
CREATE OR REPLACE procedure app_public.get_config_for_ref(ref_name STRING)
  RETURNS STRING
  LANGUAGE SQL
  AS
  $$
  BEGIN
    CASE (UPPER(ref_name))
      WHEN 'SNOWFLAKE_USER' THEN
        RETURN '{
          "type": "CONFIGURATION",
          "payload":{
            "type" : "GENERIC_STRING"}}';
      ELSE
        RETURN '{
          "type": "CONFIGURATION",
          "payload":{
            "type" : "GENERIC_STRING"}}';
END CASE;
RETURN '';
END;
$$;


grant usage on procedure app_public.get_config_for_ref(string) to application role app_user;

CREATE OR REPLACE PROCEDURE app_public.upgrade_service(
  snowflake_account STRING, 
  snowflake_database STRING, 
  snowflake_schema STRING, 
  snowflake_warehouse STRING,
  snowflake_role STRING,
  snowflake_user STRING,
  snowflake_password STRING,
  cortex_search STRING,
  replicate_token STRING,
  pinecone_key STRING
)
RETURNS VARCHAR
LANGUAGE SQL
EXECUTE AS OWNER
AS $$
BEGIN
   LET dashboard_query := 'ALTER SERVICE core.trulens_demo_dashboard
      FROM SPECIFICATION \$\$
spec:\n
  containers:\n
    - name: trulens-demo-dashboard-container\n
      image: /gtok/trulens_demo/trulens_image_repo/trulens_demo:latest\n
      env:\n
        # Env variables here cannot start with SNOWFLAKE_ otherwise they clash with vars set in the running process.\n
        TRULENS_SNOWFLAKE_DATABASE: ' || snowflake_database || '\n
        TRULENS_SNOWFLAKE_SCHEMA: ' || snowflake_schema || '\n
        TRULENS_SNOWFLAKE_WAREHOUSE: ' || snowflake_warehouse || '\n
        TRULENS_SNOWFLAKE_ROLE: ' || snowflake_role || '\n
        TRULENS_SNOWFLAKE_CORTEX_SEARCH_SERVICE: ' || cortex_search || '\n
        RUN_DASHBOARD: "1"\n
        TRULENS_SNOWFLAKE_USER: ' || snowflake_user || '\n
        TRULENS_SNOWFLAKE_USER_PASSWORD: ' || snowflake_password || '\n
        REPLICATE_API_TOKEN: ' || replicate_token || '\n
        PINECONE_API_KEY: ' || pinecone_key || '\n
        TRULENS_SNOWFLAKE_ACCOUNT: ' || snowflake_account || '\n
  endpoints:\n
    - name: trulens-demo-dashboard-endpoint\n
      port: 8484\n
      public: true\n
\$\$;
';
   EXECUTE IMMEDIATE :dashboard_query;

   LET app_query := 'ALTER SERVICE core.trulens_demo_app
      FROM SPECIFICATION \$\$
spec:\n
  containers:\n
    - name: trulens-demo-app-container\n
      image: /gtok/trulens_demo/trulens_image_repo/trulens_demo:latest\n
      env:\n
        # Env variables here cannot start with SNOWFLAKE_ otherwise they clash with vars set in the running process.\n
        TRULENS_SNOWFLAKE_DATABASE: ' || snowflake_database || '\n
        TRULENS_SNOWFLAKE_SCHEMA: ' || snowflake_schema || '\n
        TRULENS_SNOWFLAKE_WAREHOUSE: ' || snowflake_warehouse || '\n
        TRULENS_SNOWFLAKE_ROLE: ' || snowflake_role || '\n
        TRULENS_SNOWFLAKE_CORTEX_SEARCH_SERVICE: ' || cortex_search || '\n
        RUN_APP: "1"\n
        TRULENS_SNOWFLAKE_USER: ' || snowflake_user || '\n
        TRULENS_SNOWFLAKE_USER_PASSWORD: ' || snowflake_password || '\n
        REPLICATE_API_TOKEN: ' || replicate_token || '\n
        PINECONE_API_KEY: ' || pinecone_key || '\n
        TRULENS_SNOWFLAKE_ACCOUNT: ' || snowflake_account || '\n
  endpoints:\n
    - name: trulens-demo-app-endpoint\n
      port: 8501\n
      public: true\n
\$\$;
';
   EXECUTE IMMEDIATE :app_query;

   RETURN 'Starting app...'::VARCHAR;
END;
$$;

GRANT USAGE ON PROCEDURE app_public.upgrade_service(STRING,STRING,STRING,STRING,STRING,STRING,STRING,STRING,STRING,STRING) TO APPLICATION ROLE app_user;


CREATE OR REPLACE PROCEDURE app_public.dashboard_service_status()
RETURNS VARCHAR
LANGUAGE SQL
EXECUTE AS OWNER
AS $$
   DECLARE
         service_status VARCHAR;
   BEGIN
         CALL SYSTEM$GET_SERVICE_STATUS('core.trulens_demo_dashboard') INTO :service_status;
         RETURN PARSE_JSON(:service_status)[0]['status']::VARCHAR;
   END;
$$;

GRANT USAGE ON PROCEDURE app_public.dashboard_service_status() TO APPLICATION ROLE app_user;

CREATE OR REPLACE PROCEDURE app_public.app_service_status()
RETURNS VARCHAR
LANGUAGE SQL
EXECUTE AS OWNER
AS $$
   DECLARE
         service_status VARCHAR;
   BEGIN
         CALL SYSTEM$GET_SERVICE_STATUS('core.trulens_demo_app') INTO :service_status;
         RETURN PARSE_JSON(:service_status)[0]['status']::VARCHAR;
   END;
$$;

GRANT USAGE ON PROCEDURE app_public.app_service_status() TO APPLICATION ROLE app_user;

CREATE OR REPLACE PROCEDURE app_public.service_dashboard_endpoints()
RETURNS TABLE(name VARCHAR, port int, port_range varchar, protocol varchar, is_public varchar, ingress_url varchar)
LANGUAGE SQL
EXECUTE AS OWNER
AS $$
    DECLARE
        res Resultset;
    BEGIN
        res := (SHOW ENDPOINTS IN SERVICE core.trulens_demo_dashboard);
        RETURN TABLE(res);
    END;
$$;

GRANT USAGE ON PROCEDURE app_public.service_dashboard_endpoints() TO APPLICATION ROLE app_user;

CREATE OR REPLACE PROCEDURE app_public.service_app_endpoints()
RETURNS TABLE(name VARCHAR, port int, port_range varchar, protocol varchar, is_public varchar, ingress_url varchar)
LANGUAGE SQL
EXECUTE AS OWNER
AS $$
    DECLARE
        res Resultset;
    BEGIN
        res := (SHOW ENDPOINTS IN SERVICE core.trulens_demo_app);
        RETURN TABLE(res);
    END;
$$;

GRANT USAGE ON PROCEDURE app_public.service_app_endpoints() TO APPLICATION ROLE app_user;


CREATE OR REPLACE PROCEDURE app_public.get_dashboard_logs()
RETURNS TABLE(log_line VARCHAR)
LANGUAGE SQL
EXECUTE AS OWNER
AS $$
    DECLARE
        res Resultset;
    BEGIN
        res := (SELECT value AS log_line
                  FROM TABLE(
                    SPLIT_TO_TABLE(SYSTEM$GET_SERVICE_LOGS('core.trulens_demo_dashboard', '0', 'trulens-demo-dashboard-container'), '\n')
                  ));

        RETURN TABLE(res);
    END;
$$;

GRANT USAGE ON PROCEDURE app_public.get_dashboard_logs() TO APPLICATION ROLE app_user;


CREATE OR REPLACE PROCEDURE app_public.get_app_logs()
RETURNS TABLE(log_line VARCHAR)
LANGUAGE SQL
EXECUTE AS OWNER
AS $$
    DECLARE
        res Resultset;
    BEGIN
        res := (SELECT value AS log_line
                  FROM TABLE(
                    SPLIT_TO_TABLE(SYSTEM$GET_SERVICE_LOGS('core.trulens_demo_app', '0', 'trulens-demo-app-container'), '\n')
                  ));

        RETURN TABLE(res);
    END;
$$;

GRANT USAGE ON PROCEDURE app_public.get_app_logs() TO APPLICATION ROLE app_user;

-- The rest of this script is left blank for purposes of your learning and exploration. 
