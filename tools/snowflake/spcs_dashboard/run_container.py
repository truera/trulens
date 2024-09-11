import os
from snowflake.snowpark import Session
from snowflake.snowpark.functions import call_builtin

# Read environment variables
account = os.getenv('SNOWFLAKE_ACCOUNT')
user = os.getenv('SNOWFLAKE_USER')
password = os.getenv('SNOWFLAKE_PASSWORD')
role = os.getenv('SNOWFLAKE_ROLE')
warehouse = os.getenv('SNOWFLAKE_WAREHOUSE')
database = os.getenv('SNOWFLAKE_DATABASE')
schema = os.getenv('SNOWFLAKE_SCHEMA')

# Define Snowflake connection parameters
connection_parameters = {
    "account": account,
    "user": user,
    "password": password,
    "role": role,
    "warehouse": warehouse,
    "database": database,
    "schema": schema
}

# Create a Snowflake session
session = Session.builder.configs(connection_parameters).create()

# Create compute pool if it does not exist
compute_pool_name = input("Enter compute pool name: ")
compute_pools = session.sql("SHOW COMPUTE POOLS").collect()
compute_pool_exists = any(pool['name'] == compute_pool_name.upper() for pool in compute_pools)
if compute_pool_exists:
    print(f"Compute pool {compute_pool_name} already exists")
else:
    session.sql(f"CREATE COMPUTE POOL {compute_pool_name} MIN_NODES = 1 MAX_NODES = 1 INSTANCE_FAMILY = CPU_X64_M").collect()
session.sql(f"DESCRIBE COMPUTE POOL {compute_pool_name}").collect()

# Create image repository
image_repository_name = f"trulens_image_repository"
session.sql(f"CREATE IMAGE REPOSITORY {image_repository_name}").collect()
session.sql("SHOW IMAGE REPOSITORIES").collect()

# Create network rule
network_rule_name = f"{compute_pool_name}_allow_all_network_rule"
session.sql(f"CREATE OR REPLACE NETWORK RULE {network_rule_name} TYPE = 'HOST_PORT' MODE = 'EGRESS' VALUE_LIST = ('0.0.0.0:443','0.0.0.0:80')").collect()
session.sql("SHOW NETWORK RULES").collect()

# Create external access integration
access_integration_name = f"{compute_pool_name}_access_integration"
session.sql(f"CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION {access_integration_name} ALLOWED_NETWORK_RULES = ({network_rule_name}) ENABLED = true").collect()
session.sql("SHOW EXTERNAL ACCESS INTEGRATIONS").collect()

app_name = "trulens_dashboard"
secret_name = f"{schema}.{app_name}_login_credentials"
session.sql(f"CREATE SECRET {secret_name} TYPE=password USERNAME={user} PASSWORD='{password}'").collect()

service_name=compute_pool_name+"_trulens_dashboard"
session.sql("""
CREATE SERVICE {service_name}
    IN COMPUTE POOL {compute_pool_name}
    EXTERNAL_ACCESS_INTEGRATIONS = ({access_integration_name})
    FROM SPECIFICATION $$
    spec:
      containers:
      - name: {container_name}
        image: /{database}/{schema}/{container_name}/{app_name}:latest
        env:
            SNOWFLAKE_ACCOUNT: "{account}"
            SNOWFLAKE_DATABASE: "{database}"
            SNOWFLAKE_SCHEMA: "{schema}"
            SNOWFLAKE_WAREHOUSE: "{warehouse}"
            SNOWFLAKE_ROLE: "{role}"
            RUN_DASHBOARD: "1"
        secrets:
          - snowflakeSecret: {secret_name}
            secretKeyRef: username
            envVarName: SNOWFLAKE_USER
          - snowflakeSecret: {secret_name}
            secretKeyRef: password
            envVarName: SNOWFLAKE_PASSWORD
      endpoints:
      - name: trulens-demo-dashboard-endpoint
        port: 8484
        public: true
    $$
""".format(service_name=service_name,
           compute_pool_name=compute_pool_name,
           access_integration_name=access_integration_name,
           container_name=app_name+"_container",
           account=account,
           database=database,
           schema=schema,
           warehouse=warehouse,
           role=role,
           app_name=app_name)).collect()

# Show services and get their status
session.sql(f"SHOW ENDPOINTS IN SERVICE {service_name}").collect()
session.sql("CALL SYSTEM$GET_SERVICE_STATUS('dkurokawa_trulens_demo_app')").collect()
session.sql("CALL SYSTEM$GET_SERVICE_STATUS('dkurokawa_trulens_demo_dashboard')").collect()

# Close the session
session.close()