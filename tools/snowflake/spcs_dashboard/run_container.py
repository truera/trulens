from argparse import ArgumentParser

from snowflake.snowpark import Session

# get args from command line
parser = ArgumentParser(description="Run container script")
parser.add_argument(
    "--build-docker",
    action="store_true",
    help="Build and push the Docker container",
)
args = parser.parse_args()

session = Session.builder.create()
account = session.get_current_account()
user = session.get_current_user()
database = session.get_current_database()
schema = session.get_current_schema()
warehouse = session.get_current_warehouse()
role = session.get_current_role()


def run_sql_command(command: str):
    print(f"Running SQL command: {command}")
    result = session.sql(command).collect()
    print(f"Result: {result}")
    return result


# Check if the image repository exists, if not create it
repository_name = "TRULENS_REPOSITORY"
images = session.sql("SHOW IMAGE REPOSITORIES").collect()
repository_exists = any(image["name"] == repository_name for image in images)

if not repository_exists:
    session.sql(f"CREATE IMAGE REPOSITORY {repository_name}").collect()
    print(f"Image repository {repository_name} created.")
else:
    print(f"Image repository {repository_name} already exists.")

# Retrieve the repository URL
repository_url = (
    session.sql(f"SHOW IMAGE REPOSITORIES LIKE '{repository_name}'")
    .select('"repository_url"')
    .collect()[0]["repository_url"]
)

image_name = "trulens_dashboard"
image_tag = "latest"
app_name = "trulens_dashboard"
container_name = app_name + "_container"
if args.build_docker:
    # local build, with docker
    import subprocess

    subprocess.run(
        [
            "docker",
            "build",
            "--platform",
            "linux/amd64",
            "-t",
            f"{repository_url}/{image_name}:{image_tag}",
            ".",
        ],
        check=True,
    )
    subprocess.run(
        ["docker", "push", f"{repository_url}/{image_name}:{image_tag}"],
        check=True,
    )


# Create compute pool if it does not exist
compute_pool_name = input("Enter compute pool name: ")
compute_pools = session.sql("SHOW COMPUTE POOLS").collect()
compute_pool_exists = any(
    pool["name"] == compute_pool_name.upper() for pool in compute_pools
)
if compute_pool_exists:
    print(f"Compute pool {compute_pool_name} already exists")
else:
    session.sql(
        f"CREATE COMPUTE POOL {compute_pool_name} MIN_NODES = 1 MAX_NODES = 1 INSTANCE_FAMILY = CPU_X64_M"
    ).collect()
session.sql(f"DESCRIBE COMPUTE POOL {compute_pool_name}").collect()

# Create network rule
network_rule_name = f"{compute_pool_name}_allow_http_https"
session.sql(
    f"CREATE OR REPLACE NETWORK RULE {network_rule_name} TYPE = 'HOST_PORT' MODE = 'EGRESS' VALUE_LIST = ('0.0.0.0:443','0.0.0.0:80')"
).collect()
session.sql("SHOW NETWORK RULES").collect()

# Create external access integration
access_integration_name = f"{compute_pool_name}_access_integration"
session.sql(
    f"CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION {access_integration_name} ALLOWED_NETWORK_RULES = ({network_rule_name}) ENABLED = true"
).collect()
session.sql("SHOW EXTERNAL ACCESS INTEGRATIONS").collect()

service_name = compute_pool_name + "_trulens_dashboard"
session.sql(
    """
CREATE SERVICE {service_name}
    IN COMPUTE POOL {compute_pool_name}
    EXTERNAL_ACCESS_INTEGRATIONS = ({access_integration_name})
    FROM SPECIFICATION $$
    spec:
      containers:
      - name: trulens-dashboard
        image: /{database}/{schema}/{repository_name}/{app_name}:latest
        env:
            SNOWFLAKE_ACCOUNT: "{account}"
            SNOWFLAKE_DATABASE: "{database}"
            SNOWFLAKE_SCHEMA: "{schema}"
            SNOWFLAKE_WAREHOUSE: "{warehouse}"
            SNOWFLAKE_ROLE: "{role}"
            RUN_DASHBOARD: "1"
      endpoints:
      - name: trulens-demo-dashboard-endpoint
        port: 8484
        public: true
    $$
""".format(
        service_name=service_name,
        compute_pool_name=compute_pool_name,
        access_integration_name=access_integration_name,
        repository_name=repository_name,
        account=account,
        database=database,
        schema=schema,
        warehouse=warehouse,
        role=role,
        app_name=app_name,
    )
).collect()

# Show services and get their status
run_sql_command(f"SHOW ENDPOINTS IN SERVICE {service_name}")
run_sql_command(f"CALL SYSTEM$GET_SERVICE_STATUS('{service_name}')")
run_sql_command(f"CALL SYSTEM$GET_SERVICE_STATUS('{service_name}')")

# Close the session
session.close()
