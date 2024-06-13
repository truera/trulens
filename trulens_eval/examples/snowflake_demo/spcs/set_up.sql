USE DATABASE dkurokawa;
USE SCHEMA trulens_demo;
USE WAREHOUSE dkurokawa;

CREATE COMPUTE POOL dkurokawa_trulens_demo_compute_pool
    MIN_NODES = 1
    MAX_NODES = 1
    INSTANCE_FAMILY = CPU_X64_M;
SHOW COMPUTE POOLS;

CREATE IMAGE REPOSITORY dkurokawa_trulens_demo_image_repository;
SHOW IMAGE REPOSITORIES;

CREATE STAGE dkurokawa_trulens_demo_stage
    DIRECTORY = ( ENABLE = true );
SHOW STAGES;

CREATE OR REPLACE NETWORK RULE dkurokawa_trulens_demo_network_rule
    MODE = EGRESS
    TYPE = HOST_PORT
    VALUE_LIST = (
        'huggingface.co',
        'cdn-lfs-us-1.huggingface.co',
        'snowflake.com',
        'app.snowflake.com',
        'snowflakecomputing.com',
        'sfengineering-mlplatformtest.snowflakecomputing.com',
        'fab02971.snowflakecomputing.com',
        'api.pinecone.io',
        'api.replicate.com',
        'raw.githubusercontent.com',
        'streaming-api.svc.us.c.replicate.net',
        'streamlit-docs-fvvbvd0.svc.aped-4627-b74a.pinecone.io'
    );
SHOW NETWORK RULES;

CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION dkurokawa_trulens_demo_access_integration
    ALLOWED_NETWORK_RULES = (dkurokawa_trulens_demo_network_rule)
    ENABLED = true;
SHOW EXTERNAL ACCESS INTEGRATIONS;
