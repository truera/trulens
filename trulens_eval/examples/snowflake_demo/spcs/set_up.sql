USE DATABASE dkurokawa;
USE SCHEMA trulens_demo;
USE WAREHOUSE dkurokawa;

CREATE COMPUTE POOL dkurokawa_trulens_demo_compute_pool
    MIN_NODES = 1
    MAX_NODES = 1
    INSTANCE_FAMILY = CPU_X64_XS;
SHOW COMPUTE POOLS;

CREATE IMAGE REPOSITORY dkurokawa_trulens_demo_image_repository;
SHOW IMAGE REPOSITORIES;

CREATE STAGE dkurokawa_trulens_demo_stage
    DIRECTORY = ( ENABLE = true );
SHOW STAGES;
