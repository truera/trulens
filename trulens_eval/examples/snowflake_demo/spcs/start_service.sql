USE ROLE engineer;
USE DATABASE dkurokawa;
USE SCHEMA trulens_demo;
USE WAREHOUSE dkurokawa;

DESCRIBE COMPUTE POOL dkurokawa_trulens_demo_compute_pool;

DROP SERVICE dkurokawa_trulens_demo;
CREATE SERVICE dkurokawa_trulens_demo
  IN COMPUTE POOL dkurokawa_trulens_demo_compute_pool
  FROM SPECIFICATION $$
    spec:
      containers:
      - name: dkurokawa-trulens-demo-container
        image: /dkurokawa/trulens_demo/dkurokawa_trulens_demo_image_repository/streamlit_app:latest
        env:
            SF_ACCOUNT: "fab02971"
            SF_DB_NAME: "dkurokawa"
            SF_SCHEMA: "trulens_demo"
            SF_WAREHOUSE: "dkurokawa"
            SF_ROLE: "engineer"
        secrets:
          - snowflakeSecret: dkurokawa.trulens_demo.login_credentials
            secretKeyRef: username
            envVarName: SF_USER
          - snowflakeSecret: dkurokawa.trulens_demo.login_credentials
            secretKeyRef: password
            envVarName: SF_PASSWORD
          - snowflakeSecret: dkurokawa.trulens_demo.replicate_api_token
            secretKeyRef: secret_string
            envVarName: REPLICATE_API_TOKEN
          - snowflakeSecret: dkurokawa.trulens_demo.pinecone_api_token
            secretKeyRef: secret_string
            envVarName: PINECONE_API_KEY
      endpoints:
      - name: trulens-demo-endpoint
        port: 8501
        public: true
      $$
   MIN_INSTANCES=1
   MAX_INSTANCES=1;

SHOW SERVICES;
SELECT SYSTEM$GET_SERVICE_STATUS('dkurokawa_trulens_demo');
DESCRIBE SERVICE dkurokawa_trulens_demo;
SHOW ENDPOINTS IN SERVICE dkurokawa_trulens_demo;

CALL SYSTEM$GET_SERVICE_LOGS('dkurokawa_trulens_demo', '0', 'dkurokawa-trulens-demo-container', 100);
