-- Run this script once app is created.

GRANT USAGE ON INTEGRATION <external_access_integration> TO APPLICATION TRULENS_DEMO_NA; -- FILL THIS IN!
SET ENCRYPTED_PASSWORD = (SELECT ENCRYPT('<password>', '<my secret>'));
SELECT $ENCRYPTED_PASSWORD;

CALL TRULENS_DEMO_NA.APP_PUBLIC.START_APP(
    '<account>',
    '<database>',
    '<schema>',
    '<warehouse>',
    '<role>',
    '<user>',
    TO_VARCHAR(DECRYPT($ENCRYPTED_PASSWORD, '<my secret>'), 'utf-8'),
    '<coretex_search_service>',
    '<replicate_token>',
    '<pinecone_api_key>',
    '<external_access_integration>',
); -- FILL THIS IN!

-- Takes a while (~15mins) for the app to start. Use this call to get the service status.
CALL TRULENS_DEMO_NA.APP_PUBLIC.APP_SERVICE_STATUS();
CALL TRULENS_DEMO_NA.APP_PUBLIC.DASHBOARD_SERVICE_STATUS();

-- Use these to get the endpoints/logs, ONLY AFTER THE SERVICE HAS STARTED
CALL TRULENS_DEMO_NA.APP_PUBLIC.SERVICE_APP_ENDPOINTS();
CALL TRULENS_DEMO_NA.APP_PUBLIC.SERVICE_DASHBOARD_ENDPOINTS();
CALL TRULENS_DEMO_NA.APP_PUBLIC.GET_APP_LOGS();
CALL TRULENS_DEMO_NA.APP_PUBLIC.GET_DASHBOARD_LOGS();

-- If necessary, enable debug mode:
-- ALTER APPLICATION TRULENS_DEMO_NA SET DEBUG_MODE = TRUE;
-- ALTER APPLICATION TRULENS_DEMO_NA SET DEBUG_MODE = FALSE;
-- SELECT SYSTEM$GET_SERVICE_STATUS('CORE.TRULENS_DEMO_DASHBOARD');
-- DESC APPLICATION TRULENS_DEMO_NA;
-- SHOW SERVICES IN APPLICATION TRULENS_DEMO_NA;