#!/bin/sh
# Start HTTP server in background and capture its PID
npx http-server storybook-static --port 6006 -s &
SERVER_PID=$!

# Wait for server to be ready
npm exec wait-on -- -t 60000 http://127.0.0.1:6006

# Run tests and capture exit code
npx playwright test
TEST_EXIT_CODE=$?

# Kill HTTP server
kill $SERVER_PID

# Exit with the same code as the tests
exit $TEST_EXIT_CODE
