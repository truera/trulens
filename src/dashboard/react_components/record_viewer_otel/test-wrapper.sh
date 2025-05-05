#!/bin/sh

# Function to clean up resources
cleanup() {
    echo "Cleaning up resources..."
    if [ -n "$SERVER_PID" ]; then
        echo "Terminating HTTP server (PID: $SERVER_PID)"
        kill $SERVER_PID 2>/dev/null || true
    fi
}

# Set up trap to call cleanup function on script exit
trap cleanup EXIT INT TERM

# Start HTTP server in background and capture its PID
npx http-server storybook-static --port 6006 -s &
SERVER_PID=$!

# Wait for server to be ready
npm exec wait-on -- -t 60000 http://127.0.0.1:6006

# Run tests and capture exit code
npx playwright test --update-snapshots
TEST_EXIT_CODE=$?

# Exit with the same code as the tests
exit $TEST_EXIT_CODE
