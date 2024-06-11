#!/bin/sh

if [ "${RUN_DASHBOARD}" == "1" ]; then
    echo "Running dashboard..."
    python launch_trulens_dashboard.py &
fi

if [ "${RUN_APP}" == "1" ]; then
    echo "Running app..."
    streamlit run app.py &
fi
