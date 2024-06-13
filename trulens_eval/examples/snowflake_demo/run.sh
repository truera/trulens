#!/bin/bash

if [ "${RUN_DASHBOARD}" == "1" ]; then
    echo "Running dashboard..."
    python launch_trulens_dashboard.py
elif [ "${RUN_APP}" == "1" ]; then
    echo "Running app..."
    streamlit run app.py
fi
