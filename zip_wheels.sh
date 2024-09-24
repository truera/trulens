#!/bin/zsh

# This script creates zips of packages to upload to a Snowflake stage so they
# can be used in stored procedures, SiS, UDFs, and just generally anything run
# on the warehouse.
#
# This should be run before upload and is therefore part of the `Makefile`
# upload target.

set -e

OUTPUT_DIRECTORY="./src/connectors/snowflake/trulens/data/snowflake_stage_zips"

# Create function to zip wheel.
zip_wheel() {
    # Print arguments.
    echo "------------------------------------"
    echo "package name: $1"
    echo "zip name: $2"

    # Copy desired wheel over.
    rm -rf ./tmp_build_zip
    mkdir ./tmp_build_zip
    cp ./dist/$1/trulens*-py3-none-any.whl ./tmp_build_zip

    # Unzip and zip wheel so that it's compressed as desired. This is a weird hack because the wheels don't have the right compression for w/e reason.
    rm -f $2
    cd ./tmp_build_zip
    unzip trulens*-py3-none-any.whl
    rm -rf `find ./ -name snowflake_stage_zips`
    rm trulens*-py3-none-any.whl
    zip -r ../$2 *
    cd ../
    rm -rf ./tmp_build_zip
}

download_nltk_punkt() {
    local nltk_data_dir=$1

    if [ -z "$nltk_data_dir" ]; then
        echo "Please specify the nltk_data_dir as the first argument."
        exit 1
    fi

    # Create the directory if it doesn't exist
    mkdir -p "$nltk_data_dir"

    # Export the NLTK_DATA environment variable to point to the provided directory
    export NLTK_DATA="$nltk_data_dir"

    # Use Python to check if 'punkt_tab' is already downloaded and download it if necessary
    pip install nltk
    python3 - <<EOF
import nltk
from nltk.data import find
import os

nltk_data_dir = os.getenv("NLTK_DATA")

try:
    find('tokenizers/punkt_tab', nltk_data_dir)
    print("punkt_tab tokenizer is already installed in", nltk_data_dir)
except LookupError:
    print("Downloading punkt_tab tokenizer to", nltk_data_dir)
    nltk.download('punkt_tab', download_dir=nltk_data_dir)
EOF
}

# Zip wheels.
rm -rf ${OUTPUT_DIRECTORY}
mkdir -p ${OUTPUT_DIRECTORY}
zip_wheel trulens-connectors-snowflake ${OUTPUT_DIRECTORY}/trulens_connectors_snowflake.zip
zip_wheel trulens-core ${OUTPUT_DIRECTORY}/trulens_core.zip
zip_wheel trulens-feedback ${OUTPUT_DIRECTORY}/trulens_feedback.zip
zip_wheel trulens-providers-cortex ${OUTPUT_DIRECTORY}/trulens_providers_cortex.zip
snow snowpark package create snowflake-sqlalchemy==1.6.1
mv snowflake_sqlalchemy.zip ${OUTPUT_DIRECTORY}
download_nltk_punkt ${OUTPUT_DIRECTORY}/nltk_data
