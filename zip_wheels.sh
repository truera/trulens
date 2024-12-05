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

    # Copy desired wheel over.
    rm -rf ./tmp_build_zip
    mkdir ./tmp_build_zip
    cp ./dist/$1/trulens*-py3-none-any.whl ./tmp_build_zip

    # Unzip and zip wheel so that it's compressed as desired. This is a weird hack because the wheels don't have the right compression for w/e reason.
    ZIP_FILE="${OUTPUT_DIRECTORY}/$1.zip"
    rm -f ${ZIP_FILE}
    cd ./tmp_build_zip
    unzip trulens*-py3-none-any.whl
    rm -rf `find ./ -name snowflake_stage_zips`
    rm trulens*-py3-none-any.whl
    zip -r ../${ZIP_FILE} *
    cd ../
    rm -rf ./tmp_build_zip
}

# Zip wheels.
rm -rf ${OUTPUT_DIRECTORY}
mkdir -p ${OUTPUT_DIRECTORY}
zip_wheel trulens-connectors-snowflake
zip_wheel trulens-core
zip_wheel trulens-dashboard
zip_wheel trulens-feedback
zip_wheel trulens-otel-semconv
zip_wheel trulens-providers-cortex
