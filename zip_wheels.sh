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

# Zip wheels.
rm -rf ${OUTPUT_DIRECTORY}
mkdir -p ${OUTPUT_DIRECTORY}
zip_wheel trulens-connectors-snowflake ${OUTPUT_DIRECTORY}/trulens-connectors-snowflake.zip
zip_wheel trulens-core ${OUTPUT_DIRECTORY}/trulens-core.zip
zip_wheel trulens-dashboard ${OUTPUT_DIRECTORY}/trulens-dashboard.zip
zip_wheel trulens-feedback ${OUTPUT_DIRECTORY}/trulens-feedback.zip
zip_wheel trulens-providers-cortex ${OUTPUT_DIRECTORY}/trulens-providers-cortex.zip
