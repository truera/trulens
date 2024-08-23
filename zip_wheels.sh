#!/bin/zsh

set -e

OUTPUT_DIRECTORY="./src/core/trulens/data/snowflake_stage_zips"

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
    rm trulens*-py3-none-any.whl
    zip -r ../$2 *
    cd ../
    rm -rf ./tmp_build_zip
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
