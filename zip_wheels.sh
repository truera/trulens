#!/bin/zsh

set -e

# Create function to zip wheel.
zip_wheel() {
    # Print arguments.
    echo "package name: $1"
    echo "zip name: $2"

    # Copy desired wheel over.
    rm -rf ./tmp_build_zip
    mkdir ./tmp_build_zip
    pwd
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

# Build wheels.
rm -rf ./dist
gmake build

# Zip wheels.
zip_wheel trulens-core trulens_core.zip
zip_wheel trulens-feedback trulens_feedback.zip
zip_wheel trulens-providers-cortex trulens_providers_cortex.zip