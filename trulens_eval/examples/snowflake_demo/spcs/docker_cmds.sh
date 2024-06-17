#!/bin/bash

set -e

# Parameters.
ORG_NAME="sfengineering"
ACCOUNT_NAME="mlplatformtest"
DB_NAME="dkurokawa"
SCHEMA_NAME="trulens_demo"
IMAGE_REPOSITORY_NAME="dkurokawa_trulens_demo_image_repository"
IMAGE_NAME="trulens_demo"
IMAGE_TAG="latest"
USER="dkurokawa"

# Convenience variables.
REPOSITORY_URL="${ORG_NAME}-${ACCOUNT_NAME}.registry.snowflakecomputing.com/${DB_NAME}/${SCHEMA_NAME}/${IMAGE_REPOSITORY_NAME}"

# Build the wheel.
pushd ../../
rm -rf build
rm -rf dist
make build
popd

# Copy the wheel.
cp ../../dist/trulens_eval-*-py3-none-any.whl ./

# Build the docker image.
docker build --platform linux/amd64 -t ${REPOSITORY_URL}/${IMAGE_NAME}:${IMAGE_TAG} .

# Log in to SPCS (will require password).
docker login ${ORG_NAME}-${ACCOUNT_NAME}.registry.snowflakecomputing.com -u ${USER}

# Push image.
docker push ${REPOSITORY_URL}/${IMAGE_NAME}:${IMAGE_TAG}
