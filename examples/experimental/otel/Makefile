PYTHON:=python3
PIP:=pip3
OTEL_AUTO:=opentelemetry-instrument
TRULENS_PROVIDER:=trulens.experimental.otel_tracing.core.trace.TracerProvider

# Configure OTEL with environment variables:
# https://opentelemetry.io/docs/languages/sdk-configuration/general/
run_with_otel_env:
	OTEL_PYTHON_TRACER_PROVIDER=$(TRULENS_PROVIDER) \
		OTEL_SERVICE_NAME=trulens_otel_test_env \
		OTEL_TRACES_EXPORTER=zipkin \
		OTEL_METRICS_EXPORTER=none \
		$(OTEL_AUTO) $(PYTHON) dummy.py

# Configure OTEL with command line arguments:
# https://opentelemetry.io/docs/zero-code/python/configuration/
run_with_otel_cmd:
	$(OTEL_AUTO) \
		--tracer_provider $(TRULENS_PROVIDER) \
		--service_name trulens_otel_test_cmd \
		--traces_exporter zipkin \
		--metrics_exporter none \
		$(PYTHON) dummy.py

run_without_otel:
	$(PYTHON) dummy.py

install:
	$(PIP) install -r requirements.txt

# Start collector:
start_collector:
	docker run -p 4317:4317 \
		-v otel-collector-config.yaml:/etc/otel-collector-config.yaml \
		otel/opentelemetry-collector:latest \
		--config=/etc/otel-collector-config.yaml

# Start the zipkin docker container:
start_zipkin:
	docker run --rm -d -p 9411:9411 --name zipkin openzipkin/zipkin

# Stop the zipkin docker container:
stop_zipkin:
	docker stop $(docker ps -a -q --filter ancestor=openzipkin/zipkin)
