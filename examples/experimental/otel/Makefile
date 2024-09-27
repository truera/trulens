PYTHON:=python3
PIP:=pip3
OTEL:=opentelemetry-instrument --traces_exporter console

run_with_otel:
	$(OTEL) $(PYTHON) dummy.py

run_without_otel:
	$(PYTHON) dummy.py

install:
	$(PIP) install -r requirements.txt

# Start the zipkin docker container:
start_zipkin:
	docker run --rm -d -p 9411:9411 --name zipkin openzipkin/zipkin

# Stop the zipkin docker container:
stop_zipkin:
	docker stop $(docker ps -a -q --filter ancestor=openzipkin/zipkin)
