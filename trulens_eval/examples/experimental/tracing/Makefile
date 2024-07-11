TRACE:= OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED=true \
		opentelemetry-instrument \
		--traces_exporter console \
		--metrics_exporter console \
		--logs_exporter console \
		--service_name example \

run_trace: example.py
	$(TRACE) python example.py

run_notrace: example.py
	python example.py
