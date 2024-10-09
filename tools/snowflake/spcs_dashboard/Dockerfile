ARG BASE_IMAGE=python:3.11.9-slim-bullseye
FROM $BASE_IMAGE

COPY ./ /trulens_dashboard/

WORKDIR /trulens_dashboard

RUN pip install -r requirements.txt
RUN pip install trulens_connectors_snowflake-1.0.1-py3-none-any.whl
RUN pip install trulens_dashboard-1.0.1-py3-none-any.whl

CMD ["python", "run_dashboard.py"]
