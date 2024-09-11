ARG BASE_IMAGE=python:3.11.9-slim-bullseye
FROM $BASE_IMAGE

COPY ./ /trulens_dashboard/

WORKDIR /trulens_dashboard

RUN pip install -r requirements.txt

CMD ["python", "run_dashboard.py"]
