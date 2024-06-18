ARG BASE_IMAGE=python:3.11.9-slim-bullseye
FROM $BASE_IMAGE

COPY ./ /trulens_demo/

WORKDIR /trulens_demo
RUN rm -f ./.env
RUN rm -f ./default.sqlite

RUN pip install -r requirements.txt

CMD ["./run.sh"]
