FROM python:3.10

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip
RUN pip install fastapi uvicorn torch transformers

EXPOSE 8080
ENV PORT 8080

CMD exec uvicorn fastapi_app:app --host 0.0.0.0 --port ${PORT:-8080}