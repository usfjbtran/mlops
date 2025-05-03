FROM python:3.9-slim

WORKDIR /app

RUN pip install mlflow==2.8.0 psycopg2-binary==2.9.9 google-cloud-storage==2.11.0

EXPOSE 8080

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--backend-store-uri", "${POSTGRESQL_URL}", \
     "--default-artifact-root", "${STORAGE_URL}", \
     "--serve-artifacts"] 