FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY reddit_app.py .

CMD ["uvicorn", "reddit_app:app", "--host", "0.0.0.0", "--port", "8000"] 