FROM python:3.12

WORKDIR /app
COPY . /app

RUN pip install fastapi uvicorn scikit-learn pandas joblib pydantic mlflow

EXPOSE 8000

CMD ["uvicorn", "scripts.api:app", "--host", "0.0.0.0", "--port", "8000"]
