FROM python:3.10-slim
WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app.py /app/app.py
COPY data /app/data
COPY clearml.docker.conf /app/clearml.conf

ENV CLEARML_CONFIG_FILE=/app/clearml.conf

EXPOSE 8002
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]