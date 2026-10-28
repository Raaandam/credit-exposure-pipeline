FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
# The dashboard does not need Spark; skip the heavy pyspark wheel.
RUN grep -v -E '^(pyspark|pytest)' requirements.txt > req-dashboard.txt \
 && pip install --no-cache-dir -r req-dashboard.txt
COPY src ./src
COPY dashboard ./dashboard
ENV PYTHONPATH=/app/src
EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
