# Airflow + Spark client image. Spark runs in local[*] mode inside this container by
# default, or submits to the standalone cluster when SPARK_MASTER_URL points at it.
FROM apache/airflow:2.10.3-python3.11

USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends openjdk-17-jre-headless \
 && apt-get clean && rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

USER airflow
COPY docker/airflow/requirements-airflow.txt /tmp/requirements-airflow.txt
# Install against Airflow's constraints so provider/library versions stay compatible.
RUN pip install --no-cache-dir -r /tmp/requirements-airflow.txt \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.10.3/constraints-3.11.txt"

# Pre-fetch hadoop-azure + postgres JDBC jars at build time so DAG runs don't hit Maven.
ARG SPARK_JARS_PACKAGES=org.apache.hadoop:hadoop-azure:3.3.4,com.microsoft.azure:azure-storage:8.6.6,org.postgresql:postgresql:42.7.3
RUN python -c "from pyspark.sql import SparkSession as S; S.builder.master('local[1]').config('spark.jars.packages', '${SPARK_JARS_PACKAGES}').getOrCreate().stop()"
