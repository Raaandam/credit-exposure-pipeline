"""Curation and aggregation logic.

``pandas_impl`` is the reference implementation (local runs and tests); the Spark job in
``exposure_pipeline.spark_jobs.transforms`` mirrors it 1:1 with the DataFrame API. The
same unit-tested business rules (EAD, collateral cap, UNMAPPED buckets) apply in both.
"""
