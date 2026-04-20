from __future__ import annotations

import argparse
import json
import os
from urllib import request

from aml.config import get_settings


def _post_json(url: str, payload: dict) -> None:
    req = request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req) as response:
        response.read()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Spark Structured Streaming AML bridge from Kafka to API scoring")
    parser.add_argument("--checkpoint", type=str, default="/tmp/aml_spark_checkpoint")
    args = parser.parse_args()

    settings = get_settings()

    from pyspark.sql import SparkSession

    builder = SparkSession.builder.appName("aml-spark-streaming").config("spark.sql.shuffle.partitions", "2")
    spark_packages = os.getenv("SPARK_EXTRA_PACKAGES", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.2")
    if spark_packages:
        builder = builder.config("spark.jars.packages", spark_packages)
    builder = builder.config("spark.sql.session.timeZone", "UTC")
    spark_master = os.getenv("SPARK_MASTER")
    if spark_master:
        builder = builder.master(spark_master)
    spark = builder.getOrCreate()

    stream_df = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", settings.kafka_bootstrap_servers)
        .option("subscribe", settings.kafka_topic_raw)
        .option("startingOffsets", "earliest")
        .load()
    )

    json_df = stream_df.selectExpr("CAST(value AS STRING) AS payload")

    def process_batch(batch_df, epoch_id):
        for row in batch_df.toLocalIterator():
            payload = json.loads(row.payload)
            _post_json(f"{settings.api_base_url}/score", payload)

    query = (
        json_df.writeStream.foreachBatch(process_batch)
        .option("checkpointLocation", args.checkpoint)
        .start()
    )
    query.awaitTermination()


if __name__ == "__main__":
    main()
