import logging
import os, sys
from pyspark.sql import SparkSession

spark_version = '3.2.3'

spark = SparkSession \
    .builder \
    .appName("StructuredStreaming") \
    .getOrCreate()


df = spark \
  .read \
  .format("kafka") \
  .option("kafka.bootstrap.servers", "localhost:9092") \
  .option("subscribe", "health_data") \
  .load()

#df.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)")
#
# kafka_df = spark.readStream \
#     .format("kafka") \
#     .option("kafka.bootstrap.servers", "localhost:9092") \
#     .option("kafka.security.protocol", "SSL") \
#     .option("failOnDataLoss", "false") \
#     .option("subscribe", "health_data") \
#     .option("includeHeaders", "true") \
#     .option("startingOffsets", "latest") \
#     .option("spark.streaming.kafka.maxRatePerPartition", "50") \
#     .load()

# best_model = PipelineModel.load('best_model/best_model')
#
#
# def func_call(df, batch_id):
#     df.selectExpr("CAST(value AS STRING) as json")
#     requests = df.rdd.map(lambda x: x.value).collect()
#     logging.info(requests)
#
# query = kafka_df.writeStream \
#     .format("STREAM TO STREAM") \
#     .foreachBatch(func_call) \
#     .trigger(processingTime="5 seconds") \
#     .start().awaitTermination()