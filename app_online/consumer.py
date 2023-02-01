import os

from pyspark.ml import PipelineModel
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

# Се вчитува моделот на почетокот на апликацијата и се гради SparkSession
model = PipelineModel.load("models/11")
spark = SparkSession.builder.appName("Domasna3-consumer").getOrCreate()


df = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", "localhost:9092")
    # .option("host", "localhost")
    # .option("port", 9092)
    .option("subscribe", "health_data")
    .load()
)

print(type(df))

# Се вчитува датасетот и се прави предикција со вчитаниот PipelineModel
df = (
    df.selectExpr("cast(value as string) as json")
    # .select(from_json("json", schema).alias("data"))
    .select("data.*")
    .withColumn(
        "all_features",
        array(
            [
                col("HighBP"),
                col("HighChol"),
                col("CholCheck"),
                col("BMI"),
                col("Smoker"),
                col("Stroke"),
                col("HeartDiseaseorAttack"),
                col("PhysActivity"),
                col("Fruits"),
                col("Veggies"),
                col("HvyAlcoholConsump"),
                col("AnyHealthcare"),
                col("NoDocbcCost"),
                col("GenHlth"),
                col("MentHlth"),
                col("PhysHlth"),
                col("DiffWalk"),
                col("Sex"),
                col("Age"),
                col("Education"),
                col("Income"),
            ]
        ),
    )
    .withColumn(
        "scaled_features", udf(lambda x: Vectors.dense(x), VectorUDT())("features")
    )
    .withColumn("prediction", model.transform(df).select("prediction"))
)


query = (
    df.writeStream.format("kafka")
    .option("kafka.bootstrap.servers", "localhost:9092")
    .option("topic", "health_data_predicted")
    .start()
)

query.awaitTermination()