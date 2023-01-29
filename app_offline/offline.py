import datetime
import os
import shutil

import pandas as pd
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import (
    DecisionTreeClassificationModel,
    DecisionTreeClassifier,
    GBTClassificationModel,
    GBTClassifier,
    LinearSVC,
    LinearSVCModel,
    MultilayerPerceptronClassifier,
    RandomForestClassificationModel,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import SparkSession
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def transform_df(train_spark_df):
    """
    Метод за трансформација на датасетот, враќа assembler и scaler
    """
    all_features = [f[0] for f in train_spark_df.dtypes]
    vector_assembler = VectorAssembler(inputCols=all_features, outputCol="all_features")
    scaler = StandardScaler(
        inputCol="all_features",
        outputCol="scaled_features",
        withStd=True,
        withMean=True,
    )

    return vector_assembler, scaler


def main():
    # Се читаат податоците со pandas dataframe, се зачувуваат
    # во две датотеки како што е наведено во условите

    df = pd.read_csv("../archive/diabetes_binary_health_indicators_BRFSS2015.csv")
    offline_df, online_df = train_test_split(df, test_size=0.2)
    offline_df.to_csv("../data/offline.csv", index=False)
    online_df.to_csv("../data/online.csv", index=False)
    print("Online df size: " + str(online_df.size))
    print("Offline df size: " + str(offline_df.size))

    # Се креира спарк сесија, се вчитува offline.csv и се дели на
    # train и test множество
    spark = SparkSession.builder.appName("Domasna3").getOrCreate()

    df_offline = spark.read.csv("../data/offline.csv", inferSchema=True, header=True)
    (train_spark_df, test_spark_df) = df_offline.randomSplit([0.8, 0.2])

    print("Train spark dataframe: ")
    print(train_spark_df.show())

    # Вршиме трансформација на датасетот со скалирање со стандарден scaler
    # Пред тоа мора да се искористи VectorAssembler
    vector_assembler, scaler = transform_df(train_spark_df)

    # Се дефинираат различни модели и се зачувуваат во models.
    models = []

    # decision tree classifier:
    dt = DecisionTreeClassifier(labelCol="Diabetes_binary", featuresCol="all_features")
    models.append(dt)

    # random forest classifier:
    rf1 = RandomForestClassifier(
        labelCol="Diabetes_binary", featuresCol="all_features", numTrees=10, maxDepth=5
    )
    models.append(rf1)

    rf2 = RandomForestClassifier(
        labelCol="Diabetes_binary", featuresCol="all_features", numTrees=15, maxDepth=15
    )
    models.append(rf2)

    rf3 = RandomForestClassifier(
        labelCol="Diabetes_binary", featuresCol="all_features", numTrees=20, maxDepth=10
    )
    models.append(rf3)

    rf4 = RandomForestClassifier(
        labelCol="Diabetes_binary", featuresCol="all_features", numTrees=30, maxDepth=2
    )
    models.append(rf4)

    # gradient boosted tree classifier:
    gbt1 = GBTClassifier(
        labelCol="Diabetes_binary", featuresCol="all_features", maxIter=10, maxDepth=5
    )
    models.append(gbt1)

    gbt2 = GBTClassifier(
        labelCol="Diabetes_binary", featuresCol="all_features", maxIter=15, maxDepth=15
    )
    models.append(gbt2)

    gbt3 = GBTClassifier(
        labelCol="Diabetes_binary", featuresCol="all_features", maxIter=20, maxDepth=10
    )
    models.append(gbt3)

    gbt4 = GBTClassifier(
        labelCol="Diabetes_binary", featuresCol="all_features", maxIter=30, maxDepth=2
    )
    models.append(gbt4)

    # multilayer perceptron
    # layers = [21, 5, 4, 2]
    # mlp1 = MultilayerPerceptronClassifier(
    #     maxIter=150,
    #     layers=layers,
    #     blockSize=64,
    #     seed=1234,
    #     featuresCol="all_features",
    #     labelCol="Diabetes_binary",
    # )
    # models.append(mlp1)

    # layers = [21, 5, 5, 4, 2]
    # mlp2 = MultilayerPerceptronClassifier(
    #     maxIter=100,
    #     layers=layers,
    #     blockSize=128,
    #     seed=1234,
    #     featuresCol="all_features",
    #     labelCol="Diabetes_binary",
    # )
    # models.append(mlp2)

    # linear support vector machine
    lsvc1 = LinearSVC(
        maxIter=10, regParam=0.1, labelCol="Diabetes_binary", featuresCol="all_features"
    )
    models.append(lsvc1)

    lsvc2 = LinearSVC(
        maxIter=20, regParam=1, labelCol="Diabetes_binary", featuresCol="all_features"
    )
    models.append(lsvc2)

    # Избор на најдобар модел, серијализација на секој следен подобаар
    best_f1 = 0
    id = 0
    for model in tqdm(models):
        pipeline = Pipeline(stages=[vector_assembler, scaler, lsvc1])
        pipelineModel = pipeline.fit(train_spark_df)

        predictions = pipelineModel.transform(test_spark_df)

        evaluator = MulticlassClassificationEvaluator(
            labelCol="Diabetes_binary", predictionCol="prediction", metricName="f1"
        )
        f1 = evaluator.evaluate(predictions)

        print(str(model) + " - F1: " + str(f1))

        if f1 >= best_f1:
            best_f1 = f1
            id += 1
            pipelineModel.write().overwrite().save("../models/" + str(id))

    # На крај, се избира моделот со најголемо id (го зголемуваме за секој следен подобар)
    loaded_model = PipelineModel.load("../models/" + str(id))
    pred = loaded_model.transform(test_spark_df)
    evaluator = MulticlassClassificationEvaluator(
        labelCol="Diabetes_binary", predictionCol="prediction", metricName="f1"
    )
    f1 = evaluator.evaluate(pred)
    print(f1)

    # Се бришат моделите кои не ни се потребни
    for i in range(1, id):
        folder_path = "../models/" + str(i)
        remove_folder(folder_path)


def remove_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"Successfully removed folder: {folder_path}")
    else:
        print(f"Folder does not exist: {folder_path}")


if __name__ == "__main__":
    main()
