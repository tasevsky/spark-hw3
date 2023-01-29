import json
import logging
import random
import time

import pandas as pd
from kafka import KafkaProducer

logging.basicConfig(level=logging.DEBUG)

# Дефинирање на producer.
# без api_version=(0, 11, 5) фрла exception NoBrokersAvailable
producer = KafkaProducer(
    # api_version=(0, 11, 5),
    bootstrap_servers="localhost:9092",
    # bootstrap_servers=kafka_brokers
    security_protocol="PLAINTEXT",
)

# Вчитување на податоците и отфрлање на колоната за класа на пациентот
df = pd.read_csv("data/online.csv")
print(df.head())
df.drop("Diabetes_binary", axis=1, inplace=True)

# Праќање ред по ред во json формат на Apache Kafka topic health_data
for i in range(df.shape[0]):
    record = json.loads(df.iloc[i].to_json(orient="index"))
    print(json.dumps(record))
    # фрла exceopion KafkaTimeoutError.
    producer.send(topic="health_data", value=json.dumps(record).encode("utf-8"))
    time.sleep(random.randint(500, 2000) / 1000.0)