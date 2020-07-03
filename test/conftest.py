import os

import boto3

region = os.environ["AWS_REGION_NAME"]
table_name = os.environ["PREDICTIONS_TABLE_NAME"]


def predictions_table():
    client = boto3.client("dynamodb", region)
    client.create_table(
        TableName=table_name,
        KeySchema=[
            {"AttributeName": "station_id", "KeyType": "HASH"},
            {"AttributeName": "timestamp", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "station_id", "AttributeType": "N"},
            {"AttributeName": "timestamp", "AttributeType": "S"},
        ],
    )

    return boto3.resource("dynamodb", region_name=region).Table(table_name)
