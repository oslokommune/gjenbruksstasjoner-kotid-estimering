import logging
import re
from dataclasses import dataclass
from decimal import Decimal
from io import BytesIO
from pathlib import Path

import boto3
import numpy as np
from dateutil import tz
from dateutil.parser import isoparse
from tensorflow.keras.models import Sequential, load_model


@dataclass
class ModelSpec:
    """Specifies model name, filename, and their in-memory representation."""

    name: str
    filename: str
    model_object: Sequential = None


model_specs = [
    ModelSpec("queue_end_pos", "station_41_queue_end_pos.h5"),
    ModelSpec("queue_lanes", "station_41_queue_lanes.h5"),
    ModelSpec("queue_full", "station_41_queue_full.h5"),
]


def parse_image_filename(filename):
    """Return station ID and timestamp found by parsing `filename`.

    The filename is expected to be on the form 'station_id_X_T.bin', where X is
    an integer and T is an ISO 8601 timestamp in Europe/Oslo time.
    """
    try:
        m = re.match("station_id_([0-9]+)_([0-9T]+).bin", filename)

        if not m:
            raise ValueError

        groups = m.groups()

        if len(groups) != 2:
            raise ValueError

        station_id = int(groups[0])
        timestamp = (
            isoparse(groups[1]).replace(tzinfo=tz.gettz("Europe/Oslo")).timestamp()
        )

    except Exception:
        logging.error(
            "Error during image filename parsing. Expected filename on the form "
            "'station_id_X_T.bin', where X is an integer and T is an ISO 8601 "
            "timestamp."
        )
        raise

    return station_id, timestamp


def load_model_file(filename, region_name="eu-west-1"):
    """Read and return the model located at `filename`."""

    parent_dir = Path(__file__).parent.absolute()

    return load_model(f"{parent_dir}/models/{filename}")


def read_image(input_source):
    """Read and return an image file (as a Numpy array) from `input_source`."""

    with input_source.open() as f:
        with BytesIO(f.read()) as s:
            return np.load(s, allow_pickle=True)


def predict(model_specs, image):
    """Use all the passed models to predict on the data in an image.

    Return a dictionary of predictions where the model spec name is the key.
    """
    return {
        spec.name: float(spec.model_object.predict(np.array([image])))
        for spec in model_specs
    }


def estimate_cars_at_haraldrud(predictions):
    """Given the predicted position for the end of the queue and the number of
    lanes, return an estimate of the total number of cars in the line.

    TODO: To be improved! This function is highly specialized for Haraldrud and
    needs to be generalized to be applicable to several sites. Haraldrud is
    also a special case since the entry road is more or less a straight line
    and x_pos can be directly mapped to the number of cars.
    """

    # Haraldrud geometry - used to interpolate x_pos to actual meters from
    # point A.
    POINTS = [
        {"x_pos": 33, "meters": 0},
        {"x_pos": 337, "meters": 14.1},
        {"x_pos": 614, "meters": 40.3},
        {"x_pos": 778, "meters": 52.0},
        {"x_pos": 983, "meters": 86.8},
        {"x_pos": 1117, "meters": 139.4},
    ]
    CAR_DENSITY = 0.13  # cars / meter.

    x_pos = predictions["queue_end_pos"]

    # The `queue_lanes` prediction is binary, where 0 means that 1 lane is
    # predicted and 1 means that 2 lanes is predicted.
    lanes = round(predictions["queue_lanes"] + 1)

    assert isinstance(x_pos, (int, float))
    assert x_pos >= 0
    assert isinstance(lanes, (int, float))
    assert lanes >= 1 and lanes <= 2

    meters = np.interp(
        x_pos, [p["x_pos"] for p in POINTS], [p["meters"] for p in POINTS]
    )

    meters = meters * lanes

    if x_pos > 33:
        # If there is a queue, include the non-visible area in front of the
        # gate.
        meters += 34

    return meters * CAR_DENSITY


def estimate_time_in_queue(predictions, inflow_rate=74) -> np.float64:
    """Based on the estimated number of cars in the queue and the inflow_rate, an
    expected time in the queue is returned.

    The inflow rate is given in cars/hr.
    The expected time returned is given in hours.
    """

    return predictions["cars"] / inflow_rate


def write_to_dynamodb(station_id, timestamp, predictions, region="eu-west-1"):
    """Store predictions for a station at a given time in DynamoDB."""
    dynamodb = boto3.resource("dynamodb", region)
    table = dynamodb.Table("gjenbruksstasjoner-estimert-kotid")
    table.update_item(
        Key={"station_id": station_id, "timestamp": str(timestamp)},
        UpdateExpression="set {}".format(
            " ,".join([f"{key} = :{key}" for key in predictions])
        ),
        ExpressionAttributeValues={
            f":{key}": Decimal(str(value)) for key, value in predictions.items()
        },
    )


def estimate_queue(input_source):
    station_id, timestamp = parse_image_filename(input_source.path.split("/")[-1])

    for model_spec in model_specs:
        model_spec.model_object = load_model_file(model_spec.filename)

    image = read_image(input_source)
    predictions = predict(model_specs, image)
    predictions["cars"] = estimate_cars_at_haraldrud(predictions)
    predictions["expected_queue_time"] = estimate_time_in_queue(predictions)

    write_to_dynamodb(station_id, timestamp, predictions)
