from unittest.mock import Mock

import numpy as np
import pytest
from aws_xray_sdk.core import xray_recorder
from moto import mock_dynamodb2
from tensorflow.python.keras.engine.sequential import Sequential

from conftest import predictions_table, region
from queue_time_predictions.estimate_queue import (
    ModelSpec,
    estimate_cars_at_haraldrud,
    estimate_queue,
    estimate_time_in_queue,
    load_model_file,
    parse_image_filename,
    predict,
    read_image,
    write_to_dynamodb,
)

xray_recorder.begin_segment("Test")


def _mock_input_source():
    return Mock(open=Mock(return_value=open("test/data/processed_image.bin", "rb")))


def test_parse_image_filename():
    assert parse_image_filename("station_id_41_20200506T094000.bin") == (41, 1588750800)

    with pytest.raises(ValueError):
        parse_image_filename("41_20200506T094000.bin")

    with pytest.raises(ValueError):
        parse_image_filename("station_id_20200506T094000.bin")

    with pytest.raises(ValueError):
        parse_image_filename("station_id_41_20200506T094000.jpg")


def test_load_model_file():
    model = load_model_file("station_41_queue_full.h5")

    assert isinstance(model, Sequential)


def test_read_image():
    image = read_image(_mock_input_source())

    assert isinstance(image, np.ndarray)
    assert len(image) == 58368


def test_predict():
    model_specs = [
        ModelSpec("foo", None, load_model_file("station_41_queue_full.h5")),
        ModelSpec("bar", None, load_model_file("station_41_queue_lanes.h5")),
    ]
    image = read_image(_mock_input_source())

    predictions = predict(model_specs, image)

    assert set(predictions) == {ms.name for ms in model_specs}
    assert all([isinstance(p, float) for p in predictions.values()])


def test_estimate_cars_at_haraldrud():
    def estimate(queue_end_pos, queue_lanes, queue_full):
        return estimate_cars_at_haraldrud(
            {
                "queue_end_pos": queue_end_pos,
                "queue_lanes": queue_lanes,
                "queue_full": queue_full,
            }
        )

    # No visible queue
    assert estimate(0, 0, 0) == 0

    # Still no visible queue
    assert estimate(100, 0, 0) == 0

    # Queue is visible
    assert estimate(500, 0, 0) > 0

    # An extra lane means more cars
    assert estimate(500, 1, 0) > estimate(500, 0, 0)

    # But it shouldn't make a difference when the queue is invisible
    assert estimate(5, 1, 0) == estimate(5, 0, 0)

    # The result should be the same for a known full queue and an unrealistically long queue
    assert estimate(900, 0, 1) == estimate(100000, 0, 0)


def test_estimate_time_in_queue():
    assert isinstance(estimate_time_in_queue({"cars": 5}), float)
    assert estimate_time_in_queue({"cars": 5}, 70) == 5 / 70
    assert estimate_time_in_queue({"cars": 10}, 1) == 10


@mock_dynamodb2
def test_write_to_dynamodb():
    table = predictions_table()

    write_to_dynamodb(99, 1593672630, {"foo": 1.2, "bar": 3.4}, region)

    result = table.get_item(Key={"station_id": 99, "timestamp": "1593672630"})

    predictions = result.get("Item")
    assert predictions
    assert float(predictions["foo"]) == 1.2
    assert float(predictions["bar"]) == 3.4


@mock_dynamodb2
def test_estimate_queue():
    table = predictions_table()
    input_source = _mock_input_source()
    input_source.path = "foo/bar/station_id_41_20200506T094000.bin"

    estimate_queue(input_source)

    result = table.get_item(Key={"station_id": 41, "timestamp": "1588750800.0"})

    predictions = result.get("Item")
    assert predictions
    assert 0 <= float(predictions["queue_end_pos"])
    assert 0 <= float(predictions["queue_lanes"]) <= 1
    assert 0 <= float(predictions["queue_full"]) <= 1
    assert 0 <= float(predictions["cars"])
    assert 0 <= float(predictions["expected_queue_time"])
