import json

from aws_xray_sdk.core import xray_recorder

import queue_time_predictions.estimate_queue as estimate_queue

xray_recorder.begin_segment("Test")


def test_handle():
    response = estimate_queue.handle({}, {})
    body = json.loads(response["body"])
    assert response["statusCode"] == 200
    assert body["message"] == "Not yet implemented"
