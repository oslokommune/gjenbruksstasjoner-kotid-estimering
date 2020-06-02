import json

import queue_time_predictions.get_predictions as get_prediction


from aws_xray_sdk.core import xray_recorder

xray_recorder.begin_segment("Test")


def test_handle():
    response = get_prediction.handle({}, {})
    body = json.loads(response["body"])
    assert response["statusCode"] == 200
    assert body["message"] == "Not yet implemented"
