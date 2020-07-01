from aws_xray_sdk.core import xray_recorder

xray_recorder.begin_segment("Test")


def test_handle():
    pass
