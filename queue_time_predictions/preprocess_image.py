import io
import pickle
import sys

import boto3
import cv2
import numpy as np
from aws_xray_sdk.core import patch_all, xray_recorder
from dataplatform.awslambda.logging import logging_wrapper, log_add

patch_all()

# TODO: Get the configurations below from the task config.

# ROI is the Region of Interest in the image, i.e. the part of the image where
# the interesting information is located. When processing in OpenCV, it is
# normally represented by an array where the shape is [x, 2], meaning a
# selection of points where the first number is the x-coordinate (left to
# right) and the second number is the y-coordinate (top to bottom).
ROI = np.array([[0, 132], [0, 211], [1227, 125], [1075, 101]], dtype=np.int32)

BUCKET_NAME = "ok-origo-dataplatform-dev"

DESTINATION_KEY_PREFIX = r"test/espeng-testing-bucket/prediction_testing/cropped_images"


# TODO: Remove this after we start receiving proper events.
def get_image_keys_hardcoded():
    """Hardcoding for now, this will be triggered by an event. The actual images
    are loaded to prod raw/red/REN.
    """

    return [
        "test/espeng-testing-bucket/prediction_testing/raw_images/station_id_41_20200506T094000.jpg",
        "test/espeng-testing-bucket/prediction_testing/raw_images/station_id_41_20200506T095000.jpg",
        "test/espeng-testing-bucket/prediction_testing/raw_images/station_id_41_20200506T100000.jpg",
        "test/espeng-testing-bucket/prediction_testing/raw_images/station_id_41_20200506T101000.jpg",
        "test/espeng-testing-bucket/prediction_testing/raw_images/station_id_41_20200506T102000.jpg",
        "test/espeng-testing-bucket/prediction_testing/raw_images/station_id_41_20200506T103000.jpg",
    ]


def read_image(bucket_name, key, region_name="eu-west-1"):
    """Read an image file from S3 and return it as a cv2-image object (really a
    Numpy array).
    """

    s3_resource = boto3.resource("s3", region_name=region_name)
    bucket = s3_resource.Bucket(bucket_name)

    try:
        image = bucket.Object(key).get().get("Body").read()
    except s3_resource.meta.client.exceptions.NoSuchKey:
        log_add(
            error_message=f"No image found with S3 key {key}.", level="error",
        )
        raise

    image = cv2.imdecode(np.asarray(bytearray(image)), cv2.IMREAD_COLOR)

    return image


def paint_everything_outside_ROI(image, roi):
    """Paint everything outside the ROI white to remove noise (useless
    information)."""

    try:
        assert type(image) == np.ndarray
    except AssertionError:
        print(type(image))
        sys.exit(1)

    assert type(roi) == np.ndarray

    mask = np.ones_like(image) * 255
    mask = cv2.drawContours(mask, [roi], -1, 0, -1)
    image = np.where(mask == 255, mask, image)

    return image


def crop_image(image, roi):
    """Keep a rectangle minimized around the ROI (the reduce network size and
    useless processing).
    """

    assert type(image) == np.ndarray
    assert type(roi) == np.ndarray

    x_min = roi[:, 0].min()
    x_max = roi[:, 0].max()
    y_min = roi[:, 1].min()
    y_max = roi[:, 1].max()

    image = image[y_min:y_max, x_min:x_max].copy()

    return image


def normalize_image(image):
    """Normalize the image, since ANNs works best with small values."""

    image = image.astype(np.float64)
    image = image * (1.0 / 255)

    assert image.min() >= 0
    assert image.max() <= 1

    return image


def save_data_to_S3(
    obj, bucket_name, original_destination_key, region_name="eu-west-1"
):
    """Save the binary object to the given S3-location."""

    print("Processing: {0}".format(original_destination_key))
    s3_output_key = original_destination_key.replace(".jpg", ".bin")

    source_stream = io.BytesIO()
    pickle.dump(obj, source_stream)
    source_stream.seek(0)

    s3_client = boto3.client("s3", region_name=region_name)
    s3_client.upload_fileobj(source_stream, bucket_name, s3_output_key)


def crop_and_normalize_images(destination_key_prefix, bucket_name, src_keys, roi):
    """Push the images through the pipeline and save them in S3."""

    for key in src_keys:
        image = read_image(bucket_name, key)
        image = paint_everything_outside_ROI(image, roi)
        cropped_image = crop_image(image, roi)
        normalized_image = normalize_image(cropped_image)
        destination_key = "{0}/{1}".format(destination_key_prefix, key.split(r"/")[-1])

        # Saved as raw numpy array, not a .jpg.
        save_data_to_S3(normalized_image, bucket_name, destination_key)


@logging_wrapper
@xray_recorder.capture("handle")
def handle(event, context):
    image_keys = get_image_keys_hardcoded()
    crop_and_normalize_images(DESTINATION_KEY_PREFIX, BUCKET_NAME, image_keys, ROI)
