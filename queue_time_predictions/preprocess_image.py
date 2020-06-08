import io
import os
import pickle

import boto3
import cv2
import numpy as np
from aws_xray_sdk.core import patch_all, xray_recorder
from dataplatform.awslambda.logging import logging_wrapper, log_add
from dataplatform.pipeline.s3 import Config

patch_all()

# TODO: Get this from the task config.

# ROI is the Region of Interest in the image, i.e. the part of the image where
# the interesting information is located. When processing in OpenCV, it is
# normally represented by an array where the shape is [x, 2], meaning a
# selection of points where the first number is the x-coordinate (left to
# right) and the second number is the y-coordinate (top to bottom).
ROI = np.array([[0, 132], [0, 211], [1227, 125], [1075, 101]], dtype=np.int32)

BUCKET_NAME = os.environ["BUCKET_NAME"]


def read_image(bucket_name: str, prefix: str, region_name="eu-west-1") -> tuple:

    """
    Read an image file from S3 found at `prefix` and return it as a cv2-image
    object (really a Numpy array) together with its S3 key.

    Args:
        bucket_name: The bucket name.
        prefix: The key prefix or 'folder' of where the raw image is stored on the S3 bucket.
        region_name : The name of the AWS-region. Default: 'eu-west-1'

    Returns:
        image: The image which has been read.
        obj.key: The S3 key of the image.
    """

    s3_resource = boto3.resource("s3", region_name=region_name)
    bucket = s3_resource.Bucket(bucket_name)

    try:
        obj = next(iter(bucket.objects.filter(Prefix=prefix)))
    except StopIteration:
        log_add(
            error_message=f"No images found at S3 prefix {prefix}.", level="error",
        )
        raise

    image = obj.get().get("Body").read()
    image = cv2.imdecode(np.asarray(bytearray(image)), cv2.IMREAD_COLOR)

    return image, obj.key


def paint_everything_outside_ROI(image: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """
    Paint everything outside the ROI white to remove noise (useless
    information).

    Args:
        image: The image to be painted outside the region of interest, as a Numpy array.

    Returns:
        image: The painted image, as a Numpy array.

    Raises:
        AssertionError: If one of the incoming arguments are not of the correct type.
    """

    assert isinstance(image, np.ndarray)
    assert type(roi) == np.ndarray

    mask = np.ones_like(image) * 255
    mask = cv2.drawContours(mask, [roi], -1, 0, -1)
    image = np.where(mask == 255, mask, image)

    return image


def crop_image(image: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """
    Cropping the image a minimized rectangle which still contains the region of interest.
    The purpose is to reduce network size and useless processing.

    Args:
        image: The image to be cropped, as a Numpy array.

    Returns:
        image: The cropped image, as a Numpy array.

    Raises:
        AssertionError: If one of the incoming arguments are not of the correct type.
    """

    assert type(image) == np.ndarray
    assert type(roi) == np.ndarray

    x_min = roi[:, 0].min()
    x_max = roi[:, 0].max()
    y_min = roi[:, 1].min()
    y_max = roi[:, 1].max()

    image = image[y_min:y_max, x_min:x_max].copy()

    return image


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize the image to values between 0-1, since deep neural networks work
    best with small values.

    Expected input is a numpy array range 0-255.

    Args:
        image : The image, as a Numpy array.

    Returns:
         image : The image, as a Numpy array, with normalized values.

    Raises:
        AssertionError: If the input or output arrays are not within the expected ranges.
    """

    assert isinstance(image, np.ndarray)
    assert image.max() <= 255

    image = image.astype(np.float64)
    image = image * (1.0 / 255)

    assert image.min() >= 0
    assert image.max() <= 1

    return image


def save_data_to_S3(
    obj: object,
    bucket_name: str,
    original_destination_key: str,
    region_name="eu-west-1",
) -> None:
    """
    Saves the binary object to the given S3-location.

    Args:
        obj : Any Python object
        bucket_name : The bucket name.
        original_destination_key : The original key of the object.
        region_name : The name of the AWS-region. Default: 'eu-west-1'

    """

    print("Processing: {0}".format(original_destination_key))
    s3_output_key = original_destination_key.replace(".jpg", ".bin")

    source_stream = io.BytesIO()
    pickle.dump(obj, source_stream)
    source_stream.seek(0)

    s3_client = boto3.client("s3", region_name=region_name)
    s3_client.upload_fileobj(source_stream, bucket_name, s3_output_key)


def crop_and_normalize_image(
    destination_key_prefix: str, bucket_name: str, prefix: str, roi: np.ndarray
) -> None:
    """
    This function reads the image from the given destination in S3, then does
    some initial processing of it to make it more suited to be used for deep
    neural network image analysis. The processed image is then written to S3.

    Args:
        destination_key_prefix : The key prefix or 'folder' of where the processed image will be stored in the S3 bucket.
        bucket_name : The bucket name.
        prefix : The key prefix or 'folder' of where the raw image is stored on the S3 bucket.
        roi : A numpy array with shape (n, 2), defining the region of interest in the raw image.
    """

    image, key = read_image(bucket_name, prefix)
    image = paint_everything_outside_ROI(image, roi)
    cropped_image = crop_image(image, roi)
    normalized_image = normalize_image(cropped_image)
    destination_key = "{0}{1}".format(destination_key_prefix, key.split("/")[-1])

    # Saved as raw numpy array, not a .jpg.
    save_data_to_S3(normalized_image, bucket_name, destination_key)


@logging_wrapper
@xray_recorder.capture("handle")
def handle(event, context):
    config = Config.from_lambda_event(event)

    try:
        input_prefixes = config.payload.step_data.s3_input_prefixes
    except AttributeError:
        log_add(
            error_message=(
                "Malformed event: Expected to find JSON path "
                "`config.payload.step_data.s3_input_prefixes`."
            ),
            event=event,
            level="error",
        )
        raise

    try:
        input_prefix = next(iter(input_prefixes.values()))
    except StopIteration:
        log_add(
            error_message="Malformed event: `input_prefixes` is empty.",
            event=event,
            level="error",
        )
        raise

    try:
        output_prefix = config.payload.output_dataset.s3_prefix.replace(
            "%stage%", "intermediate"
        )
    except AttributeError:
        log_add(
            error_message=(
                "Malformed event: Expected to find JSON path "
                "`config.payload.output_dataset.s3_prefix`."
            ),
            event=event,
            level="error",
        )
        raise

    crop_and_normalize_image(output_prefix, BUCKET_NAME, input_prefix, ROI)
