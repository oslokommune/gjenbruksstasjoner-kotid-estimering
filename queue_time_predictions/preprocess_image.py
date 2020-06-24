import pickle

import boto3
import cv2
import numpy as np
from dataplatform.awslambda.logging import log_add
from keras.applications import VGG16
from luigi.contrib.s3 import S3Target

from queue_time_predictions.util import getenv

# TODO: Get this from the task config.

# ROI is the Region of Interest in the image, i.e. the part of the image where
# the interesting information is located. When processing in OpenCV, it is
# normally represented by an array where the shape is [x, 2], meaning a
# selection of points where the first number is the x-coordinate (left to
# right) and the second number is the y-coordinate (top to bottom).
ROI = np.array([[0, 132], [0, 211], [1227, 125], [1075, 101]], dtype=np.int32)


def read_image(prefix: str, region_name="eu-west-1") -> tuple:

    """
    Read an image file from S3 found at `prefix` and return it as a cv2-image
    object (really a Numpy array) together with its S3 key.

    Args:
        prefix: The key prefix or 'folder' of where the raw image is stored on the S3 bucket.
        region_name : The name of the AWS-region. Default: 'eu-west-1'

    Returns:
        image: The image which has been read.
    """

    s3_resource = boto3.resource("s3", region_name=region_name)
    bucket = s3_resource.Bucket(getenv("BUCKET_NAME"))

    try:
        obj = next(iter(bucket.objects.filter(Prefix=prefix)))
    except StopIteration:
        log_add(
            error_message=f"No images found at S3 prefix {prefix}.", level="error",
        )
        raise

    image = obj.get().get("Body").read()
    image = cv2.imdecode(np.asarray(bytearray(image)), cv2.IMREAD_COLOR)

    return image


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


def save_data_to_S3(obj: object, output_target: S3Target) -> None:
    """
    Save the binary object to the given S3-location.

    Args:
        obj : Any Python object
        output_target : Output stream to write to.
    """

    print("Writing: {0}".format(output_target.path))

    with output_target.open("w") as out:
        pickle.dump(obj, out)


def crop_and_normalize_image(
    input_prefix: str, output_target: S3Target, roi: np.ndarray
) -> None:
    """
    This function reads the image from the given destination in S3, then does
    some initial processing of it to make it more suited to be used for deep
    neural network image analysis. The processed image is then written to S3.

    Args:
        input_prefix : The key prefix or 'folder' of where the raw image is stored on the S3 bucket.
        output_target : Target stream that the processed image is written to.
        roi : A numpy array with shape (n, 2), defining the region of interest in the raw image.
    """

    image = read_image(input_prefix)
    painted_image = paint_everything_outside_ROI(image, roi)
    cropped_image = crop_image(painted_image, roi)
    normalized_image = normalize_image(cropped_image)

    return normalized_image


def get_VGG16_convbase(image_shape):
    """
    Get the convolutional, pre-trained base.

    VGG16 is a convolutional neural network architecture, in this case trained on the ImageNet data set and is freely
    available with Keras. We will only use the convolutional base layers, while leaving out the closely connected layers
    at the end.
    """
    return VGG16(weights="imagenet", include_top=False, input_shape=image_shape)


def run_image_through_VGG16_convbase(image):
    """
    Run `image` through the convolutional base and return the result.

    The output after processing the numpy array through it will later be processed through a custom architecture of
    densely connected layers to produce a prediction.
    """
    convbase = get_VGG16_convbase(image_shape=image.shape)
    return convbase.predict(np.array([image])).flatten()


def preprocess_image(input_prefix, output_target):
    cropped_and_normalized_image = crop_and_normalize_image(
        input_prefix, output_target, ROI
    )
    preprocessed_image = run_image_through_VGG16_convbase(cropped_and_normalized_image)

    # Saved as raw numpy array, not a .jpg.
    save_data_to_S3(preprocessed_image, output_target)
