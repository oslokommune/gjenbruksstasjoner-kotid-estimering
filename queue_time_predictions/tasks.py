import re

import luigi
from luigi.contrib.s3 import S3Target

from queue_time_predictions.estimate_queue import estimate_queue
from queue_time_predictions.preprocess_image import image_object, preprocess_image
from queue_time_predictions.util import getenv


class PreprocessImage(luigi.Task):
    """Crop and normalize the image located under `prefix` in S3."""

    prefix = luigi.Parameter()

    def run(self):
        preprocess_image(self.prefix, self.output())

    def output(self):
        image_key = image_object(self.prefix).key
        image_filename_base = image_key.split("/")[-1].split(".")[0]
        path = "/".join(
            [
                getenv("BUCKET_NAME"),
                re.sub("^[^/]+", "intermediate", self.prefix, 1),
                f"{image_filename_base}.bin",
            ]
        )
        return S3Target(f"s3://{path}", format=luigi.format.Nop)


class EstimateQueue(luigi.Task):
    """Estimate queue from image located under `prefix` in S3."""

    prefix = luigi.Parameter()

    def requires(self):
        return PreprocessImage(self.prefix)

    def run(self):
        estimate_queue(self.input())


if __name__ == "__main__":
    luigi.run()
