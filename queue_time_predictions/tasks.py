import re

import luigi
from luigi.contrib.s3 import S3Target

from queue_time_predictions.estimate_queue import estimate_queue
from queue_time_predictions.preprocess_image import preprocess_image
from queue_time_predictions.util import getenv


class PreprocessImage(luigi.Task):
    """Crop and normalize the image located under `prefix` in S3."""

    prefix = luigi.Parameter()

    def run(self):
        preprocess_image(self.prefix, self.output())

    def output(self):
        path = "/".join(
            [
                getenv("BUCKET_NAME"),
                re.sub("^[^/]+", "intermediate", self.prefix, 1),
                "processed.bin",
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
