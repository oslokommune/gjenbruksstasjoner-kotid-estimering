import os

import luigi
from luigi.contrib.s3 import S3Target

from queue_time_predictions.preprocess_image import preprocess_image
from queue_time_predictions.util import getenv


class PreprocessImage(luigi.Task):
    """Crop and normalize the image located under `prefix` in S3."""

    prefix = luigi.Parameter()

    def run(self):
        preprocess_image(self.prefix, self.output())

    def output(self):
        path = os.path.join(getenv("BUCKET_NAME"), self.prefix, "processed.bin")
        return S3Target(f"s3://{path}", format=luigi.format.Nop)


if __name__ == "__main__":
    luigi.run()
