import os

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

service_name = os.path.basename(os.getcwd())

setup(
    name=service_name,
    version="0.1.0",
    author="Origo Dataplattform",
    author_email="dataplattform@oslo.kommune.no",
    description="Lambda functions for estimating queue times",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.oslo.kommune.no/origo-dataplatform/gjenbruksstasjoner-kotid-estimering",
    packages=find_packages(),
    install_requires=[
        "Keras",
        "boto3",
        "dataplatform-common-python",
        "luigi",
        "numpy",
        "opencv-python-headless",
        "python-dateutil",
        "s3fs",
        "tensorflow",
    ],
)
