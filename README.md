Gjenbruksstasjoner køtidestimering
==================================

Luigi tasks for predicting queue time estimates for recycling stations in Oslo.

![Queue_time_predictions task flow](doc/queue_time_predictions.png)

## Setup

Requires python >= 3.7
`make init`
`python3 -m venv .venv`
`source .venv/bin/activate`

### Installing global python dependencies

You can either install globally. This might require you to run as root (use sudo).

Requires python >= 3.7
```bash
python3 -m pip install tox black pip-tools
```

Or, you can install for just your user. This is recommended as it does not
require root/sudo, but it does require `~/.local/bin` to be added to `PATH` in
your `.bashrc` or similar file for your shell. Eg:
`PATH=${HOME}/.local/bin:${PATH}`.

```bash
python3 -m pip install --user tox black pip-tools
```


### Installing local python dependencies in a virtualenv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```bash
make init
```


## Tests

Tests are run using [tox](https://pypi.org/project/tox/): `make test`

For tests and linting we use [pytest](https://pypi.org/project/pytest/),
[flake8](https://pypi.org/project/flake8/) and
[black](https://pypi.org/project/black/).

## Running locally

Running the code locally depends on the following environment variable:

```bash
export BUCKET_NAME=ok-origo-dataplatform-dev
```

Start the Luigi task runner, adjusting the `prefix` parameter as needed:

```bash
python -m luigi --module queue_time_predictions.tasks PreprocessImage --prefix=test/my-testing-bucket --local-scheduler
```

## Deploy

TODO.

## Process steps

The steps below describe the main steps in the prediction process.

1. Read the image as a numpy array.
2. Paint everything outside the Region of Interest as white to remove non-valuable information (noise).
3. Crop the image to as small as possible around the Region of Interest.
4. Normalize all image values to 0-1 as this generally improves neural network performance.
5. Process this data through the VGG16 convolutional base (see the docstring for the functions `get_VGG16_convbase` and `run_image_through_VGG16_convbase` for more detail).
6. Process the output from the convolutional base through custom trained densely connected layers.
7. A prediction is now describing where in the image the queue ends (in pixels).
8. The prediction is the mapped to meters from the gate.
9. If there are two lanes, the number of meters from the gate is doubled.
10. *Number of cars* is found by multiplying *Meters (of cars)* with a *car density* constant.
11. *Expected time in queue* is found by dividing *Number of cars* with an *Inflow rate*.

In addition to the traditionally trained machine learning model, several empirically found constants are being used.

TODO: The background for this will be documented in a separate repo.

