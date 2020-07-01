#!/bin/bash

luigid --background;
python -m luigi --module queue_time_predictions.tasks EstimateQueue --prefix=$PREFIX;
