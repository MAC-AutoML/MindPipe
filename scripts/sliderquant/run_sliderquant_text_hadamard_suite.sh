#!/usr/bin/env bash

SUITE_VARIANT="${SUITE_VARIANT:-hadamard}"
SLIDERQUANT_ROTATE="${SLIDERQUANT_ROTATE:-true}"
GPU_ID="${GPU_ID:-2}"
RUN_FP16_BASELINE="${RUN_FP16_BASELINE:-false}"
ROTATION_MODE="${ROTATION_MODE:-hadamard}"

source "$(dirname "${BASH_SOURCE[0]}")/_sliderquant_text_suite_common.sh"
