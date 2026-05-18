#!/usr/bin/env bash

SUITE_VARIANT="${SUITE_VARIANT:-no_rotate}"
SLIDERQUANT_ROTATE="${SLIDERQUANT_ROTATE:-false}"
GPU_ID="${GPU_ID:-1}"
RUN_FP16_BASELINE="${RUN_FP16_BASELINE:-true}"

source "$(dirname "${BASH_SOURCE[0]}")/_sliderquant_text_suite_common.sh"
