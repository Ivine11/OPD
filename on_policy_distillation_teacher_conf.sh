#!/bin/bash

# Teacher-confidence weighted OPD.
# This wrapper keeps the original OPD script as the single source of training
# configuration, and only enables the confidence weighting method.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TEACHER_CONF_WEIGHT_MODE=${TEACHER_CONF_WEIGHT_MODE:-"entropy_exp"}
export TEACHER_CONF_ALPHA=${TEACHER_CONF_ALPHA:-0.2}
export TEACHER_CONF_MIN=${TEACHER_CONF_MIN:-0.2}
export TEACHER_CONF_MAX=${TEACHER_CONF_MAX:-2.0}
export TEACHER_CONF_NORMALIZE=${TEACHER_CONF_NORMALIZE:-True}

bash "$SCRIPT_DIR/on_policy_distillation.sh" "$@"
