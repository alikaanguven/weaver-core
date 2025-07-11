#!/bin/bash

# Check if in the correct venv
# ----
if [[ "$VIRTUAL_ENV" != "" ]]
then
  echo "You are in the virtual environment: ${VIRTUAL_ENV}";
else
  echo "Set your virtual environment, and try again.";
  exit 1;
fi

WEAVER_PATH="${HOME}/SDV-ML/L-GATr/jupyterlab-v4.x_setup/weaver-core/weaver"
OUT_PATH="/scratch-cbe/users/alikaan.gueven/ML_KAAN/lgatr_outputs"

DATA_DIR="/eos/vbc/group/cms/ang.li/ParT_datasets"

MODEL="ParT"
# MODEL="LGATr"

weaver                                                                                                     \
  --data-train     "/eos/vbc/group/cms/ang.li/ParT_datasets/stop_M1000_980_ct2_2018_train.parquet"                                                      \
  --data-test      "${DATA_DIR}/stop_*_test.parquet"                                                       \
  --data-config    "${WEAVER_PATH}/data/${MODEL}_kin.yaml"                                                 \
  --model-prefix   "${OUT_PATH}/models/${MODEL}_original_kin/{auto}"                                                \
  --network-config "${WEAVER_PATH}/networks/${MODEL}_config_original.py" --optimizer-option weight_decay 0.01       \
  --batch-size 512 --start-lr 1e-6 --num-epochs 20 --optimizer ranger                                      \
  --num-workers 0  --gpus 0                                                                                \
  --log "${OUT_PATH}/logs/train_{auto}.log"
