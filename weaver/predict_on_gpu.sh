#!/bin/bash

WEAVER_PATH="${HOME}/SDV-ML/L-GATr/jupyterlab-v4.x_setup/weaver-core/weaver"
OUT_PATH="/scratch-cbe/users/alikaan.gueven/ML_KAAN/lgatr_outputs"


DATA_DIR="/eos/vbc/group/cms/ang.li/ParT_datasets"
files=(${DATA_DIR}/*_test.parquet)

for file in "${files[@]}"; do
	filename=$(basename "$file")
	name_no_ext="${filename%.*}"
	weaver                                                                                                                      \
	  --data-test       $file                                                                                                   \
	  --data-config    "${WEAVER_PATH}/data/LLP.yaml"                                                                           \
	  --model-prefix   "${OUT_PATH}/models/lgatr_kin/20250710-125435_LGATr_config_ranger_lr0.005_batch512_best_epoch_state.pt"  \
	  --network-config "${WEAVER_PATH}/networks/LGATr_config.py" --optimizer-option weight_decay 0.01                           \
	  --batch-size 1024 --start-lr 5e-3 --num-epochs 20 --optimizer ranger                                                       \
	  --num-workers 2 --gpus 0                                                                                                  \
	  --log logs/train_{auto}.log                                                                                               \
	  --predict \
	  --predict-output "${OUT_PATH}/predict2/${name_no_ext}_output.root"
done