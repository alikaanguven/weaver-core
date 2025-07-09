#!/bin/bash

WEAVER_PATH="/users/alikaan.gueven/SDV-ML/L-GATr/weaver-core/weaver"

# '/eos/vbc/experiments/cms/store/user/lian/CustomNanoAOD_v3/stop_M1000_985_ct20_2018/output/*.root'
# '/eos/vbc/experiments/cms/store/user/lian/CustomNanoAOD_v3/stop_M1000_980_ct2_2018/output/*.root'

weaver                                                                                                                      \
  --data-train     "/eos/vbc/group/cms/ang.li/ParT_datasets/stop_M1000_985_ct20_2018_train.parquet"                         \
  --data-test      "/eos/vbc/group/cms/ang.li/ParT_datasets/stop_M1000_985_ct20_2018_test.parquet"                          \
  --data-config    "${WEAVER_PATH}/data/LLP.yaml"                                                                           \
  --model-prefix   "${WEAVER_PATH}/models/lgatr_kin"                                                                        \
  --network-config "${WEAVER_PATH}/networks/LGATr_config.py" --optimizer-option weight_decay 0.01                           \
  --batch-size 512 --start-lr 5e-3 --num-epochs 20 --optimizer ranger                                                       \
  --num-workers 0 --gpus ""                                                                                                 \
  --log logs/train_{auto}.log