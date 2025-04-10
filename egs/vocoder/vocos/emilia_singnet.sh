# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

######## Build Experiment Environment ###########
exp_dir=$(cd `dirname $0`; pwd)
work_dir=$(dirname $(dirname $(dirname $exp_dir)))

export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8
 
######## Set Experiment Configuration ###########
exp_config="$exp_dir/emilia_singnet.json"
exp_name="emilia_singnet"

######## Train Model ###########
CUDA_VISIBLE_DEVICES="0" accelerate launch --main_process_port 11111 \
    "${work_dir}"/bins/vocoder/train.py \
    --config=$exp_config \
    --exp_name=$exp_name \
    --log_level debug