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
exp_config="$exp_dir/exp_config_latent_4_10_78.json"
exp_name="audioldm_debug_latent_size_4_10_78"

num_workers=8
export CUDA_VISIBLE_DEVICES="0" 

######## Train Model ###########
python "${work_dir}"/bins/tta/train_tta.py \
    --config=$exp_config \
    --num_workers=$num_workers \
    --exp_name=$exp_name \
    --stdout_interval=25 \