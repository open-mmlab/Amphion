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

cd $work_dir/modules/monotonic_align
mkdir -p monotonic_align
python setup.py build_ext --inplace
cd $work_dir

if [ -z "$exp_config" ]; then
    exp_config="${exp_dir}/exp_config_clean.json"
fi
echo "Experimental Configuration File: $exp_config"

# Set experiment name
exp_name="experiment_name"

# Set CUDA ID
if [ -z "$gpu" ]; then
    gpu="0,1,2,3"
fi

######## Train Model ###########
echo "Experimental Name: $exp_name"

# Specify the checkpoint folder (modify this path to your own)
checkpoint_path="path/to/checkpoint/noro_checkpoint"


# If this is a new experiment, use the following command:
# CUDA_VISIBLE_DEVICES=$gpu accelerate launch --main_process_port 26667 --mixed_precision fp16 \
# "${work_dir}/bins/vc/Noro/train.py" \
#     --config $exp_config \
#     --exp_name $exp_name \
#     --log_level debug

# To resume training or fine-tune from a checkpoint, use the following command:
# Ensure the options --resume, --resume_type resume, and --checkpoint_path are set
CUDA_VISIBLE_DEVICES=$gpu accelerate launch --main_process_port 26667 --mixed_precision fp16 \
"${work_dir}/bins/vc/Noro/train.py" \
    --config $exp_config \
    --exp_name $exp_name \
    --log_level debug \
    --resume \
    --resume_type resume \
    --checkpoint_path $checkpoint_path
