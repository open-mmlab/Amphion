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
exp_config="$exp_dir/exp_config_v2.json"
exp_name="audioldm_debug_latent_size_4_10_78"
checkpoint_path="$work_dir/ckpts/tta/audioldm_debug_latent_size_4_10_78/checkpoints/step-0325000_loss-0.1936.pt"
output_dir="$work_dir/temp"
vocoder_config_path="$work_dir/ckpts/tta/hifigan_checkpoints/config.json"
vocoder_path="$work_dir/ckpts/tta/hifigan_checkpoints/g_01250000"
num_steps=200
guidance_scale=4.0

export CUDA_VISIBLE_DEVICES="0" 

######## Parse Command Line Arguments ###########
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    --text)
    text="$2"
    shift # past argument
    shift # past value
    ;;
    *)    # unknown option
    shift # past argument
    ;;
esac
done

######## Run inference ###########
python "${work_dir}"/bins/tta/inference.py \
    --config=$exp_config \
    --checkpoint_path=$checkpoint_path \
    --text="A man is whistling" \
    --vocoder_path=$vocoder_path \
    --vocoder_config_path=$vocoder_config_path \
    --num_steps=$num_steps \
    --guidance_scale=$guidance_scale \
    --output_dir=$output_dir \