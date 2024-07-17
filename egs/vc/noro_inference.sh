# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

 
export PYTHONPATH="./"
 
######## Build Experiment Environment ###########
exp_dir=$(cd `dirname $0`; pwd)
work_dir=$(dirname $(dirname $exp_dir))

export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8

 
cd $work_dir/modules/monotonic_align
mkdir -p monotonic_align
python setup.py build_ext --inplace
cd $work_dir


if [ -z "$exp_config" ]; then
    exp_config="${exp_dir}"/exp_config_testing.json
fi

echo "Exprimental Configuration File: $exp_config"

hubert="/mnt/data2/hehaorui/ckpt/vc_new_exp/new_mhubert/checkpoint/epoch-0002_step-0689002_loss-0.571602/model.safetensors"

checkpoint_path=$hubert

cuda_id=0

output_dir="/home/hehaorui/code/Amphion-1/egs/vc" #
source_path="/home/hehaorui/code/Amphion-1/egs/vc/p233_001.wav"
reference_path="/home/hehaorui/code/Amphion-1/egs/vc/p275_425.wav"

echo "CUDA ID: $cuda_id"
echo "Zero Shot Json File Path: $zero_shot_json_file_path"
echo "Checkpoint Path: $checkpoint_path"
echo "Output Directory: $output_dir"


python "${work_dir}"/models/vc/noro_inference.py \
    --config $exp_config \
    --checkpoint_path $checkpoint_path \
    --output_dir $output_dir \
    --cuda_id ${cuda_id} \
    --source_path $source_path \
    --ref_path $reference_path \
