# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Set the PYTHONPATH to the current directory
export PYTHONPATH="./"

######## Build Experiment Environment ###########
# Get the current directory of the script
exp_dir=$(cd `dirname $0`; pwd)
# Get the parent directory of the experiment directory
work_dir=$(dirname $(dirname $(dirname $exp_dir)))

# Export environment variables for the working directory and Python path
export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8

# Build the monotonic alignment module
cd $work_dir/modules/monotonic_align
mkdir -p monotonic_align
python setup.py build_ext --inplace
cd $work_dir

if [ -z "$exp_config" ]; then
    exp_config="${exp_dir}/exp_config_clean.json"
fi

echo "Experimental Configuration File: $exp_config"

cuda_id=0

# Set paths (modify these paths to your own)
checkpoint_path="path/to/checkpoint/model.safetensors"
output_dir="path/to/output/directory"
source_path="path/to/source/audio.wav"
reference_path="path/to/reference/audio.wav"

echo "CUDA ID: $cuda_id"
echo "Checkpoint Path: $checkpoint_path"
echo "Output Directory: $output_dir"
echo "Source Audio Path: $source_path"
echo "Reference Audio Path: $reference_path"

# Run the voice conversion inference script
python "${work_dir}/models/vc/Noro/noro_inference.py" \
    --config $exp_config \
    --checkpoint_path $checkpoint_path \
    --output_dir $output_dir \
    --cuda_id ${cuda_id} \
    --source_path $source_path \
    --ref_path $reference_path

