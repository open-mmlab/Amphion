######## Build Experiment Environment ###########
exp_dir=$(cd `dirname $0`; pwd)
work_dir=$(dirname $(dirname $(dirname $exp_dir)))

export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8
 
######## Set Experiment Configuration ###########
exp_config="$exp_dir/prosody_fvq512_6.25hz.json"
exp_name="prosody_fvq512_6.25hz"

####### Train Model ###########
CUDA_VISIBLE_DEVICES="0" accelerate launch --main_process_port 14556 --mixed_precision="bf16" \
    "${work_dir}"/bins/codec/train.py \
    --config=$exp_config \
    --exp_name=$exp_name \
    --log_level debug \
    --dataloader_seed 179