export PYTHONPATH="./"

######## Build Experiment Environment ###########
exp_dir="./egs/codecs/FAcodec"
echo exp_dir: $exp_dir
work_dir="./" # Amphion root folder
echo work_dir: $work_dir

export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8

######## Set Config File Dir ##############
if [ -z "$exp_config" ]; then
    exp_config="${exp_dir}"/exp_libritts.json
fi
echo "Exprimental Configuration File: $exp_config"

######## Set the experiment name ##########
exp_name="facodec"

port=53333 # a random number for port

######## Train Model ###########
echo "Experiment Name: $exp_name"
accelerate launch --main_process_port $port "${work_dir}"/bins/codec/train.py --config $exp_config \
--exp_name $exp_name --log_level debug $1
