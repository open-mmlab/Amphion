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

######## Parse the Given Parameters from the Command ###########
options=$(getopt -o c:n:s --long gpu:,config:,name:,stage:,resume:,resume_from_ckpt_path:,resume_type:,ckpt:,convert:,outdir: -- "$@")
eval set -- "$options"

while true; do
  case $1 in
    # Experimental Configuration File
    -c | --config) shift; exp_config=$1 ; shift ;;
    # Experimental Name
    -n | --name) shift; exp_name=$1 ; shift ;;
    # Running Stage
    -s | --stage) shift; running_stage=$1 ; shift ;;
    # Visible GPU machines. The default value is "0".
    --gpu) shift; gpu=$1 ; shift ;;

    # [Only for Training] Resume configuration
    --resume) shift; resume=$1 ; shift ;;
    # [Only for Training] The specific checkpoint path that you want to resume from.
    --resume_from_ckpt_path) shift; resume_from_ckpt_path=$1 ; shift ;;
    # [Only for Training] `resume` for loading all the things (including model weights, optimizer, scheduler, and random states). `finetune` for loading only the model weights.
    --resume_type) shift; resume_type=$1 ; shift ;;

    # [Only for Inference] The path of saved checkpoint.
    --ckpt) shift; ckpt=$1 ; shift ;;
    # [Only for Inference] The path of convert file
    --convert) shift; convert=$1 ; shift ;;
    # [Only for Inference] The output dir to save inferred audios.
    --outdir) shift; outdir=$1 ; shift ;;
    # [Only for Inference] Whether to use timestamp in the output filename.
    --use_timestamp) shift; use_timestamp=$1 ; shift ;;
    --) shift ; break ;;
    *) echo "Invalid option: $1" exit 1 ;;
  esac
done


### Value check ###
if [ -z "$running_stage" ]; then
    echo "[Error] Please specify the running stage"
    exit 1
fi

if [ -z "$exp_config" ]; then
    exp_config="${exp_dir}"/exp_config.json
fi
echo "Exprimental Configuration File: $exp_config"

if [ -z "$gpu" ]; then
    gpu="0"
fi

######## Features Extraction ###########
if [ $running_stage -eq 1 ]; then
    CUDA_VISIBLE_DEVICES=$gpu python "${work_dir}"/models/vc/FreeVC/preprocess.py \
        --config $exp_config
fi

######## Training ###########
if [ $running_stage -eq 2 ]; then
    if [ -z "$exp_name" ]; then
        echo "[Error] Please specify the experiments name"
        exit 1
    fi
    echo "Exprimental Name: $exp_name"

    # add default value
    if [ -z "$resume_from_ckpt_path" ]; then
        resume_from_ckpt_path=""
    fi

    if [ -z "$resume_type" ]; then
        resume_type="resume"
    fi

    if [ "$resume" = true ]; then
        echo "Resume from the existing experiment..."
        CUDA_VISIBLE_DEVICES="$gpu" python "${work_dir}"/models/vc/FreeVC/train.py \
            --config "$exp_config" \
            --exp_name "$exp_name" \
            --log_level info \
            --resume \
            --resume_from_ckpt_path "$resume_from_ckpt_path" \
            --resume_type "$resume_type"
    else
        echo "Start a new experiment..."
        CUDA_VISIBLE_DEVICES="$gpu" python "${work_dir}"/models/vc/FreeVC/train.py \
            --config "$exp_config" \
            --exp_name "$exp_name" \
            --log_level info
    fi
fi

######## Inference/Conversion ###########
if [ $running_stage -eq 3 ]; then
    CUDA_VISIBLE_DEVICES=$gpu python "$work_dir"/models/vc/FreeVC/inference.py \
        --config $exp_config \
        --ckpt $ckpt \
        --convert $convert \
        --outdir $outdir
fi
