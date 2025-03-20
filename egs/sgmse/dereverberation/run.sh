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
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:100"

######## Parse the Given Parameters from the Commond ###########
options=$(getopt -o c:n:s --long gpu:,config:,name:,stage:,checkpoint:,resume_type:,main_process_port:,infer_mode:,infer_datasets:,infer_feature_dir:,infer_audio_dir:,infer_expt_dir:,infer_output_dir:,checkpoint_path:,test_dir:,output_dir: -- "$@")
eval set -- "$options"
export CUDA_VISIBLE_DEVICES="0" 
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
    --checkpoint_path) shift; checkpoint_path=$1 ; shift ;;
    --test_dir) shift; test_dir=$1 ; shift ;;
    --output_dir) shift; output_dir=$1 ; shift ;;
    # [Only for Training] The specific checkpoint path that you want to resume from.
    --checkpoint) shift; checkpoint=$1 ; shift ;;
    # [Only for Traiing] `main_process_port` for multi gpu training
    --main_process_port) shift; main_process_port=$1 ; shift ;;

    # [Only for Inference] The inference mode
    --infer_mode) shift; infer_mode=$1 ; shift ;;
    # [Only for Inference] The inferenced datasets
    --infer_datasets) shift; infer_datasets=$1 ; shift ;;
    # [Only for Inference] The feature dir for inference
    --infer_feature_dir) shift; infer_feature_dir=$1 ; shift ;;
    # [Only for Inference] The audio dir for inference
    --infer_audio_dir) shift; infer_audio_dir=$1 ; shift ;;
    # [Only for Inference] The experiment dir. The value is like "[Your path to save logs and checkpoints]/[YourExptName]"
    --infer_expt_dir) shift; infer_expt_dir=$1 ; shift ;;
    # [Only for Inference] The output dir to save inferred audios. Its default value is "$expt_dir/result"
    --infer_output_dir) shift; infer_output_dir=$1 ; shift ;;

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

if [ -z "$main_process_port" ]; then
    main_process_port=29500
fi
echo "Main Process Port: $main_process_port"

######## Features Extraction ###########
if [ $running_stage -eq 1 ]; then
    CUDA_VISIBLE_DEVICES=$gpu python "${work_dir}"/bins/sgmse/preprocess.py \
        --config $exp_config \
        --num_workers 8
fi
######## Training ###########
if [ $running_stage -eq 2 ]; then
    CUDA_VISIBLE_DEVICES=$gpu python "${work_dir}"/bins/sgmse/train_sgmse.py \
        --config "$exp_config" \
        --exp_name "$exp_name" \
        --log_level info
    fi

if [ $running_stage -eq 3 ]; then
     CUDA_VISIBLE_DEVICES=$gpu python "${work_dir}"/bins/sgmse/inference.py \
         --config=$exp_config \
        --checkpoint_path=$checkpoint_path \
         --test_dir="$test_dir" \
       --output_dir=$output_dir
     fi