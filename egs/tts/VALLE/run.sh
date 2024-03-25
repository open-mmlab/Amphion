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

######## Parse the Given Parameters from the Commond ###########
options=$(getopt -o c:n:s --long gpu:,config:,infer_expt_dir:,ar_model_ckpt_dir:,infer_output_dir:,infer_mode:,infer_test_list_file:,infer_text:,infer_text_prompt:,infer_audio_prompt:,model_train_stage:,name:,stage:,resume:,resume_from_ckpt_path:,resume_type: -- "$@")
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

    # [Only for Training] Model training stage. 
    --model_train_stage) shift; model_train_stage=$1 ; shift ;;
    # [Only for Training] The stage1 ckpt dir. The value is like "[Your path to save logs and checkpoints]/[YourExptName]"
    --ar_model_ckpt_dir) shift; ar_model_ckpt_dir=$1 ; shift ;;

    # [Only for Inference] The experiment dir. The value is like "[Your path to save logs and checkpoints]/[YourExptName]"
    --infer_expt_dir) shift; infer_expt_dir=$1 ; shift ;;
    # [Only for Inference] The output dir to save inferred audios. Its default value is "$expt_dir/result"
    --infer_output_dir) shift; infer_output_dir=$1 ; shift ;;
    
    # [Only for Inference] The inference mode. It can be "batch" to generate speech by batch, or "single" to generage a single clip of speech.
    --infer_mode) shift; infer_mode=$1 ; shift ;;
    # [Only for Inference] The inference test list file. It is only used when the inference model is "batch".
    --infer_test_list_file) shift; infer_test_list_file=$1 ; shift ;;
    # [Only for Inference] The text to be synthesized from. It is only used when the inference model is "single". 
    --infer_text) shift; infer_text=$1 ; shift ;;
    # [Only for Inference] The inference text prompt. It is only used when the inference model is "single".
    --infer_text_prompt) shift; infer_text_prompt=$1 ; shift ;;
    # [Only for Inference] The inference audio prompt. It is only used when the inference model is "single".
    --infer_audio_prompt) shift; infer_audio_prompt=$1 ; shift ;;

    # [Only for Training] Resume configuration
    --resume) shift; resume=$1 ; shift ;;
    # [Only for Training] The specific checkpoint path that you want to resume from.
    --resume_from_ckpt_path) shift; resume_from_ckpt_path=$1 ; shift ;;
    # [Only for Training] `resume` for loading all the things (including model weights, optimizer, scheduler, and random states). `finetune` for loading only the model weights.
    --resume_type) shift; resume_type=$1 ; shift ;;

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
    CUDA_VISIBLE_DEVICES=$gpu python "${work_dir}"/bins/tts/preprocess.py \
        --config=$exp_config \
        --num_workers=4
fi

######## Training ###########
if [ $running_stage -eq 2 ]; then
    if [ -z "$exp_name" ]; then
        echo "[Error] Please specify the experiments name"
        exit 1
    fi

    if [ "$model_train_stage" = "2" ] && [ -z "$ar_model_ckpt_dir" ]; then
        echo "[Error] Please specify the ckeckpoint path to the trained model in stage1."
        exit 1
    fi

    if [  "$model_train_stage" = "1" ]; then
        ar_model_ckpt_dir=None
    fi

    echo "Exprimental Name: $exp_name"

    # Add default value
    if [ -z "$resume_from_ckpt_path" ]; then
        resume_from_ckpt_path=""
    fi

    if [ -z "$resume_type" ]; then
        resume_type="resume"
    fi


    if [ "$resume" = true ]; then
        echo "Resume from the existing experiment..."
        CUDA_VISIBLE_DEVICES=$gpu accelerate launch --main_process_port 29510 \
        "${work_dir}"/bins/tts/train.py \
            --config $exp_config \
            --exp_name $exp_name \
            --log_level debug \
            --train_stage $model_train_stage \
            --ar_model_ckpt_dir $ar_model_ckpt_dir \
            --resume \
            --checkpoint_path "$resume_from_ckpt_path" \
            --resume_type "$resume_type"
    else
        echo "Start a new experiment..."
        CUDA_VISIBLE_DEVICES=$gpu accelerate launch --main_process_port 29510 \
        "${work_dir}"/bins/tts/train.py \
            --config $exp_config \
            --exp_name $exp_name \
            --log_level debug \
            --train_stage $model_train_stage \
            --ar_model_ckpt_dir $ar_model_ckpt_dir
    fi        
fi


######## Inference ###########
if [ $running_stage -eq 3 ]; then
    if [ -z "$infer_expt_dir" ]; then
        echo "[Error] Please specify the experimental directionary. The value is like [Your path to save logs and checkpoints]/[YourExptName]"
        exit 1
    fi

    if [ -z "$infer_output_dir" ]; then
        infer_output_dir="$expt_dir/result"
    fi

    if [ -z "$infer_mode" ]; then
        echo "[Error] Please specify the inference mode, e.g., "batch", "single""
        exit 1
    fi

    if [ "$infer_mode" = "batch" ] && [ -z "$infer_test_list_file" ]; then
        echo "[Error] Please specify the test list file used in inference when the inference mode is batch"
        exit 1
    fi

    if [ "$infer_mode" = "single" ] && [ -z "$infer_text" ]; then
        echo "[Error] Please specify the text to be synthesized when the inference mode is single"
        exit 1
    fi

    if [ "$infer_mode" = "single" ]; then
        echo 'Text: ' ${infer_text}
        infer_test_list_file=None
    elif [ "$infer_mode" = "batch" ]; then
        infer_text=""
        infer_text_prompt=""
        infer_audio_prompt=""
    fi


    CUDA_VISIBLE_DEVICES=$gpu accelerate launch "$work_dir"/bins/tts/inference.py \
        --config $exp_config \
        --log_level debug \
        --acoustics_dir $infer_expt_dir \
        --output_dir $infer_output_dir  \
        --mode $infer_mode \
        --text "$infer_text" \
        --text_prompt "$infer_text_prompt" \
        --audio_prompt $infer_audio_prompt\
        --test_list_file $infer_test_list_file \

fi
