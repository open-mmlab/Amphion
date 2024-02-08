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

mfa_dir=$work_dir/pretrained/mfa
echo $mfa_dir

######## Parse the Given Parameters from the Commond ###########
# options=$(getopt -o c:n:s --long gpu:,config:,infer_expt_dir:,infer_output_dir:,infer_source_file:,infer_source_audio_dir:,infer_target_speaker:,infer_key_shift:,infer_vocoder_dir:,name:,stage: -- "$@")
options=$(getopt -o c:n:s --long gpu:,config:,infer_expt_dir:,infer_output_dir:,infer_mode:,infer_dataset:,infer_testing_set:,infer_text:,name:,stage: -- "$@")
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

    # [Only for Inference] The experiment dir. The value is like "[Your path to save logs and checkpoints]/[YourExptName]"
    --infer_expt_dir) shift; infer_expt_dir=$1 ; shift ;;
    # [Only for Inference] The output dir to save inferred audios. Its default value is "$expt_dir/result"
    --infer_output_dir) shift; infer_output_dir=$1 ; shift ;;
    # [Only for Inference] The inference mode. It can be "batch" to generate speech by batch, or "single" to generage a single clip of speech.
    --infer_mode) shift; infer_mode=$1 ; shift ;;
    # [Only for Inference] The inference dataset. It is only used when the inference model is "batch".
    --infer_dataset) shift; infer_dataset=$1 ; shift ;;
    # [Only for Inference] The inference testing set. It is only used when the inference model is "batch". It can be "test" set split from the dataset, or "golden_test" carefully selected from the testing set.
    --infer_testing_set) shift; infer_testing_set=$1 ; shift ;;
    # [Only for Inference] The text to be synthesized from. It is only used when the inference model is "single". 
    --infer_text) shift; infer_text=$1 ; shift ;;

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
    if [ ! -d "$mfa_dir/montreal-forced-aligner" ]; then
        bash ${exp_dir}/prepare_mfa.sh
    fi
    CUDA_VISIBLE_DEVICES=$gpu python "${work_dir}"/bins/tts/preprocess.py \
        --config=$exp_config \
        --num_workers=4 \
        --prepare_alignment=true
fi

######## Training ###########
if [ $running_stage -eq 2 ]; then
    if [ -z "$exp_name" ]; then
        echo "[Error] Please specify the experiments name"
        exit 1
    fi
    echo "Exprimental Name: $exp_name"

    CUDA_VISIBLE_DEVICES=$gpu accelerate launch "${work_dir}"/bins/tts/train.py \
        --config $exp_config \
        --exp_name $exp_name \
        --log_level debug
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

    if [ "$infer_mode" = "batch" ] && [ -z "$infer_dataset" ]; then
        echo "[Error] Please specify the dataset used in inference when the inference mode is batch"
        exit 1
    fi

    if [ "$infer_mode" = "batch" ] && [ -z "$infer_testing_set" ]; then
        echo "[Error] Please specify the testing set used in inference when the inference mode is batch"
        exit 1
    fi
 
    if [ "$infer_mode" = "single" ] && [ -z "$infer_text" ]; then
        echo "[Error] Please specify the text to be synthesized when the inference mode is single"
        exit 1
    fi

    if [ "$infer_mode" = "single" ]; then
        echo 'Text: ' ${infer_text}
        infer_dataset=None
        infer_testing_set=None
    elif [ "$infer_mode" = "batch" ]; then
        infer_text=''
    fi


    CUDA_VISIBLE_DEVICES=$gpu accelerate launch "$work_dir"/bins/tts/inference.py \
        --config $exp_config \
        --acoustics_dir $infer_expt_dir \
        --output_dir $infer_output_dir  \
        --mode $infer_mode \
        --dataset $infer_dataset \
        --testing_set $infer_testing_set \
        --text "$infer_text" \
        --log_level debug \
        --vocoder_dir /mntnfs/lee_data1/chenxi/processed_data/ljspeech/model_ckpt/hifigan/checkpoints



fi