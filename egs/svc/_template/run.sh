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

######## Parse the Given Parameters from the Commond ###########
options=$(getopt -o c:n:s --long gpu:,config:,name:,stage:,resume:,resume_from_ckpt_path:,resume_type:,infer_expt_dir:,infer_output_dir:,infer_source_file:,infer_source_audio_dir:,infer_target_speaker:,infer_key_shift:,infer_vocoder_dir: -- "$@")
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

    # [Only for Inference] The experiment dir. The value is like "[Your path to save logs and checkpoints]/[YourExptName]"
    --infer_expt_dir) shift; infer_expt_dir=$1 ; shift ;;
    # [Only for Inference] The output dir to save inferred audios. Its default value is "$expt_dir/result"
    --infer_output_dir) shift; infer_output_dir=$1 ; shift ;;
    # [Only for Inference] The inference source (can be a json file or a dir). For example, the source_file can be "[Your path to save processed data]/[YourDataset]/test.json", and the source_audio_dir can be "$work_dir/source_audio" which includes several audio files (*.wav, *.mp3 or *.flac).
    --infer_source_file) shift; infer_source_file=$1 ; shift ;;
    --infer_source_audio_dir) shift; infer_source_audio_dir=$1 ; shift ;;
    # [Only for Inference] Specify the target speaker you want to convert into. You can refer to "[Your path to save logs and checkpoints]/[Your Expt Name]/singers.json". In this singer look-up table, you can see the usable speaker names (all the keys of the dictionary). For example, for opencpop dataset, the speaker name would be "opencpop_female1".
    --infer_target_speaker) shift; infer_target_speaker=$1 ; shift ;;
    # [Only for Inference] For advanced users, you can modify the trans_key parameters into an integer (which means the semitones you want to transpose). Its default value is "autoshift".
    --infer_key_shift) shift; infer_key_shift=$1 ; shift ;;
    # [Only for Inference] The vocoder dir. Its default value is Amphion/pretrained/bigvgan. See Amphion/pretrained/README.md to download the pretrained BigVGAN vocoders.
    --infer_vocoder_dir) shift; infer_vocoder_dir=$1 ; shift ;;

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
    CUDA_VISIBLE_DEVICES=$gpu python "${work_dir}"/bins/svc/preprocess.py \
        --config $exp_config \
        --num_workers 4
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
        CUDA_VISIBLE_DEVICES="$gpu" accelerate launch "${work_dir}"/bins/svc/train.py \
            --config "$exp_config" \
            --exp_name "$exp_name" \
            --log_level info \
            --resume \
            --resume_from_ckpt_path "$resume_from_ckpt_path" \
            --resume_type "$resume_type"
    else
        echo "Start a new experiment..."
        CUDA_VISIBLE_DEVICES="$gpu" accelerate launch "${work_dir}"/bins/svc/train.py \
            --config "$exp_config" \
            --exp_name "$exp_name" \
            --log_level info
    fi
fi

######## Inference/Conversion ###########
if [ $running_stage -eq 3 ]; then
    if [ -z "$infer_expt_dir" ]; then
        echo "[Error] Please specify the experimental directionary. The value is like [Your path to save logs and checkpoints]/[YourExptName]"
        exit 1
    fi

    if [ -z "$infer_output_dir" ]; then
        infer_output_dir="$expt_dir/result"
    fi

    if [ -z "$infer_source_file" ] && [ -z "$infer_source_audio_dir" ]; then
        echo "[Error] Please specify the source file/dir. The inference source (can be a json file or a dir). For example, the source_file can be "[Your path to save processed data]/[YourDataset]/test.json", and the source_audio_dir should include several audio files (*.wav, *.mp3 or *.flac)."
        exit 1
    fi

    if [ -z "$infer_source_file" ]; then
        infer_source=$infer_source_audio_dir
    fi

    if [ -z "$infer_source_audio_dir" ]; then
        infer_source=$infer_source_file
    fi

    if [ -z "$infer_target_speaker" ]; then
        echo "[Error] Please specify the target speaker. You can refer to "[Your path to save logs and checkpoints]/[Your Expt Name]/singers.json". In this singer look-up table, you can see the usable speaker names (all the keys of the dictionary). For example, for opencpop dataset, the speaker name would be "opencpop_female1""
        exit 1
    fi

    if [ -z "$infer_key_shift" ]; then
        infer_key_shift="autoshift"
    fi

    if [ -z "$infer_vocoder_dir" ]; then
        infer_vocoder_dir="$work_dir"/pretrained/bigvgan
        echo "[Warning] You don't specify the infer_vocoder_dir. It is set $infer_vocoder_dir by default. Make sure that you have followed Amphoion/pretrained/README.md to download the pretrained BigVGAN vocoder checkpoint."
    fi

    CUDA_VISIBLE_DEVICES=$gpu accelerate launch "$work_dir"/bins/svc/inference.py \
        --config $exp_config \
        --acoustics_dir $infer_expt_dir \
        --vocoder_dir $infer_vocoder_dir \
        --target_singer $infer_target_speaker \
        --trans_key $infer_key_shift \
        --source $infer_source \
        --output_dir $infer_output_dir  \
        --log_level debug
fi