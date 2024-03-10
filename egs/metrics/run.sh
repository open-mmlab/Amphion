# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

######## Build Experiment Environment ###########
exp_dir=$(cd `dirname $0`; pwd)
work_dir=$(dirname $(dirname $exp_dir))

export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8

######## Parse the Given Parameters from the Commond ###########
options=$(getopt -o c:n:s --long gpu:,reference_folder:,generated_folder:,dump_folder:,metrics:,fs:,align_method:,energy_db_scale:,f0_subtract_mean:,similarity_model:,similarity_mode:,ltr_path:,intelligibility_mode:,language: -- "$@")
eval set -- "$options"

while true; do
  case $1 in
    # Visible GPU machines. The default value is "0".
    --gpu) shift; gpu=$1 ; shift ;;
    # Reference Audio Folder
    --reference_folder) shift; ref_dir=$1 ; shift ;;
    # Generated Audio Folder
    --generated_folder) shift; deg_dir=$1 ; shift ;;
    # Result Dumping Folder
    --dump_folder) shift; dump_dir=$1 ; shift ;;
    # Metrics to Compute
    --metrics) shift; metrics=$1 ; shift ;;
    # Sampling Rate
    --fs) shift; fs=$1 ; shift ;;

    # Method for aligning F0. The default value is "cut"
    --align_method) shift; align_method=$1 ; shift ;;
    # Method for normalizing F0. The default value is "True"
    --f0_subtract_mean) shift; f0_subtract_mean=$1 ; shift ;;
    # Method for normalizing Energy. The default value is "True"
    --energy_db_scale) shift; energy_db_scale=$1 ; shift ;;

    # Model for computing speaker similarity. The default value is "wavlm"
    --similarity_model) shift; similarity_model=$1 ; shift ;;
    # Mode for computing speaker similarity. The default value is "pairwith"
    --similarity_mode) shift; similarity_mode=$1 ; shift ;;
    
    # Path for the transcript.
    --ltr_path) shift; ltr_path=$1 ; shift ;;
    # Mode for computing CER and WER. The default value is "gt_audio"
    --intelligibility_mode) shift; intelligibility_mode=$1 ; shift ;;
    # Language for computing CER and WER. The default value is "english"
    --language) shift; language=$1 ; shift ;;

    --) shift ; break ;;
    *) echo "Invalid option: $1" exit 1 ;;
  esac
done

### Value check ###
if [ -z "$ref_dir" ]; then
    echo "[Error] Please specify the reference_folder"
    exit 1
fi

if [ -z "$deg_dir" ]; then
    echo "[Error] Please specify the generated_folder"
    exit 1
fi

if [ -z "$dump_dir" ]; then
    echo "[Error] Please specify the dump_folder"
    exit 1
fi

if [ -z "$metrics" ]; then
    echo "[Error] Please specify the metrics"
    exit 1
fi

if [ -z "$gpu" ]; then
    gpu="0"
fi

if [ -z "$fs" ]; then
    fs="None"
fi

if [ -z "$align_method" ]; then
    align_method="dtw"
fi

if [ -z "$energy_db_scale" ]; then
    energy_db_scale="True"
fi

if [ -z "$f0_subtract_mean" ]; then
    f0_subtract_mean="True"
fi

if [ -z "$similarity_model" ]; then
    similarity_model="wavlm"
fi

if [ -z "$similarity_mode" ]; then
    similarity_mode="pairwith"
fi

if [ -z "$ltr_path" ]; then
    ltr_path="None"
fi

if [ -z "$intelligibility_mode" ]; then
    intelligibility_mode="gt_audio"
fi

if [ -z "$language" ]; then
    language="english"
fi

######## Calculate Objective Metrics ###########
CUDA_VISIBLE_DEVICES=$gpu python "$work_dir"/bins/calc_metrics.py \
    --ref_dir $ref_dir \
    --deg_dir $deg_dir \
    --dump_dir $dump_dir \
    --metrics $metrics \
    --fs $fs \
    --align_method $align_method \
    --db_scale $energy_db_scale \
    --f0_subtract_mean $f0_subtract_mean \
    --similarity_model $similarity_model \
    --similarity_mode $similarity_mode \
    --ltr_path $ltr_path \
    --intelligibility_mode $intelligibility_mode \
    --language $language