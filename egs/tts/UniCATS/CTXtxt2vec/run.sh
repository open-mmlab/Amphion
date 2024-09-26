exp_dir=$(cd `dirname $0`; pwd)
work_dir=$(dirname $(dirname $(dirname $(dirname $exp_dir))))

export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8

cd $work_dir
# python utils/UniCATS/utils/merge_json.py

echo "Working Directory: $work_dir"

######## Parse the Given Parameters from the Command ###########
options=$(getopt -o n:s: --long name:,stage:,gpu: -- "$@")
eval set -- "$options"

while true; do
  case $1 in
    # Experimental Name
    -n | --name) shift; exp_name=$1 ; shift ;;
    # Running Stage
    -s | --stage) shift; running_stage=$1 ; shift ;;
    # Visible GPU machines. The default value is "0".
    --gpu) shift; gpu=$1 ; shift ;;

    --) shift ; break ;;
    *) echo "Invalid option: $1"; exit 1 ;;
  esac
done

### Value check ###
if [ -z "$running_stage" ]; then
    echo "[Error] Please specify the running stage"
    exit 1
fi

if [ -z "$exp_name" ]; then
    echo "[Error] Please specify the experiment name"
    exit 1
fi

if [ -z "$gpu" ]; then
    gpu="0"
fi

######## Training ###########
if [ "$running_stage" -eq 2 ]; then
    echo "Starting Stage 2: Training"
    CUDA_VISIBLE_DEVICES=$gpu python models/tts/UniCATS/CTXtxt2vec/trainer/train.py \
        --name $exp_name \
        --config_file config/UniCATS_txt2vec.json \
        --num_node 1 \
        --tensorboard \
        --auto_resume
fi

######## Inference ###########
if [ "$running_stage" -eq 3 ]; then
    echo "Starting Stage 3: Inference"
    CUDA_VISIBLE_DEVICES=$gpu python models/tts/UniCATS/CTXtxt2vec/inference/continuation.py \
        --eval-set eval_all
fi