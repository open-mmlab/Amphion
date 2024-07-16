# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

 
export PYTHONPATH="./"
 


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

# 从这里开始
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export HF_ENDPOINT=https://hf-mirror.com


#镜像 export HF_ENDPOINT=https://hf-mirror.com

if [ -z "$exp_config" ]; then
    exp_config="${exp_dir}"/exp_config_4gpu_clean.json
fi



echo "Exprimental Configuration File: $exp_config"

# hubertnew="/mnt/petrelfs/hehaorui/data/ckpt/vc/newmhubert/model.safetensors"

hubertold="/mnt/data2/hehaorui/ckpt/zs-vc-ckpt/vc_mls_clean/model.safetensors"
whisperold="/mnt/data3/hehaorui/pretrained_models/VC/old_whisper/pytorch_model.bin"
hubert="/mnt/data2/hehaorui/ckpt/vc_new_exp/new_mhubert/checkpoint/epoch-0002_step-0689002_loss-0.571602/model.safetensors"
hubert_se="/mnt/petrelfs/hehaorui/data/ckpt/vc/mhubert-noise-se/checkpoint/epoch-0000_step-0080000_loss-1.515860/pytorch_model.bin"
whisper="/mnt/data2/hehaorui/ckpt/vc_new_exp/new_whisper/checkpoint/epoch-0000_step-0400001_loss-1.194134/model.safetensors"
whisper_se="/mnt/data2/hehaorui/ckpt/vc_new_exp/new_whisper_aug/checkpoint/epoch-0000_step-0468003_loss-2.859798/model.safetensors"
whisper_se_spk="/mnt/data2/hehaorui/ckpt/vc_new_exp/new_whisper_aug_spk/checkpoint/epoch-0000_step-0583003_loss-3.672843/model.safetensors"
hubert_se="/mnt/data2/hehaorui/ckpt/zs-vc-ckpt/epoch-0001_step-0796000_loss-0.567479/model.safetensors"
hubert_se_both="/mnt/data2/hehaorui/ckpt/vc_new_exp/new_mhubert_aug_spk_both/checkpoint/epoch-0001_step-0844000_loss-1.542532/model.safetensors"

hubert_ref_noise="/mnt/data2/hehaorui/ckpt/vc_new_exp/new_mhubert_ref_noise/checkpoint/epoch-0003_step-1171002_loss-1.399222/model.safetensors"

old_ref_noise="/mnt/data2/hehaorui/ckpt/zs-vc-ckpt/vc_mls_libri_robust/model.safetensors"
#模型的
# hubert_clean="xxx"
# hubert_ref_noise="xx"
hubert_both_noise="/mnt/data2/hehaorui/ckpt/vc_new_exp/new_mhubert_both_noise/checkpoint/epoch-0002_step-0741002_loss-1.452612/model.safetensors"

checkpoint_path=$hubert_ref_noise

# gpu的编号：一般用6/7,换卡
cuda_id=1

#prompt就是reference， target就是ground truth
zero_shot_json_file_path="/mnt/data2/hehaorui/datasets/VCTK/zero_shot_json.json" #测试用例的json文件
output_dir="/mnt/data2/hehaorui/exp_out_noro" #
vocoder_path="/mnt/data2/wangyuancheng/model_ckpts/ns2/bigvgan/g_00490000"
wavlm_path="/mnt/data3/hehaorui/pretrained_models/wavlm/wavlm-base-plus-sv"
#加一个ASR模型的path
#用来算WER


echo "CUDA ID: $cuda_id"
echo "Zero Shot Json File Path: $zero_shot_json_file_path"
echo "Checkpoint Path: $checkpoint_path"
echo "Output Directory: $output_dir"
echo "Vocoder Path: $vocoder_path"
echo "WavLM Path: $wavlm_path"


python "${work_dir}"/models/tts/vc/vc_inference.py \
    --config $exp_config \
    --checkpoint_path $checkpoint_path \
    --zero_shot_json_file_path $zero_shot_json_file_path \
    --output_dir $output_dir \
    --cuda_id ${cuda_id} \
    --vocoder_path $vocoder_path \
    --wavlm_path $wavlm_path \
    --ref_noisy \