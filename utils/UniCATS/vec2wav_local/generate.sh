#!/bin/bash

sampling_rate=16000
train_cmd=utils/UniCATS/utils/run.pl
train_set=train_all
featdir=feats
conf=config/hifigan.v1.yaml

# pretrained model
expdir=exp/${train_set}_$(basename "${conf}" .yaml)
checkpoint="$(ls -dt "${expdir}"/*.pkl | head -1 || true)"

stage=0
stop_stage=100

# generation
eval_dir=

. ./path.sh || exit 1;
. utils/UniCATS/utils/parse_options.sh || exit 1;

outdir=${eval_dir}/$(basename "${conf}" .yaml)

echo model: ${checkpoint}
echo out dir: ${outdir}

if [ ${stage} -le 0  ] && [ ${stop_stage} -ge 0 ]; then
    echo ========== Prepare utt2num_frames ==========
    feat-to-len scp:${eval_dir}/feats.scp ark,t:${eval_dir}/utt2num_frames || exit 1
fi

if [ ${stage} -le 1  ] && [ ${stop_stage} -ge 1 ]; then
    mkdir -p ${outdir}/log
    echo ========== Vocoder Generation ==========
    # sbatch -p gpu,2080ti -c 4 --mem=10G --ntasks-per-node 1 --gres=gpu:1 steps/generate.sh --stage 1
    ${train_cmd} --gpu 1 "${outdir}/log/generation.log" \
        decode.py \
            --feats-scp ${eval_dir}/feats.scp \
            --prompt-scp ${eval_dir}/prompt.scp \
            --num-frames ${eval_dir}/utt2num_frames \
            --checkpoint "${checkpoint}" \
            --outdir ${outdir}/wav \
            --verbose 1
    echo "Finished."
fi
