#!/bin/bash

. egs/tts/UniCATS/CTXvec2wav/cmd.sh || exit 1;
# . ./path.sh || exit 1;

# basic settings
stage=-1              # stage to start
stop_stage=100        # stage to stop
verbose=1             # verbosity level (lower is less info)
world_size=1          # number of workers in training
distributed_init=     # file path for init_process_group in distributed training
nj=16     # number of parallel jobs in feature extraction

exp_dir=$(cd `dirname $0`; pwd)
work_dir=$(dirname $(dirname $(dirname $(dirname $exp_dir))))
cd $work_dir

# NOTE(kan-bayashi): renamed to conf to avoid conflict in parse_options.sh
conf=config/UniCATS_vec2wav/ctxv2w.v1.yaml
sampling_rate=16000        # sampling frequency
num_mels=80     # number of mel basis
hop_size=160    # number of shift points
win_length=465  # window length

# speaker setting
part="all" # "clean" or "all"
             # if set to "clean", use only clean data
             # if set to "all", use clean + other data

# directory path setting
datadir=$work_dir/data
featdir=$work_dir/feats

# training related setting
tag=""     # tag for directory to save model
resume=""  # checkpoint path to resume training
           # (e.g. <path>/<to>/checkpoint-10000steps.pkl)

# decoding related setting
checkpoint=   # checkpoint path to be used for decoding
              # if not provided, the latest one will be used
              # (e.g. <path>/<to>/checkpoint-400000steps.pkl)

train_set="train_${part}" # name of training data directory
dev_set="dev_${part}"           # name of development data directory
# eval_set="eval_${part}"         # name of evaluation data directory
eval_set="eval_${part}"

# shellcheck disable=SC1091
. utils/UniCATS/utils/parse_options.sh || exit 1;

set -eo pipefail
chmod +x models/tts/UniCATS/CTXvec2wav/trainer/train.py models/tts/UniCATS/CTXvec2wav/bin/decode.py

vqdir=feats/vqidx/

# fix path in data/xxx_all/wav.scp and feats/normed_ppe/xxx_all/feats.scp
python utils/UniCATS/utils/fix_path.py


if [ -z "${tag}" ]; then
    expdir="exp/${train_set}_$(basename "${conf}" .yaml)"
else
    expdir="exp/${train_set}_${tag}"
fi

last_checkpoint="$(ls -dt "${expdir}"/*.pkl | head -1 || true)"
if [ -z $resume ]; then
    resume=$last_checkpoint
fi
if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    echo "Stage 2: Network training"
    [ ! -e "${expdir}" ] && mkdir -p "${expdir}"
    echo "Hostname: `hostname`."
    echo "CUDA Devices: $CUDA_VISIBLE_DEVICES"
    echo "Training start. See the progress via ${expdir}/train.log."
    ${cuda_cmd} --gpu 1 "${expdir}/log/train.log" \
        models/tts/UniCATS/CTXvec2wav/trainer/train.py \
            --config "${conf}" \
            --train-wav-scp $datadir/${train_set}/wav.scp \
            --train-vqidx-scp ${featdir}/vqidx/${train_set}/feats.scp \
            --train-mel-scp ${featdir}/normed_fbank/${train_set}/feats.scp \
            --train-aux-scp ${featdir}/normed_ppe/${train_set}/feats.scp \
            --train-num-frames ${datadir}/${train_set}/utt2num_frames \
            --dev-wav-scp ${datadir}/${dev_set}/wav.scp \
            --dev-vqidx-scp ${featdir}/vqidx/${dev_set}/feats.scp \
            --dev-mel-scp ${featdir}/normed_fbank/${dev_set}/feats.scp \
            --dev-aux-scp ${featdir}/normed_ppe/${dev_set}/feats.scp \
            --dev-num-frames $datadir/${dev_set}/utt2num_frames \
            --vq-codebook $vqdir/codebook.npy \
            --outdir "${expdir}" \
            --resume "${resume}" \
            --sampling-rate ${sampling_rate} \
            --hop-size ${hop_size} \
            --num-mels ${num_mels} \
            --win-length ${win_length} \
            --verbose "${verbose}"
    echo "Successfully finished training."
fi

if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    echo "Stage 3: Network decoding"
    # shellcheck disable=SC2012
    [ -z "${checkpoint}" ] && checkpoint="$(ls -dt "${expdir}"/*.pkl | head -1 || true)"
    outdir="${expdir}/synthesis/$(basename "${checkpoint}" .pkl)"
    for name in "${eval_set}"; do
        [ ! -e "${outdir}/${name}" ] && mkdir -p "${outdir}/${name}"

        if [ ! -e "${featdir}/normed_fbank/${name}" ]; then
            mkdir -p "${featdir}/normed_fbank/${name}"
            cat ${featdir}/normed_fbank/{dev_all,eval_all}/feats.scp | filter_scp.pl ${datadir}/${name}/wav.scp - | uniq > ${featdir}/normed_fbank/${name}/feats.scp
        fi
        if [ ! -e "${featdir}/vqidx/${name}" ]; then
            mkdir -p "${featdir}/vqidx/${name}"
            cat ${featdir}/vqidx/{dev_all,eval_all}/feats.scp | filter_scp.pl ${datadir}/${name}/wav.scp - | uniq > ${featdir}/vqidx/${name}/feats.scp
        fi
        feat-to-len.py scp:${featdir}/normed_fbank/${name}/feats.scp > ${datadir}/${name}/utt2num_frames
        echo "$(wc -l ${featdir}/normed_fbank/${name}/feats.scp) utterances for decoding"

        python utils/UniCATS/vec2wav_local/build_prompt_feat.py ${datadir}/${name}/utt2num_frames ${datadir}/${name}/utt2spk ${featdir}/normed_fbank/${name}/feats.scp 300 > ${datadir}/${name}/prompt.scp
        echo "Decoding start. See the progress via ${outdir}/${name}/log/decode.log."
        ${cuda_cmd} --gpu 1 "${outdir}/${name}/log/decode.log" \
            models/tts/UniCATS/CTXvec2wav/bin/decode.py \
                --feats-scp ${featdir}/vqidx/${name}/feats.scp \
                --prompt-scp ${datadir}/${name}/prompt.scp \
                --num-frames ${datadir}/${name}/utt2num_frames \
                --checkpoint "${checkpoint}" \
                --outdir "${outdir}/${name}/wav" \
                --verbose "${verbose}"
        echo "Successfully finished decoding of ${name} set."
    done
    echo "Successfully finished decoding."
fi
echo "Finished."
