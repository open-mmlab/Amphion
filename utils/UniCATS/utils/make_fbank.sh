#!/bin/bash

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Begin configuration section.
nj=4
fs=22050
fmax=
fmin=
n_mels=80
n_fft=1024
n_shift=512
win_length=
window=hann
write_utt2num_frames=true
cmd=run.pl
compress=true
normalize=16  # The bit-depth of the input wav files
filetype=mat # mat or hdf5
# End configuration section.

help_message=$(cat <<EOF
Usage: $0 [options] <data-dir> [<log-dir> [<fbank-dir>] ]
e.g.: $0 data/train exp/make_fbank/train mfcc
Note: <log-dir> defaults to <data-dir>/log, and <fbank-dir> defaults to <data-dir>/data
Options:
  --nj <nj>                                        # number of parallel jobs
  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs.
  --filetype <mat|hdf5|sound.hdf5>                 # Specify the format of feats file
EOF
)
echo "$0 $*"  # Print the command line for logging

. parse_options.sh || exit 1;

if [ $# -lt 1 ] || [ $# -gt 3 ]; then
    echo "${help_message}"
    exit 1;
fi

set -euo pipefail

data=$1
if [ $# -ge 2 ]; then
  logdir=$2
else
  logdir=${data}/log
fi
if [ $# -ge 3 ]; then
  fbankdir=$3
else
  fbankdir=${data}/data
fi

# make $fbankdir an absolute pathname.
fbankdir=$(perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' ${fbankdir} ${PWD})

# use "name" as part of name of the archive.
name=$(basename ${data})

mkdir -p ${fbankdir} || exit 1;
mkdir -p ${logdir} || exit 1;

if [ -f ${data}/feats.scp ]; then
  mkdir -p ${data}/.backup
  echo "$0: moving $data/feats.scp to $data/.backup"
  mv ${data}/feats.scp ${data}/.backup
fi

scp=${data}/wav.scp

utils/validate_data_dir.sh --no-text --no-feats ${data} || exit 1;

split_scps=""
for n in $(seq ${nj}); do
    split_scps="${split_scps} ${logdir}/wav.${n}.scp"
done

utils/split_scp.pl ${scp} ${split_scps} || exit 1;

if ${write_utt2num_frames}; then
  write_num_frames_opt="--write-num-frames=ark,t:${logdir}/utt2num_frames.JOB"
else
  write_num_frames_opt=
fi

if [ "${filetype}" == hdf5 ]; then
    ext=h5
else
    ext=ark
fi

if [ -f ${data}/segments ]; then
    echo "$0 [info]: segments file exists: using that."
    split_segments=""
    for n in $(seq ${nj}); do
        split_segments="${split_segments} ${logdir}/segments.${n}"
    done

    utils/split_scp.pl ${data}/segments ${split_segments}

    ${cmd} JOB=1:${nj} ${logdir}/make_fbank_${name}.JOB.log \
        compute-fbank-feats.py \
            --fs ${fs} \
            --fmax ${fmax} \
            --fmin ${fmin} \
            --n_fft ${n_fft} \
            --n_shift ${n_shift} \
            --win_length ${win_length} \
            --window ${window} \
            --n_mels ${n_mels} \
            ${write_num_frames_opt} \
            --compress=${compress} \
            --filetype ${filetype} \
            --normalize ${normalize} \
            --segment=${logdir}/segments.JOB scp:${scp} \
            ark,scp:${fbankdir}/raw_fbank_${name}.JOB.${ext},${fbankdir}/raw_fbank_${name}.JOB.scp

else
  echo "$0: [info]: no segments file exists: assuming pcm.scp indexed by utterance."
  split_scps=""
  for n in $(seq ${nj}); do
    split_scps="${split_scps} ${logdir}/wav.${n}.scp"
  done

  utils/split_scp.pl ${scp} ${split_scps}

  ${cmd} JOB=1:${nj} ${logdir}/make_fbank_${name}.JOB.log \
      compute-fbank-feats.py \
          --fs ${fs} \
          --fmax ${fmax} \
          --fmin ${fmin} \
          --n_fft ${n_fft} \
          --n_shift ${n_shift} \
          --win_length ${win_length} \
          --window ${window} \
          --n_mels ${n_mels} \
          ${write_num_frames_opt} \
          --compress=${compress} \
          --filetype ${filetype} \
          --normalize ${normalize} \
          scp:${logdir}/wav.JOB.scp \
          ark,scp:${fbankdir}/raw_fbank_${name}.JOB.${ext},${fbankdir}/raw_fbank_${name}.JOB.scp
fi


# concatenate the .scp files together.
for n in $(seq ${nj}); do
    cat ${fbankdir}/raw_fbank_${name}.${n}.scp || exit 1;
done > ${data}/feats.scp || exit 1

if ${write_utt2num_frames}; then
    for n in $(seq ${nj}); do
        cat ${logdir}/utt2num_frames.${n} || exit 1;
    done > ${data}/utt2num_frames || exit 1
    rm ${logdir}/utt2num_frames.* 2>/dev/null
fi

rm -f ${logdir}/wav.*.scp ${logdir}/segments.* 2>/dev/null

# Write the filetype, this will be used for data2json.sh
echo ${filetype} > ${data}/filetype

nf=$(wc -l < ${data}/feats.scp)
nu=$(wc -l < ${data}/wav.scp)
if [ ${nf} -ne ${nu} ]; then
    echo "It seems not all of the feature files were successfully ($nf != $nu);"
    echo "consider using utils/fix_data_dir.sh $data"
fi

echo "Succeeded creating filterbank features for $name"
