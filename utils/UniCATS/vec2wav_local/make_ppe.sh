#!/bin/bash

nj=4
cmd=run.pl

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. utils/UniCATS/utils/parse_options.sh || exit 1;

data=$1
logdir=$2
ppe_dir=$3

mkdir -p $ppe_dir || exit 1;
mkdir -p $logdir || exit 1;

name=$(basename ${data})
scp=${data}/wav.scp
split_scps=""
for n in $(seq $nj); do
  split_scps="$split_scps $logdir/wav.$n.scp"
done
utils/UniCATS/utils/split_scp.pl ${scp} $split_scps || exit 1;

pitch_feats="ark:compute-kaldi-pitch-feats --verbose=2 --config=config//UniCATS_vec2wav/pitch.conf scp,p:$logdir/wav.JOB.scp ark:- | process-kaldi-pitch-feats --add-normalized-log-pitch=false --add-delta-pitch=false --add-raw-log-pitch=true ark:- ark:- |"
energy_feats="ark:compute-mfcc-feats --config=config/UniCATS_vec2wav/mfcc.conf --use-energy=true scp,p:$logdir/wav.JOB.scp ark:- | select-feats 0 ark:- ark:- |"

$cmd JOB=1:$nj ${logdir}/make_ppe.JOB.log \
  paste-feats --length-tolerance=2 "$pitch_feats" "$energy_feats" ark,scp:$ppe_dir/feats.JOB.ark,$ppe_dir/feats.JOB.scp \
   || exit 1;

# concatenate the .scp files together.
for n in $(seq $nj); do
  cat $ppe_dir/feats.$n.scp || exit 1;
done | sort > $ppe_dir/feats.scp
