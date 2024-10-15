#!/bin/bash

# Copyright 2014  Vassil Panayotov
#           2014  Johns Hopkins University (author: Daniel Povey)
# Modifications Copyright 2019  Nagoya University (author: Takenori Yoshimura)
# Apache 2.0

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <src-dir> <dst-dir>"
  echo "e.g.: $0 /export/a15/vpanayotov/data/LibriTTS/dev-clean data/dev-clean"
  exit 1
fi

src=$1
dst=$2

spk_file=$src/../SPEAKERS.txt

mkdir -p $dst || exit 1

[ ! -d $src ] && echo "$0: no such directory $src" && exit 1
[ ! -f $spk_file ] && echo "$0: expected file $spk_file to exist" && exit 1


wav_scp=$dst/wav.scp; [[ -f "$wav_scp" ]] && rm $wav_scp
trans=$dst/text; [[ -f "$trans" ]] && rm $trans
utt2spk=$dst/utt2spk; [[ -f "$utt2spk" ]] && rm $utt2spk
spk2gender=$dst/spk2gender; [[ -f $spk2gender ]] && rm $spk2gender

for reader_dir in $(find -L $src -mindepth 1 -maxdepth 1 -type d | sed -e "s/$/_/" | sort); do
  reader_dir=$(echo $reader_dir | sed -e "s/_$//")
  reader=$(basename $reader_dir)
  if ! [ $reader -eq $reader ]; then  # not integer.
    echo "$0: unexpected subdirectory name $reader"
    exit 1
  fi

  reader_gender=$(egrep "^$reader[ ]+\|" $spk_file | awk -F'|' '{gsub(/[ ]+/, ""); print tolower($2)}')
  if [ "$reader_gender" != 'm' ] && [ "$reader_gender" != 'f' ]; then
    echo "Unexpected gender: '$reader_gender'"
    exit 1
  fi

  for chapter_dir in $(find -L $reader_dir/ -mindepth 1 -maxdepth 1 -type d | sort); do
    chapter=$(basename $chapter_dir)
    if ! [ "$chapter" -eq "$chapter" ]; then
      echo "$0: unexpected chapter-subdirectory name $chapter"
      exit 1
    fi

    spk="${reader}_${chapter}"

    find -L $chapter_dir/ -iname "*.wav" | sort | while read -r wav_file; do
       id=$(basename $wav_file .wav)
       echo "$id $wav_file" >>$wav_scp

       txt=$(cat $(echo $wav_file | sed -e "s/\.wav$/.normalized.txt/"))
       echo "$id $txt" >>$trans

       # NOTE: For now we are using per-chapter utt2spk. That is each chapter is considered
       #       to be a different speaker. This is done for simplicity and because we want
       #       e.g. the CMVN to be calculated per-chapter
       echo "$id ${reader}_" >>$utt2spk
    done

    # reader -> gender map (again using per-chapter granularity)
    # echo "$spk $reader_gender" >>$spk2gender
  done
  echo "$reader $reader_gender" >>$spk2gender
done

spk2utt=$dst/spk2utt
utils/UniCATS/utils/utt2spk_to_spk2utt.pl <$utt2spk >$spk2utt || exit 1

ntrans=$(wc -l <$trans)
nutt2spk=$(wc -l <$utt2spk)
! [ "$ntrans" -eq "$nutt2spk" ] && \
  echo "Inconsistent #transcripts($ntrans) and #utt2spk($nutt2spk)" && exit 1

utils/UniCATS/utils/validate_data_dir.sh --no-feats $dst || exit 1

echo "$0: successfully prepared data in $dst"

exit 0
