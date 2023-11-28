# Datasets Format

Amphion support the following academic datasets (sort alphabetically):

- [Datasets Format](#datasets-format)
  - [AudioCaps](#audiocaps)
  - [CSD](#csd)
  - [KiSing](#kising)
  - [LibriTTS](#libritts)
  - [LJSpeech](#ljspeech)
  - [M4Singer](#m4singer)
  - [NUS-48E](#nus-48e)
  - [Opencpop](#opencpop)
  - [OpenSinger](#opensinger)
  - [Opera](#opera)
  - [PopBuTFy](#popbutfy)
  - [PopCS](#popcs)
  - [PJS](#pjs)
  - [SVCC](#svcc)
  - [VCTK](#vctk)

The downloading link and the file structure tree of each dataset is displayed as follows.

## AudioCaps

AudioCaps is a dataset of around 44K audio-caption pairs, where each audio clip corresponds to a caption with rich semantic information. You can download the dataset [here](https://github.com/cdjkim/audiocaps). The file structure tree is like:

```plaintext
[AudioCaps dataset path]
┣ AudioCpas
┃   ┣ wav
┃   ┃   ┣ ---1_cCGK4M_0_10000.wav
┃   ┃   ┣ ---lTs1dxhU_30000_40000.wav
┃   ┃   ┣ ...
```

## CSD

The official CSD dataset can be download [here](https://zenodo.org/records/4785016). The file structure tree is like:

```plaintext
[CSD dataset path]
 ┣ english
 ┣ korean
 ┣ utterances
 ┃ ┣ en001a
 ┃ ┃ ┣ {UtterenceID}.wav
 ┃ ┣ en001b
 ┃ ┣ en002a
 ┃ ┣ en002b
 ┃ ┣ ...
 ┣ README
```

## KiSing

The official KiSing dataset can be download [here](http://shijt.site/index.php/2021/05/16/kising-the-first-open-source-mandarin-singing-voice-synthesis-corpus/). The file structure tree is like:

```plaintext
[KiSing dataset path]
 ┣ clean
 ┃ ┣ 421
 ┃ ┣ 422
 ┃ ┣ ...
```

## LibriTTS

The official LibriTTS dataset can be download [here](https://www.openslr.org/60/). The file structure tree is like:

```plaintext
[LibriTTS dataset path]
 ┣ BOOKS.txt
 ┣ CHAPTERS.txt
 ┣ eval_sentences10.tsv
 ┣ LICENSE.txt
 ┣ NOTE.txt
 ┣ reader_book.tsv
 ┣ README_librispeech.txt
 ┣ README_libritts.txt 
 ┣ speakers.tsv
 ┣ SPEAKERS.txt
 ┣ dev-clean (Subset)
 ┃ ┣ 1272{Speaker_ID}
 ┃ ┃ ┣ 128104 {Chapter_ID}
 ┃ ┃ ┃ ┣ 1272_128104_000001_000000.normalized.txt
 ┃ ┃ ┃ ┣ 1272_128104_000001_000000.original.txt
 ┃ ┃ ┃ ┣ 1272_128104_000001_000000.wav
 ┃ ┃ ┃ ┣ ...
 ┃ ┃ ┃ ┣ 1272_128104.book.tsv
 ┃ ┃ ┃ ┣ 1272_128104.trans.tsv
 ┃ ┃ ┣ ...
 ┃ ┣ ...
 ┣ dev-other (Subset)
 ┃ ┣ 116 (Speaker)
 ┃ ┃ ┣ 288045 {Chapter_ID}
 ┃ ┃ ┃ ┣ 116_288045_000003_000000.normalized.txt
 ┃ ┃ ┃ ┣ 116_288045_000003_000000.original.txt
 ┃ ┃ ┃ ┣ 116_288045_000003_000000.wav
 ┃ ┃ ┃ ┣ ...
 ┃ ┃ ┃ ┣ 116_288045.book.tsv
 ┃ ┃ ┃ ┣ 116_288045.trans.tsv
 ┃ ┃ ┣ ...
 ┃ ┣ ...
 ┃ ┣ ...
 ┣ test-clean  (Subset)
 ┃ ┣ {Speaker_ID}
 ┃ ┃ ┣ {Chapter_ID}
 ┃ ┃ ┃ ┣ {Speaker_ID}_{Chapter_ID}_{Utterance_ID}.normalized.txt
 ┃ ┃ ┃ ┣ {Speaker_ID}_{Chapter_ID}_{Utterance_ID}.original.txt
 ┃ ┃ ┃ ┣ {Speaker_ID}_{Chapter_ID}_{Utterance_ID}.wav
 ┃ ┃ ┃ ┣ ...
 ┃ ┃ ┃ ┣ {Speaker_ID}_{Chapter_ID}.book.tsv
 ┃ ┃ ┃ ┣ {Speaker_ID}_{Chapter_ID}.trans.tsv
 ┃ ┃ ┣ ...
 ┃ ┣ ...
 ┣ test-other
 ┃ ┣ {Speaker_ID}
 ┃ ┃ ┣ {Chapter_ID}
 ┃ ┃ ┃ ┣ {Speaker_ID}_{Chapter_ID}_{Utterance_ID}.normalized.txt
 ┃ ┃ ┃ ┣ {Speaker_ID}_{Chapter_ID}_{Utterance_ID}.original.txt
 ┃ ┃ ┃ ┣ {Speaker_ID}_{Chapter_ID}_{Utterance_ID}.wav
 ┃ ┃ ┃ ┣ ...
 ┃ ┃ ┃ ┣ {Speaker_ID}_{Chapter_ID}.book.tsv
 ┃ ┃ ┃ ┣ {Speaker_ID}_{Chapter_ID}.trans.tsv
 ┃ ┃ ┣ ...
 ┃ ┣ ...
 ┣ train-clean-100
 ┃ ┣ {Speaker_ID}
 ┃ ┃ ┣ {Chapter_ID}
 ┃ ┃ ┃ ┣ {Speaker_ID}_{Chapter_ID}_{Utterance_ID}.normalized.txt
 ┃ ┃ ┃ ┣ {Speaker_ID}_{Chapter_ID}_{Utterance_ID}.original.txt
 ┃ ┃ ┃ ┣ {Speaker_ID}_{Chapter_ID}_{Utterance_ID}.wav
 ┃ ┃ ┃ ┣ ...
 ┃ ┃ ┃ ┣ {Speaker_ID}_{Chapter_ID}.book.tsv
 ┃ ┃ ┃ ┣ {Speaker_ID}_{Chapter_ID}.trans.tsv
 ┃ ┃ ┣ ...
 ┃ ┣ ...
 ┣ train-clean-360
 ┃ ┣ {Speaker_ID}
 ┃ ┃ ┣ {Chapter_ID}
 ┃ ┃ ┃ ┣ {Speaker_ID}_{Chapter_ID}_{Utterance_ID}.normalized.txt
 ┃ ┃ ┃ ┣ {Speaker_ID}_{Chapter_ID}_{Utterance_ID}.original.txt
 ┃ ┃ ┃ ┣ {Speaker_ID}_{Chapter_ID}_{Utterance_ID}.wav
 ┃ ┃ ┃ ┣ ...
 ┃ ┃ ┃ ┣ {Speaker_ID}_{Chapter_ID}.book.tsv
 ┃ ┃ ┃ ┣ {Speaker_ID}_{Chapter_ID}.trans.tsv
 ┃ ┃ ┣ ...
 ┃ ┣ ...
 ┣ train-other-500
 ┃ ┣ {Speaker_ID}
 ┃ ┃ ┣ {Chapter_ID}
 ┃ ┃ ┃ ┣ {Speaker_ID}_{Chapter_ID}_{Utterance_ID}.normalized.txt
 ┃ ┃ ┃ ┣ {Speaker_ID}_{Chapter_ID}_{Utterance_ID}.original.txt
 ┃ ┃ ┃ ┣ {Speaker_ID}_{Chapter_ID}_{Utterance_ID}.wav
 ┃ ┃ ┃ ┣ ...
 ┃ ┃ ┃ ┣ {Speaker_ID}_{Chapter_ID}.book.tsv
 ┃ ┃ ┃ ┣ {Speaker_ID}_{Chapter_ID}.trans.tsv
 ┃ ┃ ┣ ...
 ┃ ┣ ...
```


## LJSpeech

The official LibriTTS dataset can be download [here](https://keithito.com/LJ-Speech-Dataset/). The file structure tree is like:

```plaintext
[LJSpeech dataset path]
 ┣ metadata.csv
 ┣ wavs
 ┃ ┣ LJ001-0001.wav
 ┃ ┣ LJ001-0002.wav 
 ┃ ┣ ...
 ┣ README
```

## M4Singer

The official M4Singer dataset can be downloaded [here](https://drive.google.com/file/d/1xC37E59EWRRFFLdG3aJkVqwtLDgtFNqW/view). The file structure tree is like:

```plaintext
[M4Singer dataset path]
 ┣ {Singer_1}#{Song_1}
 ┃ ┣ 0000.mid
 ┃ ┣ 0000.TextGrid
 ┃ ┣ 0000.wav
 ┃ ┣ ...
 ┣ {Singer_1}#{Song_2}
 ┣ ...
 ┣ {Singer_2}#{Song_1}
 ┣ {Singer_2}#{Song_2}
 ┣ ...
 ┗ meta.json
```

## NUS-48E

The official NUS-48E dataset can be download [here](https://drive.google.com/drive/folders/12pP9uUl0HTVANU3IPLnumTJiRjPtVUMx). The file structure tree is like:

```plaintext
[NUS-48E dataset path]
 ┣ {SpeakerID}
 ┃ ┣ read
 ┃ ┃ ┣ {SongID}.txt
 ┃ ┃ ┣ {SongID}.wav
 ┃ ┃ ┣ ...
 ┃ ┣ sing
 ┃ ┃ ┣ {SongID}.txt
 ┃ ┃ ┣ {SongID}.wav
 ┃ ┃ ┣ ...
 ┣ ...
 ┣ README.txt

```

## Opencpop

The official Opera dataset can be downloaded [here](https://wenet.org.cn/opencpop/). The file structure tree is like:

```plaintext
[Opencpop dataset path]
 ┣ midis
 ┃ ┣ 2001.midi
 ┃ ┣ 2002.midi
 ┃ ┣ 2003.midi
 ┃ ┣ ...
 ┣ segments
 ┃ ┣ wavs
 ┃ ┃ ┣ 2001000001.wav
 ┃ ┃ ┣ 2001000002.wav
 ┃ ┃ ┣ 2001000003.wav
 ┃ ┃ ┣ ...
 ┃ ┣ test.txt
 ┃ ┣ train.txt
 ┃ ┗ transcriptions.txt
 ┣ textgrids
 ┃ ┣ 2001.TextGrid
 ┃ ┣ 2002.TextGrid
 ┃ ┣ 2003.TextGrid
 ┃ ┣ ...
 ┣ wavs
 ┃ ┣ 2001.wav
 ┃ ┣ 2002.wav
 ┃ ┣ 2003.wav
 ┃ ┣ ...
 ┣ TERMS_OF_ACCESS
 ┗ readme.md
```

## OpenSinger

The official OpenSinger dataset can be downloaded [here](https://drive.google.com/file/d/1EofoZxvalgMjZqzUEuEdleHIZ6SHtNuK/view). The file structure tree is like:

```plaintext
[OpenSinger dataset path]
 ┣ ManRaw
 ┃ ┣ {Singer_1}_{Song_1}
 ┃ ┃ ┣ {Singer_1}_{Song_1}_0.lab
 ┃ ┃ ┣ {Singer_1}_{Song_1}_0.txt
 ┃ ┃ ┣ {Singer_1}_{Song_1}_0.wav
 ┃ ┃ ┣ ...
 ┃ ┣ {Singer_1}_{Song_2}
 ┃ ┣ ...
 ┣ WomanRaw
 ┣ LICENSE
 ┗ README.md
```

## Opera

The official Opera dataset can be downloaded [here](http://isophonics.net/SingingVoiceDataset). The file structure tree is like:

```plaintext
[Opera dataset path]
 ┣ monophonic
 ┃ ┣ chinese
 ┃ ┃ ┣ {Gender}_{SingerID}
 ┃ ┃ ┃ ┣ {Emotion}_{SongID}.wav
 ┃ ┃ ┃ ┣ ...
 ┃ ┃ ┣ ...
 ┃ ┣ western
 ┣ polyphonic
 ┃ ┣ chinese
 ┃ ┣ western
 ┣ CrossculturalDataSet.xlsx
```

## PopBuTFy

The official PopBuTFy dataset can be downloaded [here](https://github.com/MoonInTheRiver/NeuralSVB). The file structure tree is like:

```plaintext
[PopBuTFy dataset path]
 ┣ data
 ┃ ┣ {SingerID}#singing#{SongName}_Amateur
 ┃ ┃ ┣ {SingerID}#singing#{SongName}_Amateur_{UtteranceID}.mp3
 ┃ ┃ ┣ ...
 ┃ ┣ {SingerID}#singing#{SongName}_Professional
 ┃ ┃ ┣ {SingerID}#singing#{SongName}_Professional_{UtteranceID}.mp3
 ┃ ┃ ┣ ...
 ┣ text_labels
 ┗ TERMS_OF_ACCESS
```

## PopCS

The official PopCS dataset can be downloaded [here](https://github.com/MoonInTheRiver/DiffSinger/blob/master/resources/apply_form.md). The file structure tree is like:

```plaintext
[PopCS dataset path]
 ┣ popcs
 ┃ ┣ popcs-{SongName}
 ┃ ┃ ┣ {UtteranceID}_ph.txt
 ┃ ┃ ┣ {UtteranceID}_wf0.wav
 ┃ ┃ ┣ {UtteranceID}.TextGrid
 ┃ ┃ ┣ {UtteranceID}.txt
 ┃ ┃ ┣ ...
 ┃ ┣ ...
 ┗ TERMS_OF_ACCESS
```

## PJS

The official PJS dataset can be downloaded [here](https://sites.google.com/site/shinnosuketakamichi/research-topics/pjs_corpus). The file structure tree is like:

```plaintext
[PJS dataset path]
 ┣ PJS_corpus_ver1.1
 ┃ ┣ background_noise
 ┃ ┣ pjs{SongID}
 ┃ ┃ ┣ pjs{SongID}_song.wav
 ┃ ┃ ┣ pjs{SongID}_speech.wav
 ┃ ┃ ┣ pjs{SongID}.lab
 ┃ ┃ ┣ pjs{SongID}.mid
 ┃ ┃ ┣ pjs{SongID}.musicxml
 ┃ ┃ ┣ pjs{SongID}.txt
 ┃ ┣ ...
```

## SVCC

The official SVCC dataset can be downloaded [here](https://github.com/lesterphillip/SVCC23_FastSVC/tree/main/egs/generate_dataset). The file structure tree is like:

```plaintext
[SVCC dataset path]
 ┣ Data
 ┃ ┣ CDF1
 ┃ ┃ ┣ 10001.wav
 ┃ ┃ ┣ 10002.wav
 ┃ ┃ ┣ ...
 ┃ ┣ CDM1
 ┃ ┣ IDF1
 ┃ ┣ IDM1
 ┗ README.md
```

## VCTK

The official VCTK dataset can be downloaded [here](https://datashare.ed.ac.uk/handle/10283/3443). The file structure tree is like:

```plaintext
[VCTK dataset path]
 ┣ txt
 ┃ ┣ {Speaker_1}
 ┃ ┃ ┣ {Speaker_1}_001.txt
 ┃ ┃ ┣ {Speaker_1}_002.txt
 ┃ ┃ ┣ ...
 ┃ ┣ {Speaker_2}
 ┃ ┣ ...
 ┣ wav48_silence_trimmed
 ┃ ┣ {Speaker_1}
 ┃ ┃ ┣ {Speaker_1}_001_mic1.flac
 ┃ ┃ ┣ {Speaker_1}_001_mic2.flac
 ┃ ┃ ┣ {Speaker_1}_002_mic1.flac
 ┃ ┃ ┣ ...
 ┃ ┣ {Speaker_2}
 ┃ ┣ ...
 ┣ speaker-info.txt
 ┗ update.txt
```
