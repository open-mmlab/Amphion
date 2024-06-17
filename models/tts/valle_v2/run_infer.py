# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import random
import glob
import librosa

# from utils.g2p.g2p import phonemizer_g2p as g2p
from .g2p_processor import G2pProcessor

g2p = G2pProcessor()  # use g2p_en as g2p

import os
import torchaudio
import re
import numpy as np
import shutil

SAMPLE_RATE = 16000

test_wer = True
test_sim = True
test_fid = False


class WER:
    def __init__(self):
        print("Loading WER")
        from transformers import Wav2Vec2Processor, HubertForCTC

        from evaluate import load

        wer = load("wer")

        self.wer = wer
        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/hubert-large-ls960-ft"
        )
        self.model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")
        self.model = self.model.to("cuda")

    def calc(self, transcript_text, target_text):
        transcript_text = transcript_text.lower()
        transcript_text = re.sub(r"[^\w\s]", "", transcript_text)
        transcript_text = re.sub(r"\s+", " ", transcript_text)
        transcript_text = transcript_text.strip()

        target_text = target_text.lower()
        target_text = re.sub(r"[^\w\s]", "", target_text)
        target_text = re.sub(r"\s+", " ", target_text)
        target_text = target_text.strip()

        predictions = [transcript_text]
        references = [target_text]
        wer_score = self.wer.compute(predictions=predictions, references=references)
        return wer_score, transcript_text, target_text

    def __call__(self, audio, gt_text):
        # need 16khz audio, 1-dimensional
        assert len(audio.shape) == 1
        audio = np.array(audio.cpu())
        input_values = self.processor(audio, return_tensors="pt").input_values.to(
            "cuda"
        )
        logits = self.model(input_values=input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcript_text = self.processor.decode(predicted_ids[0])
        # remove special characters
        transcript_text = re.sub(r"[^\w\s]", "", transcript_text)

        wer_score, transcript_text, target_text = self.calc(transcript_text, gt_text)
        return wer_score, transcript_text, target_text


class SIM:
    def __init__(self):
        from evaluation_test.eval import (
            WAVLM_LARGE_FINTUNED_PATH,
            load,
            init_model,
            pipeline,
            Tasks,
        )

        print("Loading WavLM-large-finetuned")
        self.speaker_encoder = (
            init_model(checkpoint=WAVLM_LARGE_FINTUNED_PATH).to("cuda").eval()
        )

    def __call__(self, audio1, audio2):
        # need 16khz audio, 1-dimensional, torch tensor
        audio1 = audio1.unsqueeze(0).to("cuda")
        audio2 = audio2.unsqueeze(0).to("cuda")
        with torch.no_grad():
            embedding1 = self.speaker_encoder(audio1)
            embedding2 = self.speaker_encoder(audio2)
            sim = torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=1)
        return sim.item()


class FID:
    pass


class LibriSpeechDevDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir=None, use_vocos=False):
        self.data_dir = "/mnt/petrelfs/hehaorui/jiaqi/LibriSpeech/test-clean/*/*"
        self.wav_list = glob.glob(self.data_dir + "/*.flac") + glob.glob(
            self.data_dir + "/*.wav"
        )
        random.shuffle(self.wav_list)

        self.transcript_file = glob.glob(self.data_dir + "/*.txt")
        self.transcripts = {}
        for f_transcript in self.transcript_file:
            with open(f_transcript, "r") as f:
                for line in f:
                    line = line.strip().split()
                    self.transcripts[line[0]] = " ".join(line[1:])

    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, idx):
        wav_file = self.wav_list[idx]
        transcript = self.transcripts[os.path.basename(wav_file)[:-5]]
        orig_transcript = transcript
        transcript = g2p(transcript, "en")[1]
        transcript = torch.tensor(transcript, dtype=torch.long)

        speech, _ = librosa.load(wav_file, sr=SAMPLE_RATE)
        speech = torch.tensor(speech, dtype=torch.float32)

        return {
            "speech": speech,
            "phone_ids": transcript,
            "transcript": orig_transcript,
            "target_transcript": orig_transcript,
            "output_path": os.path.basename(wav_file)[:-5] + ".wav",
        }


import json


class LibriSpeechTestDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir=None, use_vocos=False):
        self.data_dir = "/mnt/petrelfs/hehaorui/jiaqi/vc-dev/Wave16k16bNormalized"
        self.wav_list = []
        self.transcripts = {}
        self.target_transcripts = {}

        # load json file
        with open(
            "/mnt/petrelfs/hehaorui/jiaqi/vc-dev/librispeech_ref_dur_3_test_full_with_punc_wdata.json",
            "r",
        ) as f:
            json_data = f.read()
        data = json.loads(json_data)

        test_data = data["test_cases"]

        self.output_path = []
        for wav_info in test_data:
            wav_path = os.path.join(self.data_dir, wav_info["wav_path"].split("/")[-1])
            self.wav_list.append(wav_path)
            # print(wav_info["wav_path"])
            wav_path = wav_info["wav_path"].split("/")[-1][:-4]
            self.transcripts[wav_path] = (
                wav_info["text"] + " " + wav_info["target_text"]
            )
            self.target_transcripts[wav_path] = wav_info["target_text"]
            # print(self.transcripts[wav_path])
            output_file_name = wav_info["uid"] + ".wav"
            self.output_path.append(output_file_name)

    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, idx):
        wav_file = self.wav_list[idx]
        transcript = self.transcripts[os.path.basename(wav_file)[:-4]]
        target_transcript = self.target_transcripts[os.path.basename(wav_file)[:-4]]
        # remove punctuation
        transcript = "".join(
            e for e in transcript if e.isalnum() or e.isspace()
        ).lower()
        orig_transcript = transcript
        transcript = g2p(transcript, "en")[1]
        # transcript = [LANG2CODE['en']] + transcript
        transcript = torch.tensor(transcript, dtype=torch.long)

        speech, _ = librosa.load(wav_file, sr=SAMPLE_RATE)
        speech = torch.tensor(speech, dtype=torch.float32)

        return {
            "speech": speech,  # prompt speech. do not include gt
            "phone_ids": transcript,
            "orig_transcript": orig_transcript,
            "target_transcript": target_transcript,
            "output_path": self.output_path[idx],
        }


def test():
    dataset = LibriSpeechDevDataset()
    # dataset = LibriSpeechTestDataset()
    from .valle_inference import ValleInference

    inference = ValleInference(
        use_vocos=False,
        use_speechtokenizer=True,
        ar_path="/mnt/petrelfs/hehaorui/jiaqi/vc-dev/ckpt/valle_v2/ar_mls_speechtokenizer/checkpoint/epoch-0004_step-0190000_loss-0.813551/pytorch_model.bin",
        nar_path="/mnt/petrelfs/hehaorui/jiaqi/AmphionVALLEv2/ckpt/valle_v2/nar_mls_speechtokenizer/checkpoint/epoch-0001_step-0164000_loss-1.848536/pytorch_model.bin",
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    if test_wer:
        wer = WER()
    if test_sim:
        sim = SIM()

    import tqdm

    wer_scores = []
    similarity_scores = []
    fid_scores = []
    total_cnt = 0

    shutil.rmtree("infer", ignore_errors=True)
    shutil.rmtree("wer_abnormals_output", ignore_errors=True)
    os.mkdir("infer")
    os.mkdir("wer_abnormals_output")
    for num_beams in [1]:
        for top_k in [30]:
            for top_p in [0.9]:
                for repeat_penalty in [1.0]:
                    for temperature in [0.95]:

                        for batch in tqdm.tqdm(dataloader):
                            if (
                                batch["speech"].shape[-1] < 10 * SAMPLE_RATE
                                or batch["speech"].shape[-1] > 20 * SAMPLE_RATE
                            ):
                                continue
                            # breakpoint()
                            print(batch["target_transcript"][0].lower())
                            chunks = [
                                dict(
                                    top_p=top_p,
                                    top_k=top_k,
                                    temperature=temperature,
                                    num_beams=num_beams,
                                    repeat_penalty=repeat_penalty,
                                    max_length=2000,
                                )
                            ]

                            if isinstance(dataset, LibriSpeechDevDataset):
                                output_wav = inference(
                                    batch, chunks, return_prompt=True
                                )
                            else:
                                output_wav = inference(
                                    batch, chunks, return_prompt=False
                                )

                            # output_wav = batch['speech'].unsqueeze(0)

                            torchaudio.save(
                                f"infer/{batch['output_path'][0]}",
                                output_wav[0].cpu(),
                                SAMPLE_RATE,
                            )
                            print(f"saved to " + f"infer/{batch['output_path'][0]}")

                            # breakpoint()
                            # torchaudio.save('gt.wav', batch['speech'][0].unsqueeze(0).cpu(), SAMPLE_RATE)

                            # resample to 16k
                            output_wav_resampled = torchaudio.functional.resample(
                                output_wav, orig_freq=SAMPLE_RATE, new_freq=16000
                            )
                            if test_wer:
                                # get wer score
                                wer_score, transcribed, gt_text = wer(
                                    output_wav_resampled.squeeze(0).squeeze(0),
                                    batch["target_transcript"][0],
                                )
                                print(f"WER: {wer_score}")
                                wer_scores.append(wer_score)
                                print(f"average wer: {sum(wer_scores)/len(wer_scores)}")

                                # if wer_score > 0.1:
                                #     # save
                                #     torchaudio.save(f'wer_abnormals_output/{batch["output_path"][0]}', output_wav[0].cpu(), SAMPLE_RATE)
                                #     # torchaudio.save(f'wer_abnormals_gt/{batch["output_path"][0]}', output_wav[0].cpu(), SAMPLE_RATE)
                                #     with open(f'wer_abnormals_output/{batch["output_path"][0][:-4]}.txt', 'w') as f:
                                #         f.write('target: ')
                                #         f.write(gt_text)
                                #         f.write('\n')
                                #         f.write('transcribed: ')
                                #         f.write(transcribed)
                                #         f.write('\n')
                                #         f.write(f'wer: {wer_score}')
                                #         print(f'target: {batch["target_transcript"][0]}, transcribed: {transcribed.lower()}')
                                #         print(f'wer_abnormals_output/{batch["output_path"][0][:-4]}.txt')
                            if test_sim:
                                # get similarity score
                                batch_speech_resampled = torchaudio.functional.resample(
                                    batch["speech"],
                                    orig_freq=SAMPLE_RATE,
                                    new_freq=16000,
                                )
                                sim_score = sim(
                                    output_wav_resampled.squeeze(0).squeeze(0),
                                    batch_speech_resampled.squeeze(0),
                                )
                                similarity_scores.append(sim_score)
                                print(f"SIM: {sim_score}")
                                print(
                                    f"average sim: {sum(similarity_scores)/len(similarity_scores)}"
                                )


if __name__ == "__main__":
    test()
