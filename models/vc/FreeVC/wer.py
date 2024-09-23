from transformers import Wav2Vec2Processor, HubertForCTC
import os
import argparse
import torch
import librosa
from tqdm import tqdm
from glob import glob
from jiwer import wer, cer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wavdir", type=str, default=r"data\VCTK\test_output")
    parser.add_argument(
        "--outdir", type=str, default="result", help="path to output dir"
    )
    parser.add_argument("--use_cuda", default=True, action="store_true")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # load model and processor
    model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")
    if args.use_cuda:
        model = model.cuda()  # type:ignore
    processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")

    # gt
    gt_dict = {}
    with open("gt.txt", "r") as f:
        for line in f.readlines():
            path, text = line.strip().split("|")
            title = os.path.basename(path)[:-4]
            gt_dict[title] = text

    # get transcriptions
    wavs = glob(f"{args.wavdir}/*.wav")
    wavs.sort()
    trans_dict = {}

    with open(f"{args.outdir}/text.txt", "w") as f:
        for path in tqdm(wavs):
            wav = [librosa.load(path, sr=16000)[0]]
            input_values = processor(
                wav, return_tensors="pt"
            ).input_values  # type:ignore
            if args.use_cuda:
                input_values = input_values.cuda()
            logits = model(input_values).logits  # type:ignore
            predicted_ids = torch.argmax(logits, dim=-1)
            text = processor.batch_decode(predicted_ids)[0]  # type:ignore
            f.write(f"{path}|{text}\n")
            title = os.path.basename(path)[:-4]
            trans_dict[title] = text

    # calc
    gts, trans = [], []
    for key in trans_dict.keys():
        text = trans_dict[key]
        trans.append(text)
        # gttext = gt_dict[key.split("-")[0]]
        gttext = gt_dict[key[:8]]
        gts.append(gttext)

    wer = wer(gts, trans)
    cer = cer(gts, trans)
    with open(f"{args.outdir}/wer.txt", "w") as f:
        f.write(f"wer: {wer}\n")
        f.write(f"cer: {cer}\n")
    print("WER:", wer)
    print("CER:", cer)
