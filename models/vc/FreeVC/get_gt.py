from transformers import Wav2Vec2Processor, HubertForCTC
import argparse
import torch
import librosa
from tqdm import tqdm
from glob import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--txtpath", type=str, default="gt.txt", help="path to tgt txt file"
    )
    parser.add_argument("--wavdir", type=str, default=r"data\VCTK\test_data")
    args = parser.parse_args()

    # load model and processor
    model_text = HubertForCTC.from_pretrained(
        "facebook/hubert-large-ls960-ft"
    ).cuda()  # type:ignore
    processor_text = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")

    # get transcriptions
    wavs = glob(f"{args.wavdir}/*.wav")
    wavs.sort()
    with open(f"{args.txtpath}", "w") as f:
        for path in tqdm(wavs):
            wav = [librosa.load(path, sr=16000)[0]]
            input_values = processor_text(
                wav, return_tensors="pt"
            ).input_values.cuda()  # text # type:ignore
            logits = model_text(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            text = processor_text.batch_decode(predicted_ids)[0]  # type:ignore
            f.write(f"{path}|{text}\n")
