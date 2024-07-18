import argparse
import torch
import numpy as np
import librosa
from safetensors.torch import load_model
import os
from utils.util import load_config
from models.vc.vc_trainer import VCTrainer
from models.vc.ns2_uniamphion import UniAmphionVC
from models.vc.hubert_kmeans import HubertWithKmeans
from models.vc.vc_utils import mel_spectrogram, extract_world_f0

def build_trainer(args, cfg):
    supported_trainer = {
        "VC": VCTrainer,
    }
    trainer_class = supported_trainer[cfg.model_type]
    trainer = trainer_class(args, cfg)
    return trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config.json",
        help="JSON file for configurations.",
        required=True,
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Checkpoint for resume training or fine-tuning.",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        help="Output path",
        required=True,
    )
    parser.add_argument(
        "--ref_path",
        type=str,
        help="Reference voice path",
    )
    parser.add_argument(
        "--source_path",
        type=str,
        help="Source voice path",
    )
    parser.add_argument(
        "--cuda_id",
        type=int,
        default=0,
        help="CUDA id for training."
    )
    
    parser.add_argument("--local_rank", default=-1, type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
 
    cuda_id = args.cuda_id
    args.local_rank = torch.device(f"cuda:{cuda_id}")
    print("Local rank:", args.local_rank)

    args.content_extractor = "mhubert"

    with torch.cuda.device(args.local_rank):
        torch.cuda.empty_cache()
    ckpt_path = args.checkpoint_path
    
    w2v = HubertWithKmeans()
    w2v = w2v.to(device=args.local_rank)
    w2v.eval()

    model = UniAmphionVC(cfg=cfg.model)
    print("Loading model")

    load_model(model, ckpt_path)
    print("Model loaded")
    model.cuda(args.local_rank)
    model.eval()
    
    wav_path = args.source_path
    ref_wav_path = args.ref_path
    
    wav, _ = librosa.load(wav_path, sr=16000)
    wav = np.pad(wav, (0, 1600 - len(wav) % 1600))
    audio = torch.from_numpy(wav).to(args.local_rank)
    audio = audio[None, :]
    
    ref_wav, _ = librosa.load(ref_wav_path, sr=16000)
    ref_wav = np.pad(ref_wav, (0, 200 - len(ref_wav) % 200))
    ref_audio = torch.from_numpy(ref_wav).to(args.local_rank)
    ref_audio = ref_audio[None, :]
    
    with torch.no_grad():
        ref_mel = mel_spectrogram(ref_audio)
        ref_mel = ref_mel.transpose(1, 2).to(device=args.local_rank)
        ref_mask = torch.ones(ref_mel.shape[0], ref_mel.shape[1]).to(args.local_rank).bool()

        _, content_feature = w2v(audio)
        content_feature = content_feature.to(device=args.local_rank)

        pitch_raw = extract_world_f0(audio)
        pitch = (pitch_raw - pitch_raw.mean(dim=1, keepdim=True)) / (
            pitch_raw.std(dim=1, keepdim=True) + 1e-6
        )

        x0 = model.inference(
            content_feature=content_feature,
            pitch=pitch,
            x_ref=ref_mel,
            x_ref_mask=ref_mask,
            inference_steps=200,
            sigma=1.2,
        )  # 150-300 0.95-1.5

        recon_path = f"{args.output_dir}/recon_mel.npy"
        np.save(recon_path, x0.transpose(1, 2).detach().cpu().numpy())
        print(f"Mel spectrogram saved to: {recon_path}")

if __name__ == "__main__":
    main()

