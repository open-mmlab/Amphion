import torch
import fairseq
import librosa
import numpy as np
import os
import struct

# Define paths
dataset_dir = os.path.expanduser("Amphion/LibriTTS") # Change this to your dataset path
output_dir = "feats"
# Load vq-wav2vec model
cp_path = 'vq-wav2vec_kmeans.pt'

def process_audio(file_path, model):
    # Load the audio file
    wav, sr = librosa.load(file_path, sr=16000)
    
    # Convert to tensor
    wav_input = torch.tensor(wav).unsqueeze(0)
    
    # Extract features
    z = model.feature_extractor(wav_input)
    _, idxs = model.vector_quantizer.forward_idx(z)
    
    # Flatten to a 1D array
    idxs = idxs.view(-1, 2).numpy()
    
    return idxs

def write_kaldi_archive(output_dir, scp_file, ark_file, audio_files, model):
    with open(scp_file, "w") as scp, open(ark_file, "wb") as ark:
        for i, audio_file in enumerate(audio_files):
            # Extract features
            features = process_audio(audio_file, model)
            
            # Convert to a byte string for storage
            byte_str = features.tobytes()

            # Create an ID based on the file index
            audio_id = os.path.splitext(os.path.basename(audio_file))[0]

            # Write to ARK file
            ark.write(struct.pack('>II', i, len(byte_str)))
            ark.write(byte_str)

            # Write to SCP file
            scp.write(f"{audio_id} {ark_file}:{i*len(byte_str)}:{len(byte_str)}\n")

# Main code to process the dataset
if __name__ == "__main__":
    
    # List audio files to process
    audio_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(dataset_dir) for f in fn if f.endswith(".wav")]

    print(f"Found {len(audio_files)} audio files.")

    # cp = torch.load(cp_path)
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
    model = model[0]
    model.eval()

    # Output files
    scp_file = os.path.join(output_dir, "feats.scp")
    ark_file = os.path.join(output_dir, "feats.ark")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process and save
    write_kaldi_archive(output_dir, scp_file, ark_file, audio_files, model)
