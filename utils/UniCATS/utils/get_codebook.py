import torch
import numpy as np

# Load the pre-trained model checkpoint
cp_path = 'vq-wav2vec_kmeans.pt'
cp = torch.load(cp_path)
model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp])
model = model[0]

# Access the codebook
codebook = model.vector_quantizer.embedding.weight.data.cpu().numpy()

# Save the codebook to a numpy file
np.save('codebook.npy', codebook)
