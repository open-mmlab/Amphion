import os

def modify_wav_scp(path):
    """
    Modify the wav.scp file to use relative paths.
    """
    input_file = os.path.join(path, "wav.scp")
    output_file = os.path.join(path, "wav.scp")

    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            parts = line.split()
            if len(parts) == 2:
                key, full_path = parts
                # Replace the absolute path with relative path
                relative_path = full_path.split('/your/path/')[-1]
                outfile.write(f"{key} {relative_path}\n")

def modify_feats_scp(path):
    """
    Modify the feats.scp file to use relative paths.
    """
    input_file = os.path.join(path, "feats.scp")
    output_file = os.path.join(path, "feats.scp")

    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            parts = line.split()
            if len(parts) == 2:
                key, full_path = parts
                # Replace the absolute path with relative path
                relative_path = full_path.split('/path/to/project/')[-1]
                outfile.write(f"{key} {relative_path}\n")

# Paths for WAV and feats files
wav_paths = [
    "../../../data/dev_all",
    "../../../data/eval_all",
    "../../../data/train_all"
]
feats_paths = [
    "../../../feats/normed_ppe/dev_all",
    "../../../feats/normed_ppe/eval_all",
    "../../../feats/normed_ppe/train_all"
]

# Process each path
for path in wav_paths:
    modify_wav_scp(path)

for path in feats_paths:
    modify_feats_scp(path)