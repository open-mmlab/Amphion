# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#!/bin/bash

# Navigate to the 'pretrained' directory
cd pretrained || { echo "Failed to change directory to 'pretrained'"; exit 1; }

# Create and navigate to the 'mfa' directory
mkdir -p mfa && cd mfa || { echo "Failed to create or change directory to 'mfa'"; exit 1; }

# Define the MFA file URL and the file name
mfa_url="https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/releases/download/v1.1.0-beta.2/montreal-forced-aligner_linux.tar.gz"
mfa_file="montreal-forced-aligner_linux.tar.gz"

# Download MFA if it doesn't exist
if [ ! -f "$mfa_file" ]; then
    wget "$mfa_url" || { echo "Failed to download MFA"; exit 1; }
fi

# Extract MFA
tar -zxvf "$mfa_file" || { echo "Failed to extract MFA"; exit 1; }

# Optionally, remove the tar.gz file after extraction
rm "$mfa_file"

echo "MFA setup completed successfully."
