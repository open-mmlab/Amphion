import sys
feats_scp = sys.argv[1]
utt2prompt = sys.argv[2]

with open(feats_scp) as f1, open(utt2prompt) as f2:
    # Read the first file (feats.scp) and create a dictionary
    a = dict()
    for line in f1.readlines():
        columns = line.strip().split()  # Split the line into columns
        key = columns[0]  # Extract the key (first column)
        value = columns[1]  # Extract the value (second column)
        a[key] = value  # Assign the key-value pair to the dictionary

    # Process the second file (utt2prompt) and print the modified lines
    for line in f2.readlines():
        columns = line.strip().split()
        print(f"{columns[0]} {a.get(columns[1], '')}")
