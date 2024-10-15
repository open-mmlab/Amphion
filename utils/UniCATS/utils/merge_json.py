import json

base_json_path = 'config/UniCATS_txt2vec.json'  
example_json_path = 'egs/tts/UniCATS/CTXtxt2vec/exp_config.json'  

merged_output_path = base_json_path

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def save_merged_json(base_config, override_config, output_path):
    # Merge override_config into base_config
    merge_dicts(base_config, override_config)
    
    # Write the merged configuration back to base.json
    with open(output_path, 'w') as file:
        json.dump(base_config, file, indent=4)

def merge_dicts(base_dict, override_dict):
    for key, value in override_dict.items():
        if key not in base_dict:
            base_dict[key] = value
        elif isinstance(value, dict) and isinstance(base_dict[key], dict):
            merge_dicts(base_dict[key], value)
        else:
            base_dict[key] = value

# Load base.json and example.json
base_config = load_json(base_json_path)
example_config = load_json(example_json_path)

# Output path set to base.json to update it with merged configuration
output_path = merged_output_path

# Save the merged configuration to base.json
save_merged_json(base_config, example_config, output_path)
