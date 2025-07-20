import json
import os


def parse_config(json_file: str, show: bool = False):
    with open(json_file, "r") as f:
        data = json.load(f)
    if show:
        print(json.dumps(data, indent=4))
    return data.keys(), data


def save_config(config, output_path):
    os.makedirs(output_path, exist_ok=True)
    config_file_path = os.path.join(output_path, "config.json")
    with open(config_file_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to {config_file_path}")
