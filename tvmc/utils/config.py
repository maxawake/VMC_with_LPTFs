import json
import os


def parse_config(json_file: str, show: bool = False):
    """Parse a JSON configuration file.

    Args:
        json_file (str): Path to the JSON file.
        show (bool, optional): Whether to print the configuration. Defaults to False.

    Returns:
        tuple: A tuple containing the keys and the parsed configuration.
    """
    with open(json_file, "r") as f:
        data = json.load(f)
    if show:
        print(json.dumps(data, indent=4))
    return data.keys(), data


def save_config(config, output_path):
    """Save the configuration to a JSON file.
    Args:
        config (dict): Configuration dictionary.
        output_path (str): Path to the output directory.
    """
    os.makedirs(output_path, exist_ok=True)
    config_file_path = os.path.join(output_path, "config.json")
    with open(config_file_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to {config_file_path}")
