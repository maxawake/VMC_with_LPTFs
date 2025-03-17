import json

def parse_config(json_file: str):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data.keys(), data
