import json
import csv
import yaml
import random
import xml.etree.ElementTree as ET

def parse_json(file):
    return [line.strip() for line in file]

def parse_csv(file):
    reader = csv.DictReader(file)
    return [row for row in reader]

def parse_text(file):
    return [line.strip() for line in file]

def parse_xml(file):
    objects = []
    for line in file:
        line = line.strip()
        if line:
            try:
                xml_obj = ET.fromstring(line)
                objects.append(ET.tostring(xml_obj, encoding='unicode'))
            except ET.ParseError:
                pass  
    return objects

def parse_yaml(file):
    try:
        yaml_data = yaml.safe_load(file)  
        if isinstance(yaml_data, list):
            return yaml_data
        elif isinstance(yaml_data, dict):
            data = list(yaml_data.values())
            return [json.dumps(obj) for obj in data]
        else:
            raise ValueError("Invalid YAML format")
    except yaml.YAMLError as exc:
        raise ValueError(f"Error parsing YAML file: {exc}")


def parse_custom_format(file):
    lines = file.readlines()
    header = lines[0].strip()
    return [header + '\n' + line.strip() for line in lines[1:]]

def read_file(filename):
    parsers = {
        '.json': parse_json,
        '.csv': parse_custom_format,
        '.txt': parse_text,
        '.log': parse_text,
        '.xml': parse_xml,
        '.yaml': parse_yaml,
        '.abc': parse_custom_format 
    }

    file_extension = filename[filename.rfind('.'):].lower()
    parser = parsers.get(file_extension)
    
    if not parser:
        raise ValueError(f"Unsupported file type: {file_extension}")

    with open(filename, 'r', encoding='utf-8') as file:
        return parser(file)

def load_and_sample_data(filename):
    objects = read_file(filename)
    if objects:
        return random.choice(objects)
    else:
        raise ValueError(f"No valid data found in the file: {filename}")

if __name__ == '__main__':
    filename = 'parser_data\\data\\self_made_data.abc'

    random_object = load_and_sample_data(filename)

    print("random object:\n", random_object)
