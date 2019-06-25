import json
import os
import glob

from argparse import ArgumentParser
parser = ArgumentParser(description = 'JSON Compressor')
parser.add_argument('--file_dir', required = True, help = 'JSON Directory')
parser.add_argument('--path', required = True, help = 'Path to save the compressed file')

def compress(file_dir, path):
    result = []
    infiles = os.listdir(file_dir)
    infiles = [os.path.join(file_dir, file_name) for file_name in infiles]
    for f in infiles:
        with open(f, "r") as infile:
            try:
                result.append(json.load(infile))
            except ValueError:
                print(f)

    with open(path, "w") as outfile:
        json.dump(result, outfile)

    return path

def extract(json_file, path):
    with open(json_file, 'r') as f:
        list_dict = json.load(f)

    with open(path, "w") as outfile:
        for element in list_dict:
            outfile.write(element["text"])

    return path

if __name__ == "__main__":
    args = parser.parse_args()
    file_dir = args.file_dir
    path = args.path

    json_file = compress(file_dir, os.path.join(os.getcwd(), 'compressed.json'))
    extract(json_file, path)
    os.remove(json_file)
        