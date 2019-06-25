import json

from argparse import ArgumentParser
parser = ArgumentParser(description = 'JSON Compressor')
parser.add_argument('--file_dir', required = True, help = 'JSON Directory')
parser.add_argument('--path', required = True, help = 'Path to save the compressed file')

def compress(file_dir, path):
    pass