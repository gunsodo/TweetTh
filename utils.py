import json
import os
import glob
import pandas as pd 

from argparse import ArgumentParser
parser = ArgumentParser(description = 'JSON Compressor')
parser.add_argument('--file_dir', required = True, help = 'JSON Directory')
parser.add_argument('--path', required = True, help = 'Path to save the compressed file')
parser.add_argument('--format', help = 'Either .txt (default) or .csv (more details)')

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

def get_hashtag(string, sign = True):
    """
    Get hashtags from a string
    """
    hashtag_set = set(part[1:] for part in string.split() if part.startswith('#'))
    hashtag_list = list(hashtag_set)
    if sign:
        hashtag_list = ['#'+item for item in hashtag_list]
    return hashtag_list

def collect_hashtag(text_file, sign = True, flatten = True):
    with open (text_file, "r") as file:
        data = file.readlines()
    data = [item.replace('\n', '') for item in data]
    hashtag_list = []
    for tweet in data:
        hashtag = get_hashtag(tweet, sign = sign)
        if flatten:
            hashtag_list.extend(hashtag)
        else:
            hashtag_list.append(hashtag)
    return hashtag_list

def count_hashtag(hashtag_list):
    element_set = set(hashtag_list)
    element_list = list(element_set)
    count = []
    for element in element_list:
        occ = hashtag_list.count(element)
        count.append(occ)
    df_count = pd.DataFrame({'Hashtags':element_list, 'Occurences':count})
    df_count = df_count.sort_values(by = ['Occurences'], ascending = False)
    df_count = df_count.reset_index(drop = True)
    print(df_count.head(10))
    return df_count

if __name__ == "__main__":
    args = parser.parse_args()
    file_dir = args.file_dir
    path = args.path
    file_format = args.format

    json_file = compress(file_dir, os.path.join(os.getcwd(), 'compressed.json'))
    extract(json_file, path)
    os.remove(json_file)
        