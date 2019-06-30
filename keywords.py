import re
import string
import pandas as pd
import nltk
import sys
import progressbar
import numpy as np
import collections

from pythainlp.tokenize import word_tokenize
from pythainlp.corpus.common import thai_stopwords
from pythainlp.corpus import wordnet

from nltk.stem.porter import PorterStemmer
from nltk.corpus import words
from nltk.collocations import BigramCollocationFinder 
from nltk.collocations import TrigramCollocationFinder 
from nltk.metrics import BigramAssocMeasures
from nltk.metrics import TrigramAssocMeasures

from stop_words import get_stop_words

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

if len(sys.argv) != 3:
    print("Usage: {} CSV_FILE N_WORDS".format(sys.argv[0]))
    sys.exit(1)

def clean_msg(msg):
    # Remove all text within "<>"
    msg = re.sub(r'<.*?>','', msg)
    
    # Remove "#"
    msg = re.sub(r'#','',msg)
    
    # Remove punctuation
    for c in string.punctuation:
        msg = re.sub(r'\{}'.format(c),'',msg)
    
    # Remove separator e.g. \n \t
    msg = ' '.join(msg.split())
    
    # Remove links like httplinemetipuPmN0rkrnl, pictwittercomE6PgjFH2on, twittercom
    msg = re.sub(r'http.*','', msg)
    msg = re.sub(r'pictwitter.*','', msg)
    msg = re.sub(r'twittercom.*','', msg)
    return msg

def split_word(text):
    th_stop = tuple(thai_stopwords())
    en_stop = tuple(get_stop_words('en'))
    p_stemmer = PorterStemmer()

    tokens = word_tokenize(text,engine='newmm')
    
    # Remove Thai and English stop words
    tokens = [i for i in tokens if not i in th_stop and not i in en_stop]

    # Find Thai and English stem words
    # English
    tokens = [p_stemmer.stem(i) for i in tokens]
    
    # Thai
    tokens_temp=[]
    for i in tokens:
        w_syn = wordnet.synsets(i)
        if (len(w_syn)>0) and (len(w_syn[0].lemma_names('tha'))>0):
            tokens_temp.append(w_syn[0].lemma_names('tha')[0])
        else:
            tokens_temp.append(i)
    
    tokens = tokens_temp
    
    # Remove numbers
    tokens = [i for i in tokens if not i.isnumeric()]
    
    # Remove space
    tokens = [i for i in tokens if not ' ' in i]

    return tokens

def get_th_stop():
    return tuple(thai_stopwords())

nltk.download('words')

df = pd.read_csv(sys.argv[1], error_bad_lines=False, warn_bad_lines=False)

print("Cleaning text...")
cleaned_tweets = [clean_msg(tweets) for tweets in df['text']]

print("Splitting words...")
bar = progressbar.ProgressBar(maxval = len(cleaned_tweets), widgets = ['Progress: ', progressbar.SimpleProgress(), ' ', progressbar.Bar('=', '[', ']'), ' ', progressbar.ETA(), ' ', progressbar.Percentage()])
bar.start()
split_words = []
for index, txt in enumerate(cleaned_tweets):
    split_words.append(split_word(txt))
    bar.update(index+1)
bar.finish()

print("Processing...")
split_words_j = [','.join(tkn) for tkn in split_words]

tvec = TfidfVectorizer(analyzer=lambda x:x.split(','),)
t_feat = tvec.fit_transform(split_words_j)

sorted_vocab = sorted(tvec.vocabulary_.items(), key = lambda kv: kv[1])
sorted_dict = collections.OrderedDict(sorted_vocab)
list_vocab = [k for k in sorted_dict]

t_array = np.transpose(t_feat.toarray())
t_df = pd.DataFrame(t_array, columns = range(len(cleaned_tweets)), index = list_vocab)

sum_df = t_df.sum(axis = 1, skipna = True)
sum_df = pd.DataFrame(sum_df, columns = ['Frequency'])
sum_df = sum_df.sort_values(by = 'Frequency', ascending = False)
print(sum_df.head(50))
print(sum_df.sum())

#-----------------------------#

cvec = CountVectorizer(analyzer=lambda x:x.split(','))
c_feat = cvec.fit_transform(split_words_j)
# vocabs = [w for w in cvec.vocabulary_.keys()]

flattened_split_words = [y for x in split_words for y in x]
biagram_collocation = BigramCollocationFinder.from_words(flattened_split_words) 
th_stop = get_th_stop()
filter_stops = lambda w: len(w) < 3 or w in th_stop
biagram_collocation.apply_word_filter(filter_stops)

biagram = biagram_collocation.score_ngrams(BigramAssocMeasures.likelihood_ratio)

prefix_keys = collections.defaultdict(list)
for key, scores in biagram:
   prefix_keys[key[0]].append((key[1], scores))

for key in prefix_keys:
   prefix_keys[key].sort(key = lambda x: -x[1])

n_words = int(sys.argv[2])

top_index = sum_df.index.values.tolist()
count = 0
i = 0
while count < n_words:
    if len(prefix_keys[top_index[i]][:10]) != 0:
        print(top_index[i], prefix_keys[top_index[i]][:10], "\n")
        count+=1
    else:
        pass
    i+=1