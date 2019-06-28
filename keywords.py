from pythainlp.tokenize import word_tokenize
import pandas as pd

import re
import string
import pandas as pd
import nltk
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import stopwords
from pythainlp.corpus import wordnet
from nltk.stem.porter import PorterStemmer
from nltk.corpus import words
from stop_words import get_stop_words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("extract_salim.csv",error_bad_lines=False, warn_bad_lines=False)

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
    
    #Suggessting to remove links like httplinemetipuPmN0rkrnl, pictwittercomE6PgjFH2on
    
    return msg

def split_word(text):
            
    tokens = word_tokenize(text,engine='newmm')
    
    # Remove Thai and English stop words
    tokens = [i for i in tokens if not i in th_stop and not i in en_stop]

    # Find Thai and Emglish stem words
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

print("Cleaning text...")

nltk.download('words')
th_stop = tuple(thai_stopwords.words('thai'))
en_stop = tuple(get_stop_words('en'))
p_stemmer = PorterStemmer()

cleaned_tweets = [clean_msg(tweets) for tweets in df['text']]

split_words = []

print("Splitting words...")

split_words = [split_word(txt) for txt in cleaned_tweets]

split_words_j = [','.join(tkn) for tkn in split_words]
cvec = CountVectorizer(analyzer=lambda x:x.split(','))
c_feat = cvec.fit_transform(split_words_j)

print(cvec.vocabulary_)

# Count of the first 20 ids
print(c_feat[:,:20].todense())
