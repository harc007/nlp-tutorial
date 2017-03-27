######################################################
#Code for showing what Dan Jurafsky and Chris Manning
#teach in their Stanford NLP Code
#author: harc007
#######################################################

from collections import Counter
import nltk

'''
Input string = "Whatay playa's\nWhatay wonderful playa\n"
Output: (True, Counter({'playa': 2, 'Whatay': 2, 'wonderful': 1}))
'''
def get_word_count(string):
    try:
        words = nltk.word_tokenize(string)
        stemmer = nltk.stem.snowball.SnowballStemmer("english")
        stemmed_words = [stemmer.stem(i) for i in words]
        return (True, Counter(stemmed_words))
    except Exception as e:
        return (False, e)

'''
get list of sentences from string. sent_tokenizer uses classifier
'''
def get_sentences(string):
    try:
        return (True, nltk.sent_tokenizer(string))
    except Exception as e:
        return (False, e)

def get_minimum_distance(str1, str2):
    try:
        return (True, nltk.metrics.distance.edit_distance(str1, str2))
    except Exception as e:
        return (False, e)
