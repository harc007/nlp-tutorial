######################################################
#Code for showing what Dan Jurafsky and Chris Manning
#teach in their Stanford NLP Code
#author: harc007
#######################################################

from collections import Counter
import nltk

'''
fit n grams model and predict
'''
def generate_model(cfdist, start_string, num=1):
    try:
        next_words = []
        for i in range(num):
            if word in cfdist:
                word = cfdist[word].max()
            else:
                word = ''
            next_words.append(word)
        return (True, next_words)
    except Exception as e:
        return (False, e)

def fit_ngrams_model(text, n, start_string):
    try:
        print "Building ngrams model"
        ngrams_model = nltk.ngrams(text.split(), n)
        cfd = nltk.ConditionalFreqDist(bigrams)
        next_words = generate_model(cfd, start_string)
        if next_words[0]:
            return (True, next_words[1])
        else:
            return next_words
    except Exception as e:
        return (False, e)
