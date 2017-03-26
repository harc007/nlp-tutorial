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
        lower_string = string.lower().replace("'s", "")
        tokens = nltk.word_tokenize(lower_string)
        text = nltk.Text(tokens)
        return (True, Counter(text))
    except Exception as e:
        return (False, e)

