######################################################
#Code for showing what Dan Jurafsky and Chris Manning
#teach in their Stanford NLP Code
#author: harc007
#######################################################

from collections import Counter

'''
Input string = "Whatay playa\nWhatay wonderful playa\n"
Output: (True, Counter({'playa': 2, 'Whatay': 2, 'wonderful': 1}))
'''
def get_word_count(string):
    try:
        word_list = string.split()
        return (True, Counter(word_list))
    except Exception as e:
        return (False, e)

