######################################################
#Code for showing what Dan Jurafsky and Chris Manning
#teach in their Stanford NLP Code
#author: harc007
#######################################################

#######################################################
#Rule:re.match(pattern, string, flags=0)'
######################################################
import re

def get_search(string, substring, case=True, findall=True, multiple=True, words=True, groups=True):
    try:
        if multiple:
            substring = substring + "+"
        if words:
            substring = "[\w.-]+" + substring + "[\w.-]+"
        if groups:
            substring = '([\w.-]+)' + substring + '([\w.-]+)'
        if case and not findall:
            return (True, re.search(substring, string, flags=re.I))
        elif case and findall:
            return (True, re.finditer(substring, string, flags=re.I))
        elif not case and findall:
            return (True, re.findall(substring, string))
        elif not case and not findall:
            return (True, re.search(substring, string))
        else:
            return (False, "Some weird situation has occured")
    except Exception as e:
        return (False, e)

