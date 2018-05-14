import os
import nltk
from nltk import bigrams
import numpy as np
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

path = os.path.join('corpus')
files = os.listdir(path)

s = ''

for f in files:
    with open(os.path.join(path, f), 'r') as f1:
        s1 = f1.read()
    s = s + s1[:1000]

words = nltk.word_tokenize(s)

text = nltk.Text(words)

vocab = list(set(text))

vocab_to_index = { word:i for i, word in enumerate(vocab) }

bi_grams = list(bigrams(text))

bigram_freq = nltk.FreqDist(bi_grams).most_common(len(bi_grams))

co_occurrence_matrix = np.zeros((len(vocab), len(vocab)))

for bigram in bigram_freq:
    current = bigram[0][1]
    previous = bigram[0][0]
    count = bigram[1]
    pos_current = vocab_to_index[current]
    pos_previous = vocab_to_index[previous]
    co_occurrence_matrix[pos_current][pos_previous] = count

co_occurrence_matrix = np.matrix(co_occurrence_matrix)

svd = TruncatedSVD(n_components=2)
svd.fit(co_occurrence_matrix)
result = svd.transform(co_occurrence_matrix)

print(len(vocab))
print(co_occurrence_matrix.shape)
print(result.shape)

plt.scatter(result[:, 0], result[:, 1])
plt.ylim([0, 0.1])
plt.xlim([0, 1])
for label, x, y in zip(vocab, result[:, 0], result[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
plt.show()
