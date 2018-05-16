import os
import nltk
from nltk import bigrams
import numpy as np
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def read_corpus():
    try:
        path = os.path.join('corpus')
        files = os.listdir(path)
        s = ''
        for f in files:
            with open(os.path.join(path, f), 'r') as f1:
                s1 = f1.read()
            s1 = s1.replace('\n', ' ').replace('"', '')
            s = s + s1
        return s
    except Exception as e:
        print(traceback.format_exc())
        raise e

def build_svd_word2vec(s):
    try:
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
        plt.scatter(result[:, 0], result[:, 1])
        plt.ylim([0, 0.1])
        plt.xlim([0, 1])
        for label, x, y in zip(vocab, result[:, 0], result[:, 1]):
            plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
        plt.show()
        return svd
    except Exception as e:
        print(traceback.format_exc())
        raise e


def build_ngrams_word2vec(s):
    try:
        context_size = 2
        embedding_dim = 10
        test_sentence = s.split()
        trigrams = [([test_sentence[i], test_sentence[i+1]], test_sentence[i+2]) for i in range(len(test_sentence)-2)]
        print(1, trigrams[:3])
        vocab = set(test_sentence)
        word_to_ix = {word:i for i, word in enumerate(vocab)}

        class NGramLanguageModeler(nn.Module):
            def __init__(self, vocab_size, embedding_dim, context_size):
                super(NGramLanguageModeler, self).__init__()
                self.embeddings = nn.Embedding(vocab_size, embedding_dim)
                self.linear1 = nn.Linear(context_size*embedding_dim, 128)
                self.linear2 = nn.Linear(128, vocab_size)

            def forward(self, inputs):
                embeds = self.embeddings(inputs).view((1, -1))
                out = F.relu(self.linear1(embeds))
                out = self.linear2(out)
                log_probs = F.log_softmax(out, dim=1)
                return log_probs

        losses = []
        loss_function = nn.NLLLoss()
        model = NGramLanguageModeler(len(vocab), embedding_dim, context_size)
        optimizer = optim.SGD(model.parameters(), lr=0.001)

        for epoch in range(10):
            total_loss = torch.Tensor([0])
            for context, target in trigrams:
                context_ids = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
                model.zero_grad()
                log_probs = model(context_ids)
                loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            losses.append(total_loss)
        print(losses)
        return losses
    except Exception as e:
        print(traceback.format_exc())
        raise e


if __name__ == '__main__':
    corpus = read_corpus()
    losses = build_ngrams_word2vec(corpus)
