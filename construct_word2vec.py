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
from torch.autograd import Variable
from torch.optim import SGD

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
        torch.save(model.state_dict(), 'ngrams.pth')
        return model
    except Exception as e:
        print(traceback.format_exc())
        raise e

def build_cbow_word2vec(s):
    try:
        CONTEXT_SIZE = 4
        EMBEDDING_DIM = 300
        EPOCH = 10
        VERVOSE = 5

        corpus_text = s


        class CBOW(nn.Module):
            def __init__(self, vocab_size, embedding_size, context_size):
                super(CBOW, self).__init__()
                self.vocab_size = vocab_size
                self.embedding_size = embedding_size
                self.context_size = context_size
                self.embeddings = nn.Embedding(self.vocab_size, self.embedding_size)
                # return vector size will be context_size*2*embedding_size
                self.lin1 = nn.Linear(self.context_size * 2 * self.embedding_size, 512)
                self.lin2 = nn.Linear(512, self.vocab_size)
                                                                                                            
            def forward(self, inp):
                out = self.embeddings(inp).view(1, -1)
                out = out.view(1, -1)
                out = self.lin1(out)
                out = F.relu(out)
                out = self.lin2(out)
                out = F.log_softmax(out, dim=1)
                return out
                                                                                                                                                                            
            def get_word_vector(self, word_idx):
                word = Variable(torch.LongTensor([word_idx]))
                return self.embeddings(word).view(1, -1)


        def train_cbow(data, unique_vocab, word_to_idx):
            cbow = CBOW(len(unique_vocab), EMBEDDING_DIM, CONTEXT_SIZE)
            nll_loss = nn.NLLLoss()  # loss function
            optimizer = SGD(cbow.parameters(), lr=0.001)
            print(len(data))
            for epoch in range(EPOCH):
                total_loss = 0
                for context, target in data:            
                    inp_var = Variable(torch.LongTensor([word_to_idx[word] for word in context]))
                    target_var = Variable(torch.LongTensor([word_to_idx[target]]))
                    cbow.zero_grad()
                    log_prob = cbow(inp_var)
                    loss = nll_loss(log_prob, target_var)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.data
                if epoch % VERVOSE == 0:
                    loss_avg = float(total_loss / len(data))
                    print("{}/{} loss {:.2f}".format(epoch, EPOCH, loss_avg))
            return cbow


        def test_cbow(cbow, unique_vocab, word_to_idx):
            word_1 = unique_vocab[2]
            word_2 = unique_vocab[3]
            word_1_vec = cbow.get_word_vector(word_to_idx[word_1])
            word_2_vec = cbow.get_word_vector(word_to_idx[word_2])
            ws = (word_1_vec.reshape(word_1_vec.shape[1],).dot(word_2_vec.reshape(word_2_vec.shape[1],)) / (torch.norm(word_1_vec) * torch.norm(word_2_vec)))
            word_similarity = ws.data.numpy()
            print("Similarity between '{}' & '{}' : {:0.4f}".format(word_1, word_2, word_similarity))


        # content processed as context/target
        # consider 2*CONTEXT_SIZE as context window where middle word as target
        data = list()
        for i in range(CONTEXT_SIZE, len(corpus_text) - CONTEXT_SIZE):
            data_context = list()
            for j in range(CONTEXT_SIZE):
                data_context.append(corpus_text[i - CONTEXT_SIZE + j])
            for j in range(1, CONTEXT_SIZE + 1):
                data_context.append(corpus_text[i + j])
            data_target = corpus_text[i]
            data.append((data_context, data_target))
        print("Some data: ",data[:3])
        unique_vocab = list(set(corpus_text))

        # mapping to index
        word_to_idx = {w: i for i, w in enumerate(unique_vocab)}

        # train model- changed global variable if needed
        cbow = train_cbow(data, unique_vocab, word_to_idx)
        torch.save(cbow.state_dict(), 'cbow.pth')
        
        # get two words similarity
        test_cbow(cbow, unique_vocab, word_to_idx)
        return cbow
    except Exception as e:
        print(traceback.format_exc())
        raise e


if __name__ == '__main__':
    corpus = read_corpus()
    losses = build_ngrams_word2vec(corpus[:1000])
    #cbow_model = build_cbow_word2vec(corpus[:1000].split())
