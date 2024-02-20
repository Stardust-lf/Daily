import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import numpy as np


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, vocab_size)

    def forward(self, inputs):
        embeds = sum(self.embeddings(inputs)).view((1, -1))
        out = self.linear1(embeds)
        log_probs = torch.log_softmax(out, dim=1)
        return log_probs


class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = self.linear1(embeds)
        log_probs = torch.log_softmax(out, dim=1)
        return log_probs


def train_model(model, data, word_to_ix, epochs=10):
    losses = []
    loss_fn = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        total_loss = 0
        for context, target in data:
            context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
            model.zero_grad()
            log_probs = model(context_idxs)
            loss = loss_fn(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss)
    return losses


sentence = "I am taking CS6493 this semester and studying NLP is really fascinating".split()
trigrams = [([sentence[i], sentence[i + 1]], sentence[i + 2]) for i in range(len(sentence) - 2)]
vocab = set(sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}

EMBEDDING_DIM = 100
CONTEXT_SIZE = 2

cbow_model = CBOW(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
skipgram_model = SkipGram(len(vocab), EMBEDDING_DIM)

cbow_loss = train_model(cbow_model, trigrams, word_to_ix)
skipgram_loss = train_model(skipgram_model, trigrams, word_to_ix)

print(f'CBOW Loss: {cbow_loss[-1]}')
print(f'SkipGram Loss: {skipgram_loss[-1]}')
