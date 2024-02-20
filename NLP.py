import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD


class LanguageModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, hidden_size=128):
        super(LanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

def train_model(sentence, trigrams, vocab, model, optimizer, epochs=10):
    losses = []
    loss_fn = nn.NLLLoss()
    for epoch in range(epochs):
        total_loss = 0
        for context, target in trigrams:
            context_idxs = torch.tensor([vocab[w] for w in context], dtype=torch.long)
            model.zero_grad()
            log_probs = model(context_idxs)
            loss = loss_fn(log_probs, torch.tensor([vocab[target]], dtype=torch.long))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss)
    return losses

sentence = "I am taking CS6493 this semester and studying NLP is really fascinating".split()
trigrams = [([sentence[i], sentence[i + 1], sentence[i + 2]], sentence[i + 3]) for i in range(len(sentence) - 3)]
vocab = set(sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for i, word in enumerate(vocab)}

embedding_dims = [32, 64, 128]
losses = {}
for embedding_dim in embedding_dims:
    model = LanguageModeler(len(vocab), embedding_dim, 3)
    optimizer = SGD(model.parameters(), lr=0.001)
    loss = train_model(sentence, trigrams, word_to_ix, model, optimizer)
    losses[embedding_dim] = loss

for embedding_dim, loss in losses.items():
    print(f'Embedding Dimension: {embedding_dim}, Loss: {loss}')
