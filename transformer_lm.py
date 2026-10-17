# models.py

import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import PositionalEncoding

class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size
 
    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0 / self.voc_size)
 
    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0 / self.voc_size) * len(next_chars)

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, num_positions=20, d_model=128, nhead=4,
                 num_layers=3, dim_ff=256, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, num_positions, batched=True)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: LongTensor [B, T]
        T = x.shape[1]
        mask = nn.Transformer.generate_square_subsequent_mask(T)   # 0 on/below diagonal, -inf above
        h = self.encoder(self.pos_enc(self.embed(x)), mask=mask)   # [B, T, d_model]
        return F.log_softmax(self.out(h), dim=-1)  
                    # [B, T, 27]

class NeuralLanguageModel(LanguageModel):
    def __init__(self, model, vocab_index, chunk_len=20):
        self.model = model
        self.vocab_index = vocab_index
        self.chunk_len = chunk_len
 
    def get_next_char_log_probs(self, context):
        self.model.eval()                                   # turn off dropout
        ctx = " " + context[-(self.chunk_len - 1):]         # SOS + up to 19 chars
        idxs = [self.vocab_index.index_of(c) for c in ctx]
        x = torch.LongTensor([idxs])                        # [1, len(ctx)]
        with torch.no_grad():
            log_probs = self.model(x)                       # [1, len(ctx), 27]
        return log_probs[0, -1].numpy()                     # prediction after the last char
 
    def get_log_prob_sequence(self, next_chars, context):
        total = 0.0
        for i, c in enumerate(next_chars):
            log_probs = self.get_next_char_log_probs(context + next_chars[:i])
            total += log_probs[self.vocab_index.index_of(c)]
        return float(total)


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """
    chunk_len = 20
    vocab_size = len(vocab_index)
    sos = vocab_index.index_of(' ')
    data = np.array([vocab_index.index_of(c) for c in train_text])

    model = TransformerLM(vocab_size, num_positions=chunk_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch_size = 64
    num_epochs = 6
    num_windows = len(data) - chunk_len
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        starts = np.random.permutation(num_windows)
        total_loss, num_batches = 0.0, 0
        for b in range(0, num_windows, batch_size):
            batch_starts = starts[b:b + batch_size]
            targets = np.stack([data[s:s + chunk_len] for s in batch_starts])      # [B, 20]
            inputs = np.concatenate(
                [np.full((len(batch_starts), 1), sos), targets[:, :-1]], axis=1)   # shift right
            inputs = torch.from_numpy(inputs).long()
            targets = torch.from_numpy(targets).long()

            log_probs = model(inputs)                                              # [B, 20, 27]
            loss = F.nll_loss(log_probs.reshape(-1, vocab_size), targets.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        print("Epoch %d: train loss %.4f (%.0fs elapsed)"
              % (epoch, total_loss / num_batches, time.time() - start_time))

    model.eval()
    return NeuralLanguageModel(model, vocab_index, chunk_len)