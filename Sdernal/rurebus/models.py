from typing import Tuple

import torch
import torch.nn as nn
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from pytorch_transformers import BertModel

from .indexers import DocumentIndexer, BertIndexer

default_dims = {
    'hidden': 400,
    'feedforward': 100,
    'dropout': 0.4,
    'char_emb_size': 20,
    'case_emb_size': 10,
    'chars_hidden': 50,
    'pos_emb_size': 30,
    'ner_emb_size': 10,
    'dropout_input': 0.2,
    'dropout_lstm': 0.2,
    'attention_dim': 30
}

# -------------------- Models for NER ----------------------------------------------------------------------------------

class SelfAttentionLSTMEncoder(nn.Module):
    def __init__(self, indexer: DocumentIndexer, embedding_matrix: torch.Tensor, dims=None):
        super(SelfAttentionLSTMEncoder, self).__init__()
        if dims is None:
            dims = default_dims
        self.dims = dims
        words_emb_size = embedding_matrix.size(1)
        self.word_embedder = nn.Embedding.from_pretrained(embedding_matrix)
        self.char_embedder = nn.Embedding(len(indexer.char_vocab), dims['char_emb_size'])
        self.case_embedder = nn.Embedding(len(indexer.case_vocab), dims['case_emb_size'])
        self.pos_embedder = nn.Embedding(len(indexer.pos_vocab), dims['pos_emb_size'])
        self.char_encoder = PytorchSeq2VecWrapper(nn.LSTM(dims['char_emb_size'], dims['chars_hidden'],
                                                          batch_first=True, bidirectional=True))

        total_emb_size = words_emb_size + dims['case_emb_size'] + 2*dims['chars_hidden'] + dims['pos_emb_size']
        self.encoder = PytorchSeq2SeqWrapper(nn.LSTM(total_emb_size, dims['hidden'],
                                                     batch_first=True, bidirectional=True, num_layers=2))
        self.feedforward = FeedForward(2*dims['hidden'], 1, dims['feedforward'], activations=nn.Tanh())
        self.Q = nn.Linear(dims['feedforward'], dims['feedforward'])
        self.K = nn.Linear(dims['feedforward'], dims['feedforward'])
        self.V = nn.Linear(dims['feedforward'], dims['feedforward'])
        self.ln = nn.LayerNorm(dims['feedforward'])
        self.out = nn.Linear(dims['feedforward'], len(indexer.ner_vocab))
        self.dropout = nn.Dropout(dims['dropout'])

    def forward(self, words: torch.Tensor, chars: torch.Tensor, cases: torch.Tensor, pos: torch.Tensor):
        mask = (words != 0).to(torch.long)
        batch_size = words.size(0)
        words = self.word_embedder(words)  # batch_size, seq_len, word_emb
        cases = self.case_embedder(cases)
        pos = self.pos_embedder(pos)

        chars = chars.view(chars.size(0) * chars.size(1), -1)
        chars_mask = (chars != 0).to(torch.long)
        chars = self.char_embedder(chars)
        chars = self.char_encoder(chars, chars_mask)
        chars = chars.view(batch_size, chars.size(0) // batch_size, -1)

        features = torch.cat([words, chars, cases, pos], dim=-1) # batch_size, seq_len, total_emb

        z = self.encoder(features, mask)  # batch_size, seq_len, hidden
        z = self.dropout(z)
        z = self.feedforward(z)  # batch_size, seq_len, feats

        # attention
        qs, ks, vs = self.Q(z), self.K(z), self.V(z)
        scores = torch.bmm(qs, ks.transpose(1, 2))  # batch_size, seq_len, seq_len
        scores = scores / torch.sqrt(torch.tensor(self.dims['feedforward'], dtype=torch.float))
        scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e10)
        scores = torch.softmax(scores, dim=-1)  # batch_size, seq_len, seq_len

        z = torch.bmm(scores, vs)
        z = self.ln(z)  # batch_size, seq_len, feats

        out = self.out(z)
        return out


class Classifier(nn.Module):

    def __init__(self, hidden_size: int = 1024, num_labels: int = 2):
        super(Classifier, self).__init__()
        dropout = 0.1
        self.hidden_size = hidden_size
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class BertEncoder(nn.Module):
    def __init__(self, bert: BertModel, labels_count):
        super(BertEncoder, self).__init__()
        self.bert = bert
        self.classifier = Classifier(768, labels_count)

    def forward(self, tokens: torch.Tensor):
        mask = (tokens != 0).to(torch.long)
        bert_output = self.bert(tokens, attention_mask=mask)

        z = bert_output[0][0]
        out = self.classifier(z)
        return out


# -------------------- Models for RE -----------------------------------------------------------------------------------

class SampleEncoder(nn.Module):
    def __init__(self, indexer: DocumentIndexer, embedding_matrix: torch.Tensor, dims=None):
        super(SampleEncoder, self).__init__()
        if dims is None:
            dims = default_dims
        self.dims = dims
        words_emb_size = embedding_matrix.size(1)
        self.word_embedder = nn.Embedding.from_pretrained(embedding_matrix)
        self.word_dropout = nn.Dropout(dims['dropout_input'])

        self.char_embedder = nn.Embedding(len(indexer.char_vocab), dims['char_emb_size'])
        self.case_embedder = nn.Embedding(len(indexer.case_vocab), dims['case_emb_size'])
        self.pos_embedder = nn.Embedding(len(indexer.pos_vocab), dims['pos_emb_size'])
        self.ner_embedder = nn.Embedding(len(indexer.ner_vocab), dims['ner_emb_size'])
        self.char_encoder = PytorchSeq2VecWrapper(nn.LSTM(dims['char_emb_size'], dims['chars_hidden'],
                                                          batch_first=True, bidirectional=True))

        total_emb_size = words_emb_size + dims['case_emb_size'] + 2 * dims['chars_hidden'] \
                         + dims['pos_emb_size'] + dims['ner_emb_size']

        self.encoder = PytorchSeq2SeqWrapper(nn.LSTM(total_emb_size, dims['hidden'],
                                                     batch_first=True, bidirectional=True, num_layers=2))
        self.sent_dropout = nn.Dropout(dims['dropout_lstm'])

        self.feedforward = FeedForward(2 * dims['hidden'], 1, dims['feedforward'], activations=nn.Tanh())
        self.attention = nn.Linear(2 * dims['hidden'], dims['attention_dim'])
        self.scores = nn.Linear(dims['attention_dim'], 1)
        self.hidden2tag = nn.Linear(2 * dims['hidden'], len(indexer.relation_type_vocab))
        self.out_dropout = nn.Dropout(dims['dropout_lstm'])

    def forward(self, words: torch.Tensor, chars: torch.Tensor, cases: torch.Tensor,
                pos: torch.Tensor, ner: torch.Tensor):
        mask = (words != 0).to(torch.long)
        batch_size = words.size(0)
        words = self.word_embedder(words)  # batch_size, seq_len, word_emb
        cases = self.case_embedder(cases)
        pos = self.pos_embedder(pos)
        ner = self.ner_embedder(ner)

        chars = chars.view(chars.size(0) * chars.size(1), -1)
        chars_mask = (chars != 0).to(torch.long)
        chars = self.char_embedder(chars)
        chars = self.char_encoder(chars, chars_mask)
        chars = chars.view(batch_size, chars.size(0) // batch_size, -1)

        features = torch.cat([words, chars, cases, pos, ner], dim=-1)  # batch_size, seq_len, total_emb

        features = self.sent_dropout(features)
        out = self.encoder(features, mask)
        out = self.out_dropout(out)

        # "attention"
        scores = self.attention(out)  # batch_size, seq_len, attn_dim
        scores = torch.tanh(scores)
        scores = self.scores(scores)  # batch_size, seq_len, 1
        scores = torch.softmax(scores, dim=1)
        scores = scores.transpose(1, 2)  # batch_size, 1, seq_len,
        out = torch.bmm(scores, out)  # batch_size, 1, n_dim
        out = out.squeeze(1)  # batch_size, n_dim

        out = self.hidden2tag(out)  # batch_size, n_labels
        return out


class BertRelationClassifier(nn.Module):
    def __init__(self, bert: BertModel, indexer: BertIndexer):
        super(BertRelationClassifier, self).__init__()
        bert_out = 768
        ner_embedding_size = 32

        self.bert = bert
        self.bert_dropout = nn.Dropout(0.2)
        self.ner_embedder = nn.Embedding(len(indexer.ner_vocab), ner_embedding_size)

        total_size = bert_out + ner_embedding_size
        self.attention = nn.Linear(total_size, total_size)
        self.scores = nn.Linear(total_size, 1)

        self.classifier = Classifier(bert_out + ner_embedding_size, len(indexer.rel_vocab))

    def forward(self, tokens: torch.Tensor, ner: torch.Tensor):
        mask = (tokens != 0).to(torch.long)
        bert_output = self.bert(tokens, attention_mask=mask,  output_all_encoded_layers=False)

        bert = bert_output[0]
        bert = self.bert_dropout(bert)
        ner = self.ner_embedder(ner)

        z = torch.cat([bert, ner], dim=-1)  # batch_size, seq_len, total_size

        # "attention"
        scores = self.attention(z)  # batch_size, seq_len, attn_dim
        scores = torch.tanh(scores)
        scores = self.scores(scores)  # batch_size, seq_len, 1
        scores = torch.softmax(scores, dim=1)
        scores = scores.transpose(1, 2)  # batch_size, 1, seq_len,
        out = torch.bmm(scores, z)  # batch_size, 1, n_dim
        out = out.squeeze(1)  # batch_size, n_dim
        out = self.classifier(out)

        return out


class AddOnLayer(nn.Module):
    def __init__(self, hidden_size: int = 768):
        super(AddOnLayer, self).__init__()
        dropout = 0.1
        self.hidden_size = hidden_size
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return x


class RBert(nn.Module):
    def __init__(self, bert: BertModel, indexer: BertIndexer):
        super(RBert, self).__init__()
        bert_out = 768

        self.bert = bert
        self.h0 = AddOnLayer(bert_out)
        self.h1 = AddOnLayer(bert_out)
        self.h2 = AddOnLayer(bert_out)
        self.out = nn.Linear(bert_out*3, len(indexer.rel_vocab))

    def forward(self, tokens: torch.Tensor, e1_pos: Tuple[int, int], e2_pos: Tuple[int, int]):
        mask = (tokens != 0).to(torch.long)
        bert_output = self.bert(tokens, attention_mask=mask, output_all_encoded_layers=False)[0]
        # bert_output shape: batch_size, seq_len, 768
        h0 = bert_output[:, 0, :]  # batch_size, 1, 768

        h1_batch = []
        h2_batch = []
        for i, (e1, e2) in enumerate(zip(e1_pos, e2_pos)):
            bert_batch = bert_output[i]  # seq_len, 768
            h1 = bert_batch[e1[0]:e1[1]+1, :]  # e1_len, 768
            h2 = bert_batch[e2[0]:e2[1]+1, :]  # e2_len, 768

            h1 = h1.mean(dim=0)  # 768
            h2 = h2.mean(dim=0)  # 768
            h1_batch.append(h1)
            h2_batch.append(h2)

        h1 = torch.stack(h1_batch)  # batch_size, 768
        h2 = torch.stack(h2_batch)  # batch_size, 768

        # h0 = h0.squeeze(dim=1)  # batch_size, 768

        h0 = self.h0(h0)  # batch_size, 768
        h1 = self.h1(h1)  # batch_size, 768
        h2 = self.h2(h2)  # batch_size, 768

        h3 = torch.cat([h0, h1, h2], dim=-1)
        out = self.out(h3)

        return out


