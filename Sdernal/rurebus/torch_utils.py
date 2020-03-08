from typing import List

import torch
from tqdm import tqdm

from .vocab import Vocab


def pad_matrix(matrices: List[torch.Tensor]):
    max_paths = max([m.size(0) for m in matrices])
    max_len = max([m.size(1) for m in matrices])

    out_tensor = torch.zeros((len(matrices), max_paths, max_len), dtype=torch.long)
    for i, matrix in enumerate(matrices):
        paths = matrix.size(0)
        length = matrix.size(1)

        out_tensor[i, :paths, :length] = matrix

    return out_tensor


def load_embeddings(tokens_vocab: Vocab, embeddings_file: str, embeddings_size: int):
    embeddings_matrix = torch.zeros((len(tokens_vocab), embeddings_size))
    with open(embeddings_file, encoding='utf-8') as f:
        for l in tqdm(f):
            line = l.split()
            word = line[0]
            if word in tokens_vocab:
                weights = list(map(float, line[1:]))
                weights = torch.tensor(weights)
                embeddings_matrix[tokens_vocab[word]] = weights
    return embeddings_matrix


def random_embeddings(tokens_vocab: Vocab, embeddings_size: int):
    return torch.rand((len(tokens_vocab), embeddings_size))
