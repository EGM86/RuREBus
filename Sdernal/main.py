import os
from os.path import exists, join, splitext

import torch
from pytorch_pretrained_bert import BertModel
from tokenizers import BertWordPieceTokenizer
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from torch.utils.tensorboard import SummaryWriter

from rurebus.datasets import RBertDataset
from rurebus.evaluators import RelEvaluator
from rurebus.indexers import BertIndexer
from rurebus.models import RBert
from rurebus.modules import RelClassifier
from rurebus.readers import BertReader
from rurebus.samplers import WholeSampler, PartialSampler
from rurebus.trainer import RelTrainer

device = torch.device('cuda')
# creating tokenizer from vocab downloaded from huggingface
tokenizer = BertWordPieceTokenizer('multi_cased/vocab.txt')
reader = BertReader(tokenizer, 'debug.txt', max_sent_len=100)

test_folder = 'data/test_ner_only'
documents = reader.read_folder('data/train')
test_docs = reader.read_folder(test_folder)

train_docs = []
dev_docs = []
# split on train and dev
for i, doc in enumerate(documents):
    if i % 5 != 0:
        train_docs.append(doc)
    else:
        dev_docs.append(doc)

# fill vocabs
indexer = BertIndexer()
indexer.index_documents(train_docs+dev_docs+test_docs)
train_relation_sampler = PartialSampler()
test_relation_sampler = WholeSampler()

# load pretrained BERT and create model
bert = BertModel.from_pretrained('bert-base-multilingual-cased')
model = RBert(bert, indexer)
module = RelClassifier(indexer.rel_vocab, model)
model.to(device)

results_folder = './results/R_BERT'
if not exists(results_folder):
    os.mkdir(results_folder)

writer = SummaryWriter()
trainer = RelTrainer(module, writer, results_folder)

# fixes samples of dev dataset
dev_dataset = RBertDataset(indexer, train_relation_sampler, dev_docs, device)
dev_sampler = RandomSampler(dev_dataset)
dev_iterator = DataLoader(dev_dataset, batch_size=12, sampler=dev_sampler, collate_fn=dev_dataset.collate_fn)

for i in range(5):
    # samples different negative examples for training
    train_dataset = RBertDataset(indexer, train_relation_sampler, documents, device)
    train_sampler = RandomSampler(train_dataset)
    train_iterator = DataLoader(train_dataset, batch_size=12, sampler=train_sampler,
                                collate_fn=train_dataset.collate_fn)

    print('Epoch: ', trainer.epoch)
    trainer.train_epoch(train_iterator)
    trainer.test_epoch(dev_iterator)
    print()

# load best epoch
model.load_state_dict(torch.load(join(results_folder, 'best.pt')))

# evaluate on test
test_dataset = RBertDataset(indexer, test_relation_sampler, test_docs, device)
test_sampler = SequentialSampler(test_dataset)
test_iterator = DataLoader(test_dataset, batch_size=12, sampler=test_sampler, collate_fn=test_dataset.collate_fn)

# save results
relations_dir = join(results_folder, 'relations')
result = trainer.decode(test_iterator)
evaluator = RelEvaluator()
evaluator.save_result(test_dataset, result, relations_dir)

# join with NER markup
result_dir = join(results_folder, 'set_2')
if not exists(result_dir):
    os.mkdir(result_dir)


for file in os.listdir(test_folder):
    file_name, ext = splitext(file)
    if ext == '.ann':
        ner_file = join(test_folder, file)
        rel_file = join(relations_dir, file)
        res_file = join(result_dir, file)
        with open(res_file, 'w', encoding='utf-8') as res:
            res.write(open(ner_file, 'r', encoding='utf-8').read())
            if exists(rel_file):
                res.write(open(rel_file, 'r', encoding='utf-8').read())
