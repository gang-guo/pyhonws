import torch
import string,re
import torchtext

MAX_WORDS=10000 #仅考虑最高频的10000个词
MAX_LEN=200 #每个样本仅保留200个词的长度
BATCH_SIZE=64 #批次

tokenizer = lambda x:re.sub('[%s]'%string.punctuation,'',x).split(' ')

def filterLowFreqWords(arr,vocab):

    return [[x if x < MAX_WORDS else 0 for x in example] for example in arr]


TEXT = torchtext.data.Field(sequential=True,tokenize=tokenizer,lower=True,fix_length=MAX_LEN,postprocessing=filterLowFreqWords, batch_first=True)
LABEL = torchtext.data.Field(sequential=False, use_vocab=False, batch_first=True)

ds_train,ds_test = torchtext.data.TabularDataset.splits(path='./',train='train.tsv',test='test.tsv',format='tsv',
                                                  fields=[('label',LABEL),('text',TEXT)],skip_header=False)

TEXT.build_vocab(ds_train)
VOCAB_SIZE = len(TEXT.vocab)
train_iter,test_iter = torchtext.data.Iterator.splits((ds_train,ds_test), sort_within_batch=True, sort_key=lambda x:len(x.text),
                                                      batch_sizes=(BATCH_SIZE,BATCH_SIZE))

class DataLoader:
    def __init__(self, data_iter):
        self.data_iter = data_iter
        self.length = len(data_iter)

    def __len__(self):
        return self.length

    def __iter__(self):
        for batch in self.data_iter:
            yield (torch.transpose(batch.text, 0, 1),
                   torch.unsqueeze(batch.label.float(), dim=1))


dl_train = DataLoader(train_iter)
dl_test = DataLoader(test_iter)

import torchkeras
from torch import nn

torch.random.seed()


class Net(torchkeras.Model):
    def __init__(self):
        super(Net, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=MAX_WORDS, embedding_dim=128, padding_idx=1)
        self.lstm = nn.LSTM(input_size=128, hidden_size=256)

    def forward(self, x, hidden):
        em = self.embedding()
        re, h = self.lstm(em, hidden)
        return re,h,em

    def init_hidden(self, bsz, requires_grad=True):
        weight = next(self.parameters())
        return (weight.new_zeros((1, bsz, 256), requires_grad=requires_grad),
                    weight.new_zeros((1, bsz, 256), requires_grad=requires_grad))

model = Net()

print(model.embedding(torch.LongTensor([32])))

# class Net2(torchkeras.Model):
#     def __init__(self):
#         super(Net2, self).__init__()
#
#         self.lstm = nn.LSTM(input_size=256, hidden_size=128)
#
#     def forward(self, x, hidden):
#         re, h = self.lstm(x, hidden)
#         return re,h
#
#     def init_hidden(self, bsz, requires_grad=True):
#         weight = next(self.parameters())
#         return (weight.new_zeros((1, bsz, 128), requires_grad=requires_grad),
#                     weight.new_zeros((1, bsz, 128), requires_grad=requires_grad))
#
# model2 = Net2()
#
# loss_fn = nn.MSELoss()
# optimer1 = torch.optim.Adam(model.parameters(), lr=1e-4)
# optimer2 = torch.optim.Adam(model2.parameters(), lr=1e-4)
#
# # Remove this part
# def repackage_hidden(h):
#     """Wraps hidden states in new Tensors, to detach them from their history."""
#     if isinstance(h, torch.Tensor):
#         return h.detach()
#     else:
#         return tuple(repackage_hidden(v) for v in h)
#
# for epoch in range(1000):
#     model.train()
#     model2.train()
#     it = iter(train_iter)
#     hidden = model.init_hidden(MAX_LEN)
#     hidden2 = model2.init_hidden(MAX_LEN)
#     for i, batch in enumerate(it):
#         data = batch.text
#         hidden = repackage_hidden(hidden)
#         hidden2 = repackage_hidden(hidden2)
#         model.zero_grad()
#         model2.zero_grad()
#         output, hidden,em = model(data, hidden)
#         output, hidden2 = model2(output, hidden2)
#         loss = loss_fn(em, output)
#         loss.backward()
#         optimer1.step()
#         optimer2.step()
#         if i % 10 == 0:
#             print("epoch", epoch, "iter", i, "loss", loss.item())
#
