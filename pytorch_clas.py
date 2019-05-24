from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, utils

train = pd.read_csv("train.csv")
x_train = train["text"].values
y_train = train["label"].values

print(x_train)
print(y_train)


np.random.seed(123)
torch.manual_seed(123)
# torch.cuda.manual_seed(123)
torch.backends.cudnn.deterministic = True


## create tokens, max 1000 tokens
tokenizer = Tokenizer(num_words = 1000)
tokenizer.fit_on_texts(x_train)
word_index = tokenizer.word_index


## convert texts to padded sequences up to 70 tokens
x_train = tokenizer.texts_to_sequences(x_train)
x_train = pad_sequences(x_train, maxlen = 70)


# create an embedding matrix containing the word vector for every word in the vocabulary.
EMBEDDING_FILE = 'glove.6B/glove.6B.300d.txt'

embeddings_index = {}
for i, line in enumerate(open(EMBEDDING_FILE)):
    val = line.split()
    embeddings_index[val[0]] = np.asarray(val[1:], dtype='float32')



embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# print(embedding_matrix[1])

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        max_features, embed_size = embedding_matrix.shape
        print("num of embeddings ", max_features)
        print("embedding dimension ", embed_size)
        ## Embedding Layer, Add parameter
        # words in vocab, dimensional embeddings
        self.embedding = nn.Embedding(max_features, embed_size)
        et = torch.tensor(embedding_matrix, dtype=torch.float32)
        self.embedding.weight = nn.Parameter(et)
        self.embedding.weight.requires_grad = False

        self.embedding_dropout = nn.Dropout2d(0.1)
        self.lstm = nn.LSTM(300, 40)
        self.linear = nn.Linear(40, 16)
        self.out = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        h_embedding = self.embedding(x)
        h_lstm, _ = self.lstm(h_embedding)
        max_pool, _ = torch.max(h_lstm, 1)
        linear = self.relu(self.linear(max_pool))
        out = self.out(linear)
        return out

model = Model()

from torch.utils.data import TensorDataset

train_df = pd.read_csv("train.csv")

## create training and validation split
split_size = int(0.8 * len(train_df))
index_list = list(range(len(train_df)))
train_idx, valid_idx = index_list[:split_size], index_list[split_size:]

# print("train_idx ", train_idx)
# print("valid_idx ", valid_idx)

print("x_train ", x_train)
#
# print("x_train ", x_train[train_idx])
# print("y_train ", y_train[train_idx])


## create iterator objects for train and valid datasets
x_tr = torch.tensor(x_train[train_idx], dtype=torch.long)
y_tr = torch.tensor(y_train[train_idx], dtype=torch.float32)
train = TensorDataset(x_tr, y_tr)
trainloader = DataLoader(train, batch_size=128)


x_val = torch.tensor(x_train[valid_idx], dtype=torch.long)
y_val = torch.tensor(y_train[valid_idx], dtype=torch.float32)
valid = TensorDataset(x_val, y_val)
validloader = DataLoader(valid, batch_size=128)

loss_function = nn.BCEWithLogitsLoss(reduction='mean')
optimizer = optim.Adam(model.parameters())


# Train model
## run for 10 Epochs
for epoch in range(1, 11):
    train_loss, valid_loss = [], []

## training part
    model.train()
    for data, target in trainloader:
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target.view(-1,1))
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

    ## evaluation part
    model.eval()
    for data, target in validloader:
        output = model(data)

        loss = loss_function(output, target.view(-1,1))
        valid_loss.append(loss.item())


# get predictions
dataiter = iter(validloader)
data, labels = dataiter.next()
output = model(data)
_, preds_tensor = torch.max(output, 1)
preds = np.squeeze(preds_tensor.numpy())
