# Code adapted from: https://towardsdatascience.com/build-it-yourself-chatbot-api-with-keras-tensorflow-model-f6d75ce957a5
# https://towardsdatascience.com/@andrejusb - Andrejus Baranovskis
import nltk
# nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy as np
import pandas as pd
import pickle
import random
from gensim.test.utils import common_texts
from gensim.test.utils import datapath
from gensim.models.fasttext import FastText, load_facebook_model
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset

np.random.seed(123)
torch.manual_seed(123)
# torch.cuda.manual_seed(123)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# , load_facebook_vector
# things we need for Tensorflow


# import our chat-bot intents file
import json
with open('intents.json') as json_data:
    intents = json.load(json_data)

words = []
sentences = []
classes = []
documents = []
corpus = []
ignore_words = ['?']
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = nltk.word_tokenize(pattern)
        # add to our words list
        words.extend(w)
        # add to documents in our corpus
        documents.append((w, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # add to documents in our corpus
        if (intent['tag']in classes):
            label = classes.index(intent['tag'])
        else:
            label = 9999
        corpus.append((pattern, intent['tag'], label))




# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# # sort classes
classes = sorted(list(set(classes)))
# documents = combination between patterns and intents
# print (len(documents), "documents", documents)
# # classes = intents
print (len(classes), "classes", classes)
# # words = all words, vocabulary
# print (len(words), "unique stemmed words", words)

# print(len(corpus), "corpus", corpus)


# create train and test lists. X - patterns, Y - intents
# use bow for word embeddings or numericalization
# ToDo use FastText for word embeddings

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = nltk.word_tokenize(pattern)
        sentences.append(w)

# create our training data
train_df = pd.DataFrame(corpus, columns=['text', 'intent', 'label'])
x_train = train_df["text"].values
y_train = train_df["label"].values

# myset = set(y_train)
# print("my y values ", myset)


## create tokens, max 1000 tokens
tokenizer = Tokenizer(num_words = 1000)
tokenizer.fit_on_texts(x_train)
word_index = tokenizer.word_index

# # print("common_texts ", common_texts)
# print("sentences ", sentences)
embedding_model = FastText(size=50, window=3, min_count=1, sentences=sentences, iter=10)
# print(embedding_model.wv['blood'])
# print(embedding_model.wv.most_similar("pharmacy"))
# print(embedding_model.wv.similarity('pharmacy', 'patient'))



target_vocab = words
# create emedding layer/matrix
# Embedding layer(dataset’s vocabulary length, word vectors dimension)
# matrix_len = len(target_vocab)
matrix_len = len(word_index) + 1
weights_matrix = np.zeros((matrix_len, 50))
words_found = 0

for i, word in enumerate(target_vocab):
    try:
        weights_matrix[i] = embedding_model.wv[word]
        words_found += 1
    except KeyError:
        weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim, ))

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        max_features, embed_size = weights_matrix.shape
        print("num of embeddings ", max_features)
        print("embedding dimension ", embed_size)
        ## Embedding Layer, Add parameter
        # words in vocab, dimensional embeddings
        self.embedding = nn.Embedding(max_features, embed_size)
        et = torch.tensor(weights_matrix, dtype=torch.float32)
        self.embedding.weight = nn.Parameter(et)
        self.embedding.weight.requires_grad = False

        self.embedding_dropout = nn.Dropout2d(0.1)
        self.lstm = nn.LSTM(50, 40)
        self.linear = nn.Linear(40, 16)
        self.out = nn.Linear(16, 9)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        h_embedding = self.embedding(x)
        h_lstm, _ = self.lstm(h_embedding)
        max_pool, _ = torch.max(h_lstm, 1)
        linear = self.relu(self.linear(max_pool))
        out = self.out(linear)
        return out

model = Model()





## convert texts to padded sequences up to 70 tokens
x_train = tokenizer.texts_to_sequences(x_train)
x_train = pad_sequences(x_train, maxlen = 70)

## create training and validation split
split_size = int(0.8 * len(train_df))
index_list = list(range(len(train_df)))
train_idx, valid_idx = index_list[:split_size], index_list[split_size:]

# print("x_train ", x_train)


## create iterator objects for train and valid datasets
x_tr = torch.tensor(x_train[train_idx], dtype=torch.long)
y_tr = torch.tensor(y_train[train_idx]) # dtype=torch.float32
train = TensorDataset(x_tr, y_tr)
trainloader = DataLoader(train, batch_size=128)


x_val = torch.tensor(x_train[valid_idx], dtype=torch.long)
y_val = torch.tensor(y_train[valid_idx])
valid = TensorDataset(x_val, y_val)
validloader = DataLoader(valid, batch_size=128)

# loss_function = nn.BCEWithLogitsLoss(reduction='mean')
# optimizer = optim.Adam(model.parameters())
loss_function = nn.CrossEntropyLoss()
# criterion = nn.CrossEntropyLoss()
# loss = criterion(output, target)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay= 1e-6, momentum = 0.9, nesterov = True)


# Train model
## run for 10 Epochs
for epoch in range(1, 11):
    train_loss, valid_loss = [], []

## training part
    model.train()
    for data, target in trainloader:
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

    ## evaluation part
    model.eval()
    for data, target in validloader:
        output = model(data)

        loss = loss_function(output, target)
        valid_loss.append(loss.item())
        print(loss.item())

        # m = nn.Softmax()
        # outputs=m(output)
    print ("Epoch:", epoch, "Training Loss: ", np.mean(train_loss), "Valid Loss: ", np.mean(valid_loss))


# get predictions
dataiter = iter(validloader)
data, labels = dataiter.next()
output = model(data)
# m = nn.Softmax()
# outputs=m(output)
_, preds_tensor = torch.max(output, 1)
preds = np.squeeze(preds_tensor.numpy())

print ("Actual:", labels)
print ("Predicted:", preds)


#
# print(len(train_x[0]))
# print(len(train_y[0]))

## create training and validation split
# split_size = int(0.8 * len(training))
# index_list = list(range(len(training)))
# train_idx, valid_idx = index_list[:split_size], index_list[split_size:]

# print("train_idx ", train_idx)
# print("valid_idx ", valid_idx)

# print("x_train ", train_x)
# print("x_train ", train_x[train_idx])
# print("y_train ", train_y[train_idx])

# ## create iterator objects for train and valid datasets
# x_tr = torch.tensor(train_x[train_idx], dtype=torch.long)
# y_tr = torch.tensor(train_y[train_idx], dtype=torch.float32)
# train = TensorDataset(x_tr, y_tr)
# trainloader = DataLoader(train, batch_size=128)
#
# x_val = torch.tensor(train_x[valid_idx], dtype=torch.long)
# y_val = torch.tensor(train_y[valid_idx], dtype=torch.float32)
# valid = TensorDataset(x_val, y_val)
# validloader = DataLoader(valid, batch_size=128)
#
# loss_function = nn.BCEWithLogitsLoss(reduction='mean')
# optimizer = optim.Adam(model.parameters())




#
#
# # create our training data
#
# # create an empty array for our output
#
# # training set, bag of words for each sentence
#
# # shuffle our features and turn into np.array
#
# # create train and test lists. X - patterns, Y - intents
#
#
#
# # loss = nn.CrossEntropyLoss()
#
# # ToDo - use PyTorch instead
# # Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# # equal to number of intents to predict output intent with softmax
# model = IntentClassificationNN(weights_matrix, 50, 4)
#
# lr = 1e-1 # learning rate
# # define optimizer ?
# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
#
# # Defines a MSE loss function
# loss_fn = nn.MSELoss(reduction='mean')

#
# # inspect its parameters using its state_dict
# print(model.state_dict())
#
# # Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
# def make_train_step(model, loss_fn, optimizer):
#     # Builds function that performs a step in the train loop
#     def train_step(x, y):
#         # Sets model to TRAIN mode
#         model.train()
#         # Makes predictions
#         yhat = model(x)
#         # Computes loss
#         loss = loss_fn(y, yhat)
#         # Computes gradients
#         loss.backward()
#         # Updates parameters and zeroes gradients
#         optimizer.step()
#         optimizer.zero_grad()
#         # Returns the loss
#         return loss.item()
#
#     # Returns the function that will be called inside the train loop
#     return train_step
#
# # Creates the train_step function for our model, loss function and optimizer
# train_step = make_train_step(model, loss_fn, optimizer)
# losses = []
#
# # For each epoch...
# for epoch in range(n_epochs):
#     # Performs one train step and returns the corresponding loss
#     loss = train_step(x_train_tensor, y_train_tensor)
#     losses.append(loss)
#
# # Checks model's parameters
# print(model.state_dict())
#
# # Fit the model
#
#
#
# def clean_up_sentence(sentence):
#     # tokenize the pattern - split words into array
#     sentence_words = nltk.word_tokenize(sentence)
#     # stem each word - create short form for word
#     sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
#     return sentence_words
#
# # return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
# def bow(sentence, words, show_details=True):
#     # tokenize the pattern
#     sentence_words = clean_up_sentence(sentence)
#     # bag of words - matrix of N words, vocabulary matrix
#     bag = [0]*len(words)
#     for s in sentence_words:
#         for i,w in enumerate(words):
#             if w == s:
#                 # assign 1 if current word is in the vocabulary position
#                 bag[i] = 1
#                 if show_details:
#                     print ("found in bag: %s" % w)
#
#     return(np.array(bag))

#
# p = bow("Load blood pessure for patient", words)
# print (p)
# print (classes)
#
# inputvar = pd.DataFrame([p], dtype=float, index=['input'])
#
# # predict inputvar
# print(model.predict(inputvar))

# save model to file
# pickle.dump(model, open("chatme-model.pkl", "wb"))

# save all of our data structures
# pickle.dump( {'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open( "chatme-data.pkl", "wb" ) )
