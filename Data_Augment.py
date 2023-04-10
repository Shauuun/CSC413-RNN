import json
import torch
import nltk
nltk.download('punkt')
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.model_selection import train_test_split
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from RNN_Network import RNNAttention
from data_aug import *




# Load raw data and process it
def process_data(file_path, num_samples=10000):
    texts = []
    labels = []

    with open(file_path, "r", encoding='utf-8') as file:
        for idx, line in enumerate(file):
            if idx >= num_samples:
                break

            data = json.loads(line)
            text = data["text"].strip().replace('\n', '').replace('\r','')
            stars = data["stars"]
            label = 1 if stars >= 4 else 0

            texts.append(text)
            labels.append(label)
    return texts, labels


# Create a dataset class
class TextDataset(Dataset):
    def __init__(self, comment, labels, vocab, seq_length):
        self.texts = comment
        self.labels = labels
        self.vocab = vocab
        self.seq_length = seq_length

    def __len__(self):
        return len(self.labels)

    def encode_and_pad_text(self, tokens):
        encoded_text = []
        for token in tokens:
            token_id = self.vocab.get(token,
                                      1)  # If the token is not found in the vocabulary, use the ID for the "_UNK_" token (1)
            encoded_text.append(token_id)

        if len(encoded_text) < self.seq_length:
            padding_length = self.seq_length - len(encoded_text)
            encoded_text.extend([0] * padding_length)
        else:
            encoded_text = encoded_text[:self.seq_length]

        return encoded_text

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        tokens = word_tokenize(text.lower())
        padded_text = self.encode_and_pad_text(tokens)

        return torch.tensor(padded_text), torch.tensor(label)


# Define the function to create vocabulary
def build_vocab(texts, max_vocab_size):
    counter = Counter()
    for text in texts:
        tokens = word_tokenize(text.lower())
        counter.update(tokens)

    vocab = {token: i + 2 for i, (token, _) in enumerate(counter.most_common(max_vocab_size - 2))}
    vocab["_PAD_"] = 0
    vocab["_UNK_"] = 1

    return vocab


def get_accuracy(model, data):
    # note: why should we use a larger batch size here?
    loader = torch.utils.data.DataLoader(data, batch_size=256)

    model.eval()  # annotate model for evaluation (why do we need to do this?)

    correct = 0
    total = 0
    for inputs, labels in loader:
        output = model(inputs)
        pred = output.max(1, keepdim=True)[1]
        # print(pred)
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += inputs.shape[0]

    return correct / total


def train(model, train_data, valid_data, batch_size=32, weight_decay=0.0,
          learning_rate=0.001, num_epochs=7, momentum=0.9):
    loss_fnc = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    iters, losses, train_acc, val_acc = [], [], [], []
    n = 0
    # loading data
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        for inputs, labels in iter(train_loader):
            if inputs.size()[0] < batch_size:
                continue  # skip the last batch which in many case is smaller than the rest
            # print(inputs.shape)

            model.train()  # annotate model for training
            out = model(inputs)
            loss = loss_fnc(out, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # save the current training information
        iters.append(n)
        losses.append(float(loss) / batch_size)  # compute *average* loss
        train_acc.append(get_accuracy(model, train_data))  # compute training accuracy
        print("Traning Accuracy For Epoch " + str(epoch) + " is : {}".format(train_acc[-1]))
        if valid_data != None:
            val_acc.append(get_accuracy(model, valid_data))  # compute validation accuracy
            print("Validation Accuracy For Epoch " + str(epoch) + " is: {}".format(val_acc[-1]))
        n += 1
        if train_acc[-1] == 1:
            break

    # plotting
    plt.title("Learning Curve")
    plt.plot(iters, losses, label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()

    plt.title("Learning Curve")
    plt.plot(iters, train_acc, label="Train")
    if valid_data != None:
        plt.plot(iters, val_acc, label="Validation")
        print("Final Validation Accuracy: {}".format(val_acc[-1]))
    plt.xlabel("Iterations")
    plt.ylabel("Training Accuracy")
    plt.legend(loc='best')
    plt.show()

    print("Final Training Accuracy: {}".format(train_acc[-1]))


def overfit(model, over_fit_data, learn_rate, batch_size, num_epochs=1):
    valid_data = None

    train(model, over_fit_data, valid_data, learning_rate=learn_rate, batch_size=batch_size,
          num_epochs=num_epochs)


if __name__ == "__main__":
    # Load and process the data
    raw_data = 'archive/yelp_academic_dataset_review.json'
    num_data = 100
    texts, labels = process_data(raw_data, num_samples=num_data)

    # First, split the data into train set (60%) and a temporary test set (40%)
    train_texts, temp_test_texts, train_labels, temp_test_labels = train_test_split(texts, labels, test_size=0.2,
                                                                                    random_state=42)
    print(train_texts[20:30])
    print(train_labels[20:30])
    

    # Now, split the temporary test set into validation set (50%) and test set (50%)
    val_texts, test_texts, val_labels, test_labels = train_test_split(temp_test_texts, temp_test_labels, test_size=0.5,
                                                                      random_state=42)

    # Create vocabulary and encode the texts
    max_vocab_size = 15000
    vocab = build_vocab(train_texts, max_vocab_size)

    # Define sequence length for padding
    seq_length = 200


    # train data agumentation
    train_texts, train_labels = data_augment(train_texts,train_labels)

    # Create datasets and data loaders
    train_dataset = TextDataset(train_texts, train_labels, vocab, seq_length)
    val_dataset = TextDataset(val_texts, val_labels, vocab, seq_length)
    test_dataset = TextDataset(test_texts, test_labels, vocab, seq_length)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    embedding_dim = 64
    hidden_dim = 64
    output_dim = 2
    batch_size = 32
    num_head = 8
    learn_rate = 0.01
    num_epochs = 10

    model = RNNAttention(vocab_size=len(vocab), embedding_dim=embedding_dim, hidden_dim=hidden_dim,
                        output_dim=output_dim, num_heads=num_head)
    model.to(device)
    train(model, train_dataset, val_dataset, learning_rate=learn_rate, batch_size=batch_size,
        num_epochs=num_epochs)
