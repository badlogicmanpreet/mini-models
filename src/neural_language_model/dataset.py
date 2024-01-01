import torch
import random

class Dataset:

    def __init__(self, path, block_size):
        self.words = open(path).read().splitlines()
        self.chars = sorted(list(set(''.join(self.words))))
        self.stoi = {ch : i+1 for i, ch in enumerate(self.chars)}
        self.stoi['.'] = 0
        self.itos = {i : ch for ch, i in self.stoi.items()}
        self.block_size = block_size # context length: how many characters do we take to predict the next one?
    
    def get_vocab(self):
        return self.chars

    def get_vocab_size(self):
        return len(self.chars) + 1
    
    def get_words(self):
        return self.words

    def get_stoi(self):
        return self.stoi
    
    def get_itos(self):
        return self.itos
    
    def get_block_size(self):
        return self.block_size

    def build_dataset(self, words):
        X, Y = [], []

        for word in words:
            # print(word)
            context = [0] * self.block_size
            for ch in word + '.':
                ix = self.stoi[ch]
                X.append(context)
                Y.append(ix)
                # print(''.join(itos[i] for i in context), '--->', itos[ix])
                context = context[1:] + [ix]

        X = torch.tensor(X)
        Y = torch.tensor(Y)
        print(X.shape, X.dtype, Y.shape, Y.dtype)
        return X, Y

    def train_dev_test_split(self):
        random.seed(42)
        random.shuffle(self.words)
        n1 = int(len(self.words) * 0.8) # 80% of the words
        n2 = int(len(self.words) * 0.9) # 90% of the words

        Xtr, Ytr = self.build_dataset(self.words[:n1]) # training set -> upto n1
        Xdev, Ydev = self.build_dataset(self.words[n1:n2]) # development set -> (n2 - n1)
        Xte, Yte = self.build_dataset(self.words[n2:]) # test set -> (n - n2)
        return Xtr, Ytr, Xdev, Ydev, Xte, Yte