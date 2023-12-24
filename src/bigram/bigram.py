import torch
import torch.nn.functional as F


class Bigram:
    def __init__(self):
            # initialize the neural network
            g = torch.Generator().manual_seed(2147483647)
            self.W = torch.randn((27, 27), generator=g, requires_grad=True)

    # load the data
    def load_data(self):
        with open('src/bigram/names.txt', 'r') as f:
            words = f.read().splitlines()
        return words

    # create the vocabulary
    def create_vocab(self, words):
        # create the vocabulary
        chars = sorted(list(set(''.join(words)))) # sorted list of unique chars
        
        stoi = {ch: i+1 for i, ch in enumerate(chars)} # string to int
        stoi['.'] = 0 # end symbol

        itos = {i: ch for ch, i in stoi.items()} # int to string
        return stoi, itos

    # create the training set of bigrams
    def create_dataset_bigram(self, words, stoi, itos):
        # create the training set of bigrams
        xs, ys = [], []
        for word in words:
            chs = ['.'] + list(word) + ['.'] # add start and end symbols
            for ch1, ch2 in zip(chs, chs[1:]): # group the consective chars in a word
                i = stoi[ch1]
                j = stoi[ch2]
                xs.append(i)
                ys.append(j)

        xs = torch.tensor(xs) # tensor vs Tensor (int32 vs float32)
        ys = torch.tensor(ys)
        num = xs.nelement() # number of elements in the tensor
        print(f'number of elements in the tensor: {num}')
        return xs, ys, num

    def neural_network(self, xs, ys, num):
        for k in range(100):
            # forward pass
            xenc = F.one_hot(xs, num_classes=27).float() # input to nn should be float
            logits = xenc @ self.W # log counts [remember that W cn be made 0, which will get us more uniform distribution, always try keeping the weights small i.e. close to 0, known as smoothing]
            counts = logits.exp() # equal to N
            probs = counts / counts.sum(1, keepdim=True) # normalize the probabilities
            loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(self.W**2).mean() # average negative log likelihood loss, add regularization to the loss, basically make the weights small
            print(f'loss at iteration {k+1}: {loss.item():.4f}')

            # backward pass
            self.W.grad = None # clear the gradients
            loss.backward() # calculate the gradients

            # update the weights
            self.W.data += -50 * self.W.grad # learning rate of 0.1

    def generate_name(self, itos):
        g = torch.Generator().manual_seed(2147483647) # random seed, generator is used to be deter

        for i in range(5):
            out = []
            ix = 0 # start with start symbol

            while True:
                # p = probs[ix] older way of doing it
                # lets use neural network to predict the next character
                xenc = F.one_hot(torch.tensor(ix), num_classes=27).float()
                logits = xenc @ self.W
                counts = logits.exp()
                p = counts / counts.sum(0, keepdim=True)

                ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
                out.append(itos[ix])
                if ix == 0:
                    break
            print(''.join(out))

def main():
    bigram = Bigram()
    words = bigram.load_data()
    stoi, itos = bigram.create_vocab(words)
    xs, ys, num = bigram.create_dataset_bigram(words, stoi, itos)
    bigram.neural_network(xs, ys, num)
    bigram.generate_name(itos)

if __name__ == "__main__":
    main()
