import torch
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Network:

    def __init__(self, vocab_size, emb_size, input_size, num_neurons) -> None:
        self.g = torch.Generator().manual_seed(2147483647) # random seed
        #C = torch.randn((27, 2), generator=g) # embedding table
        #W1 = torch.randn((6, 300), generator=g) # first layer weights
        # self.C = torch.randn((27, 10), generator=self.g) # embedding table, increased the embedding size to 10
        # self.W1 = torch.randn((30, 200), generator=self.g) # first layer weights, increased the input size to 30 (10 * 3)
        # self.b1 = torch.randn((200), generator=self.g) # first layer biases
        # self.W2 = torch.randn((200, 27), generator=self.g) # second layer weights
        # self.b2 = torch.randn((27), generator=self.g) # second layer biases
        self.embedding_size = emb_size
        self.input_size = input_size
        self.C = torch.randn((vocab_size, emb_size), generator=self.g) # embedding table, increased the embedding size to 10
        self.W1 = torch.randn((input_size, num_neurons), generator=self.g) # first layer weights, increased the input size to 30 (10 * 3)
        self.b1 = torch.randn((num_neurons), generator=self.g) # first layer biases
        self.W2 = torch.randn((num_neurons, vocab_size), generator=self.g) # second layer weights
        self.b2 = torch.randn((vocab_size), generator=self.g) # second layer biases
        self.parameters = [self.C, self.W1, self.b1, self.W2, self.b2] # list of parameters
        self.lri = []
        self.lossi = []
        self.stepi = []

    def get_total_trainable_params(self):
        return sum(p.nelement() for p in self.parameters) # total no. of parameters
    
    def requires_grads(self):
        for p in self.parameters:
            p.requires_grad = True

    def get_learning_rate(self):
        # lets calculate a good learning rate
        # learning rate exponent, between [.001, 1]
        lre = torch.linspace(-3, 0, 1000) # -3 because .001 is 10^-3, 0 because 1 is 10^0
        lrs = 10 ** lre
        return lrs

    def train(self, Xtr, Ytr):
        for i in range(200000):
            # lets create a mini batch, select few indexes in 228000 examples
            ix = torch.randint(0, Xtr.shape[0], (32,))

            # forward pass
            emb = self.C[Xtr[ix]] # embedding of the input X
            #h = torch.tanh(emb.view(-1, 6) @ W1 + b1) # output of the first layer
            h = torch.tanh(emb.view(-1, self.input_size) @ self.W1 + self.b1) # output of the first layer, increased the input size to 30 (10 * 3)
            logits = h @ self.W2 + self.b2 # output of the second layer (32, 27)
            # counts = logits.exp()
            # prob = counts / counts.sum(dim=1, keepdim=True) # softmax
            # loss = -prob[torch.arange(32), Y].log().mean() # negative log likelihood

            loss = F.cross_entropy(logits, Ytr[ix]) # cross entropy, pytorch does this very efficiently (fused kernel), very easy to backpropagate

            # backward pass
            for p in self.parameters:
                p.grad = None # zero out all the gradients
            loss.backward() # backpropagate

            # change the parameters
        #    lr = lrs[i]
            lr = 0.1 if i < 100000 else 0.01 # at some point we should start decaying the learning rate
            for p in self.parameters:
                p.data += -lr * p.grad # gradient descent

            # track the loss and learning rate
        #    lri.append(lr)
            self.stepi.append(i)
            self.lossi.append(loss.log10().item()) # log10 of the loss, because when you plot the loss it appears as a hockey stick, we squash it using log10

    def plot_loss(self, trainable_params_size):
        plt.plot(self.stepi, self.lossi)
        plt.savefig(f'src/neural_language_model/params_loss_plots/{trainable_params_size}_loss.png') # save the plot as a png file

    def evaluate_training_set(self, Xtr, Ytr):
        # evaluate on the training set
        emb = self.C[Xtr] # embedding of the input X
        h = torch.tanh(emb.view(-1, self.input_size) @ self.W1 + self.b1) # output of the first layer
        logits = h @ self.W2 + self.b2 # output of the second layer (32, 27)
        loss = F.cross_entropy(logits, Ytr) 
        return loss.item()

    def evaluate_dev_set(self, Xdev, Ydev):
        # evaluate on the dev set
        emb = self.C[Xdev] # embedding of the input X
        h = torch.tanh(emb.view(-1, self.input_size) @ self.W1 + self.b1) # output of the first layer
        logits = h @ self.W2 + self.b2 # output of the second layer (32, 27)
        loss = F.cross_entropy(logits, Ydev)
        return loss.item()

    def evaluate_test_set(self, Xte, Yte):
        # evaluate on the dev set
        emb = self.C[Xte] # embedding of the input X
        h = torch.tanh(emb.view(-1, self.input_size) @ self.W1 + self.b1) # output of the first layer
        logits = h @ self.W2 + self.b2 # output of the second layer (32, 27)
        loss = F.cross_entropy(logits, Yte)
        return loss.item()

    def generate(self, block_size, itos):
        # sample from the model
        g = torch.Generator().manual_seed(2147483647 + 10) # random seed

        output = []
        for _ in range(10):
            out = []
            context = [0] * block_size # start with the context of length 3
            while True:
                emb = self.C[torch.tensor(context)] # embedding of the input X (1, block_size, 2)
                h = torch.tanh(emb.view(1, -1) @ self.W1 + self.b1)
                logits = h @ self.W2 + self.b2 # output of the second layer (1, 27)
                probs = F.softmax(logits, dim=1)
                ix = torch.multinomial(probs, num_samples=1, generator=g).item() # sample from the distribution
                context = context[1:] + [ix]
                out.append(ix)
                if ix == 0:
                    break
                
            output.append(''.join(itos[i] for i in out)) # convert the indices to characters and return the string

        return output