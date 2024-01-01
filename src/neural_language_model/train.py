from network import Network
from dataset import Dataset

class Train:
    def __init__(self, block_size, emb_size, number_inputs, total_input_size, num_neurons):
        self.block_size = block_size
        self.emb_size = emb_size
        self.number_inputs = number_inputs
        self.total_input_size = total_input_size
        self.num_neurons = num_neurons

    def train_loop(self):
        # Call the build_dataset function from dataset.py
        dataset = Dataset('src/bigram/names.txt', self.block_size)
        Xtr, Ytr, Xdev, Ydev, Xte, Yte = dataset.train_dev_test_split()
        itos = dataset.get_itos()
        vocab_size = dataset.get_vocab_size()

        # Call the train function from network.py
        network = Network(vocab_size=vocab_size, emb_size=self.emb_size, input_size=self.number_inputs * self.emb_size, num_neurons=self.num_neurons)
        trainable_params_size = network.get_total_trainable_params()
        network.requires_grads()
        lrs = network.train(Xtr, Ytr)
        network.plot_loss(trainable_params_size)

        # Call the evaluate_training_set function from network.py
        tr_loss = network.evaluate_training_set(Xtr, Ytr)
        print(f'training loss: {tr_loss}')
        dev_loss = network.evaluate_dev_set(Xdev, Ydev)
        print(f'development loss: {dev_loss}')

        # Call the generate function from network.py
        names = network.generate(self.block_size, itos)
        print(f'generated names: {names}')

        return vocab_size, trainable_params_size, tr_loss, dev_loss