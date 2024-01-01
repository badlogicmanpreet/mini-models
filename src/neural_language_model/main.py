from dataset import Dataset
from network import Network
from train import Train
from utils import Utils
import matplotlib.pyplot as plt
import numpy as np

def main():
    model_data = []

    def run(block_size, emb_size, number_inputs, num_neurons):
        total_input_size = emb_size * number_inputs
        train = Train(block_size, emb_size, number_inputs, total_input_size, num_neurons)
        vocab_size, trainable_params_size, tr_loss, dev_loss = train.train_loop()
        model_data.append([vocab_size, emb_size, number_inputs, total_input_size, num_neurons, trainable_params_size, tr_loss, dev_loss])

    '''run the model with different parameters'''
    run(block_size=3, emb_size=2, number_inputs=3, num_neurons=100)
    run(block_size=3, emb_size=10, number_inputs=3, num_neurons=200)
    run(block_size=4, emb_size=10, number_inputs=4, num_neurons=200)
    run(block_size=3, emb_size=15, number_inputs=3, num_neurons=300)
    run(block_size=3, emb_size=20, number_inputs=3, num_neurons=300)
    run(block_size=3, emb_size=30, number_inputs=3, num_neurons=300)
    run(block_size=3, emb_size=25, number_inputs=3, num_neurons=400)
    run(block_size=3, emb_size=30, number_inputs=3, num_neurons=300)
    run(block_size=3, emb_size=30, number_inputs=3, num_neurons=400)
    run(block_size=3, emb_size=40, number_inputs=3, num_neurons=500) # -> good model -> 75107 trainable parameters
        
    run(block_size=3, emb_size=50, number_inputs=3, num_neurons=500)
    run(block_size=3, emb_size=60, number_inputs=3, num_neurons=500)
    run(block_size=3, emb_size=70, number_inputs=3, num_neurons=500)
    run(block_size=3, emb_size=40, number_inputs=3, num_neurons=1000)
    run(block_size=3, emb_size=50, number_inputs=3, num_neurons=1000)
    run(block_size=3, emb_size=60, number_inputs=3, num_neurons=1000)
    run(block_size=3, emb_size=70, number_inputs=3, num_neurons=1000)
    run(block_size=3, emb_size=100, number_inputs=3, num_neurons=1000) 
    run(block_size=4, emb_size=150, number_inputs=4, num_neurons=1000) # -> good model -> 632077 trainable parameters

    run(block_size=3, emb_size=200, number_inputs=3, num_neurons=2000)
    run(block_size=4, emb_size=200, number_inputs=4, num_neurons=2000)
    run(block_size=4, emb_size=200, number_inputs=4, num_neurons=3000) # -> good model -> 2,489,427 trainable parameters -> 2.5 million trainable parameters

    # Call the plot_table function from utils.py
    utils = Utils()
    utils.plot_table(model_data=np.transpose(model_data))
    

if __name__ == "__main__":
    main()