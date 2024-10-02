
batch_size = 64
learning_rate = 0.001  # outer loop learning rate
dropout_prob = 0


# compass : 8
# adult : 27
# communities and crime : 98

input_dim = 8
hidden_dim = [8, 16]
output_dim = 1

n_step = 1
n_iter = 10 #number of innner loop iterations 

inner_args = {
    'lr': 0.005     # Learning rate for inner-loop optimization
}

meta_args = {
    'meta_train': True,     # Whether to train the model in meta-learning mode
    'lr': 0.001,             # Learning rate for the meta-update step
    'num_iter': 1           # num
}

