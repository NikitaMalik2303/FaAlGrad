
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
n_iter = 10 

inner_args = {
    'lr': 0.005     # Learning rate for inner-loop optimization
}

meta_args = {
    'meta_train': True,     
    'lr': 0.001,            
    'num_iter': 1          
}

