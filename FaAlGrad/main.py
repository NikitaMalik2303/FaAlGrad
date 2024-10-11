import random
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from config import *
from maml import *

from fairlearn.metrics import MetricFrame
import torch.utils.tensorboard as tb
import shutil
import shap
import matplotlib.pyplot as plt
import os
from dataloader import *
from metrics import *
from mlp import *


def seed_everything(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
seed = 102
seed_everything(seed)


def write_to_logs(path, string, mode="a"):
    with open(path, mode) as f:
        f.write(string + "\n")

def predict_with_pytorch_model(input_data):
    input_tensor = torch.tensor(input_data, dtype=torch.float32)

    model.eval()

    with torch.no_grad():
        predictions = model(input_tensor).numpy() 
        
    return predictions
    
def train(model, X_support, X_query, Y_support, Y_query, A_support, A_query, optimizer, criterion):
 
    Y_support = Y_support.view(Y_support.size(0), -1)
    Y_query = Y_query.view(Y_query.size(0), -1)
    
    X = torch.cat((X_support, X_query), dim=0)
    Y = torch.cat((Y_support, Y_query), dim=0)
    Y = Y.squeeze()
    A = torch.cat((A_support, A_query), dim=0)
        
    model.train()
    train_loss = 0.0

    optimizer.zero_grad()
    
    all_logits = maml.forward( X_support, X_query, Y_support, inner_args, meta_args)
    loss = criterion(all_logits.reshape(-1, model.output_dim), Y_query.view(-1, 1))

    loss.backward()
    optimizer.step()
    
    model.eval()
    
    params = OrderedDict(model.named_parameters())
    logits = model.forward(X, params)
    logits = logits.squeeze()
    loss = criterion(logits, Y)
    
    Y_pred = (torch.sigmoid(logits) > 0.5 ).float()
    
    

    Y_pred = Y_pred.squeeze()
    # num = torch.sum(torch.sigmoid(all_logits)>0.5)
    Y = Y.squeeze()
    
    y_true_p_ones = torch.sum(torch.logical_and(A==1, Y==1))
    y_true_p_zeroes = torch.sum(torch.logical_and(A==1, Y==0))
    
    y_true_un_ones = torch.sum(torch.logical_and(A==0, Y==1))
    y_true_un_zeroes = torch.sum(torch.logical_and(A==0, Y==0))
    
    y_pred_p_ones = torch.sum(torch.logical_and(A==1, Y_pred==1)) 
    y_pred_p_zeroes = torch.sum(torch.logical_and(A==1, Y_pred==0)) 
    
    y_pred_un_ones = torch.sum(torch.logical_and(A==0, Y_pred==1)) 
    y_pred_un_zeroes= torch.sum( torch.logical_and(A==0, Y_pred==0)) 

    
    train_accuracy = (Y_pred == Y.view(-1, 1)).sum().item() / len(Y)
    dp_diff = demographic_parity_diff(Y, Y_pred.detach().numpy(), A.detach().numpy())
    dp_ratio = demographic_parity_ratio_(Y, Y_pred.detach().numpy(), A.detach().numpy())
    eod_diff= equal_odds_diff(Y, Y_pred.detach().numpy(), A.detach().numpy())
    eod_ratio = equal_odds_ratio(Y, Y_pred.detach().numpy(), A.detach().numpy())
    
    train_loss = loss.item()
    
    writer.add_scalar('Loss/Train', train_loss, epoch)
    writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
    writer.add_scalar('Demographic Parity/Train', dp_ratio, epoch)
    writer.add_scalar('Equal Odds/Train', eod_ratio, epoch)
    
    
    writer.flush()

    return train_loss, train_accuracy, dp_diff, dp_ratio, eod_diff, eod_ratio


def test(model, X, Y, A, criterion):
    
    model.eval()
    test_loss = 0.0    
    
    params = OrderedDict(model.named_parameters())
    logits = model.forward(X, params)
    logits = logits.squeeze()
    loss = criterion(logits, Y)
    test_loss = loss.item()
    Y_pred = (torch.sigmoid(logits)>0.5).float()
    test_accuracy = (Y_pred == Y).sum().item()/len(Y)
    
    X = X.squeeze()
    Y = Y.squeeze()
    A = A.squeeze()
    Y_pred = Y_pred.squeeze()
    
    y_true_p_ones = torch.sum(torch.logical_and(A==1, Y==1))
    y_true_p_zeroes = torch.sum(torch.logical_and(A==1, Y==0))
    
    y_true_un_ones = torch.sum(torch.logical_and(A==0, Y==1))
    y_true_un_zeroes = torch.sum(torch.logical_and(A==0, Y==0))
    
    y_pred_p_ones = torch.sum(torch.logical_and(A==1, Y_pred==1)) 
    y_pred_p_zeroes = torch.sum(torch.logical_and(A==1, Y_pred==0)) 
    
    y_pred_un_ones = torch.sum(torch.logical_and(A==0, Y_pred==1)) 
    y_pred_un_zeroes= torch.sum(torch.logical_and(A==0, Y_pred==0)) 

    
    dp_diff = demographic_parity_diff(Y, Y_pred.numpy(), A.numpy())
    dp_ratio = demographic_parity_ratio_(Y, Y_pred.numpy(), A.numpy())
    eod_diff = equal_odds_diff(Y, Y_pred.numpy(), A.numpy())
    eod_ratio = equal_odds_ratio(Y, Y_pred.numpy(), A.numpy())
    
    dp_diff = demographic_parity_diff(Y, Y_pred.numpy(), A.numpy())
    dp_ratio = demographic_parity_ratio_(Y, Y_pred.numpy(), A.numpy())
    eod_diff = equal_odds_diff(Y, Y_pred.numpy(), A.numpy())
    eod_ratio = equal_odds_ratio(Y, Y_pred.numpy(), A.numpy())
    
    
    
    writer.add_scalar('Loss/Val', test_loss, epoch)
    writer.add_scalar('Accuracy/Val', test_accuracy, epoch)
    writer.add_scalar('Demographic Parity/Val', dp_ratio, epoch)
    writer.add_scalar('Equal Odds/Val', eod_ratio, epoch)
    
    writer.flush()

    return test_loss, test_accuracy, dp_diff, dp_ratio, eod_diff, eod_ratio

        


#%%

log_dir = './logs'

if os.path.exists(log_dir) and os.path.isdir(log_dir):
    shutil.rmtree(log_dir)

writer = tb.SummaryWriter(log_dir)

# Load and preprocess your dataset
X, Y, sens =  get_and_preprocess_compas_data()

X_train_temp, X_test, Y_train_temp, Y_test = train_test_split(X, Y, test_size=0.2 )
X_train, X_val, Y_train, Y_val = train_test_split(X_train_temp, Y_train_temp, test_size=0.125)


X_train = X_train.astype(int)
X_val = X_val.astype(int)
X_test = X_test.astype(int)
X_train = torch.tensor(X_train.values, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32)

X_val = torch.tensor(X_val.values, dtype=torch.float32)
Y_val = torch.tensor(Y_val, dtype=torch.float32)

X_test = torch.tensor(X_test.values, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.float32)


X_support, X_query, Y_support, Y_query = train_test_split(X_train, Y_train, test_size = 0.8)


# uncomment the part to get the below configuration and comment other configs


#  X_s: protected group 0's + unprotected 1's and X_q: unprotected group 0's + protected 1's

# =============================================================================
#
# X_protected = X_train[X_train[:, 7]==1]
# X_unprotected = X_train[X_train[:,7]==0]
# 
# Y_protected = Y_train[X_train[:,7]==1]
# Y_unprotected = Y_train[X_train[:, 7]==0]
# 
# X_support = torch.cat((X_protected[Y_protected == 0], X_unprotected[Y_unprotected == 1]), dim=0)
# X_query = torch.cat((X_protected[Y_protected == 1], X_unprotected[Y_unprotected== 0]), dim =0)
# 
# Y_support = torch.cat((Y_protected[Y_protected == 0], Y_unprotected[Y_unprotected == 1]), dim=0)
# Y_query = torch.cat((Y_protected[Y_protected == 1], Y_unprotected[Y_unprotected== 0]), dim =0)
# 
# =============================================================================

 

# X_support : protected group , X_query : unprotected group

X_support_p, Y_support_p = X_support[X_support[:,7]==1], Y_support[X_support[:,7]==1]
X_support_u, Y_support_u = X_support[X_support[:,7]==0], Y_support[X_support[:,7]==0]

X_query_p, Y_query_p = X_query[X_query[:, 7]==1], Y_query[X_query[:, 7]==1]
X_query_u, Y_query_u = X_query[X_query[:, 7]==0], Y_query[X_query[:, 7]==0]

X_support = torch.cat((X_support_p, X_query_p), dim=0)
X_query = torch.cat((X_support_u, X_query_u), dim=0)

Y_support = torch.cat((Y_support_p, Y_query_p), dim=0)
Y_query = torch.cat((Y_support_u, Y_query_u), dim=0)


# split case swapping

# =============================================================================
# X_support_p, Y_support_p = X_support[X_support[:,7]==1], Y_support[X_support[:,7]==1]
# X_support_u, Y_support_u = X_support[X_support[:,7]==0], Y_support[X_support[:,7]==0]
# 
# X_query_p, Y_query_p = X_query[X_query[:, 7]==1], Y_query[X_query[:,7]==1]
# X_query_u, Y_query_u = X_query[X_query[:, 7]==0], Y_query[X_query[:,7]==0]
# 
# len_X_support_u = len(X_support_u)
# X_support = torch.cat((X_support_p, X_query_p[:len_X_support_u, :]), dim=0)
# Y_support = torch.cat((Y_support_p, Y_query_p[:len_X_support_u]), dim=0)
# 
# X_query = torch.cat((X_query_u, X_query_p[len_X_support_u:, :]), dim =0)
# X_query = torch.cat((X_query, X_support_u), dim =0)
# 
# Y_query = torch.cat((Y_query_u, Y_query_p[len_X_support_u:]), dim =0)
# Y_query = torch.cat((Y_query, Y_support_u), dim =0)
# 
# len_X_support_p = len(X_support_p)
# X_support = torch.cat((X_support_u, X_query_u[:len_X_support_p, :]), dim=0)
# Y_support = torch.cat((Y_support_u, Y_query_u[:len_X_support_p]), dim=0)
# 
# X_query = torch.cat((X_query_p, X_query_u[len_X_support_p:, :]), dim =0)
# X_query = torch.cat((X_query, X_support_p), dim =0)
# 
# Y_query = torch.cat((Y_query_p, Y_query_u[len_X_support_p:]), dim =0)
# Y_query = torch.cat((Y_query, Y_support_p), dim =0)
# =============================================================================



print(f"X_support : {X_support.shape}, 1's : {torch.sum(Y_support==1).item()}, protected class : {torch.sum(X_support[:,7]==1)}")
print(f"X_query : {X_query.shape}, 1's : {torch.sum(Y_query==1).item()}, Protected class : {torch.sum(X_query[:,7]==1)}")
print(f"X_val : {X_val.shape}, 1's : {torch.sum(Y_val==1).item()}, Protected Class : {torch.sum(X_val[:,7]==1)}")
print(f"X_test : {X_test.shape}, 1's : {torch.sum(Y_test==1).item()}, Protected Class : {torch.sum(X_test[:,7]==1)}")


A_train = X_train[:, 7]
A_support = X_support[:, 7]
A_query = X_query[:, 7]
A_val = X_val[:, 7]
A_test = X_test[:, 7]


X_val_p_1 = torch.sum(Y_val[X_val[:, 7]==1])
X_test_p_1 = torch.sum(Y_test[X_test[:, 7]==1])

X = X.squeeze()
Y = Y.squeeze()

y_true_p_ones = torch.sum(torch.logical_and(A_train==1, Y_train==1))
y_true_p_zeroes = torch.sum(torch.logical_and(A_train==1, Y_train==0))

y_true_un_ones = torch.sum(torch.logical_and(A_train==0, Y_train==1))
y_true_un_zeroes = torch.sum(torch.logical_and(A_train==0, Y_train==0))

# =============================================================================
# y_pred_p_ones = torch.sum(torch.logical_and(A==1, Y_pred==1)) 
# y_pred_p_zeroes = torch.sum(torch.logical_and(A==1, Y_pred==0)) 
# 
# y_pred_un_ones = torch.sum(torch.logical_and(A==0, Y_pred==1)) 
# y_pred_un_zeroes= torch.sum(torch.logical_and(A==0, Y_pred==0)) 
# =============================================================================

print("num ground positives protected ", y_true_p_ones)
print("num ground negaties protected ", y_true_p_zeroes)
print("num ground positives unprotected ", y_true_un_ones)
print("num ground negatives unprotected ", y_true_un_zeroes)

# =============================================================================
# print("num predicted positives protected ", y_pred_p_ones)
# print("num predicted negatives protected ", y_pred_p_zeroes)
# print("num predicted positives unprotected ", y_pred_un_ones)
# print("num predicted negatives unprotected ", y_pred_un_zeroes)
# =============================================================================


##### Model and Optimizer #####

model = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
maml = MAML(model)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()



num_epochs = 100


# %%

# MAML CODE


best_val_loss = float('inf')
current_patience = 0
max_patience = 10 

write_to_logs(f"./checkpoints/Compass/record_train.txt", "----train----", "w")
write_to_logs(f"./checkpoints/Compass/record_val.txt", "----validation----", "w")
write_to_logs(f"./checkpoints/Compass/record_test.txt", "----test----", "w") 


for epoch in range(num_epochs):
    
    train_loss, train_accuracy, train_dp_diff, train_dp_ratio, train_eod_diff, train_eod_ratio = train(model, X_support, X_query, Y_support, Y_query, A_support, A_query, optimizer, criterion)
    test_loss, test_accuracy, test_dp_diff, test_dp_ratio, test_eod_diff, test_eod_ratio = test(model, X_val, Y_val, A_val, criterion)

    
    print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Train Accuracy : {train_accuracy:.4f}, dp_diff : {train_dp_diff:.4f}, dp_ratio : {train_dp_ratio:.4f}, eod_Diff : {train_eod_diff:.4f}, eod_ratio : {train_eod_ratio:.4f}")
    print(f"Epoch {epoch}: Val Loss = {test_loss:.4f}, Val Accuracy : {test_accuracy:.4f}, dp_diff : {test_dp_diff:.4f}, dp_ratio : {test_dp_ratio:.4f}, eod_Diff : {test_eod_diff:.4f}, eod_ratio : {test_eod_ratio:.4f}")
    
    if test_loss < best_val_loss:
        best_val_loss = test_loss
        current_patience = 0
        torch.save(model.state_dict(), "best_model.pt") 
    else:
        current_patience += 1
        
    if current_patience >= max_patience:
        print("Early stopping: No improvement in validation loss for consecutive epochs.")
        break

model.load_state_dict(torch.load("best_model.pt"))

final_loss, final_accuracy, dp_diff, dp_ratio, eod_diff, eod_ratio = test(model, X_test, Y_test, A_test, criterion)


print(f"Test Loss = {final_loss:.4f}, Test Accuracy : {final_accuracy:.4f}, dp_diff : {dp_diff:.4f}, dp_ratio : {dp_ratio:.4f}, eod_Diff : {eod_diff:.4f}, eod_ratio : {eod_ratio:.4f}")
print("Training complete.")

explainer = shap.Explainer(predict_with_pytorch_model, X_train.numpy())
shap_values = explainer(X_test.numpy())

shap.summary_plot(shap_values, X_test, feature_names=X.columns, show=False, title = "shap graph for compass dataset using fairgrads")
plt.savefig("compass_maml.pdf", dpi=600, bbox_inches="tight")

writer.close()

