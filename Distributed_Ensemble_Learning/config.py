import os
import torch

use_gpu = True
use_gpu = use_gpu and torch.cuda.is_available()
device = torch.device('cuda:0') if use_gpu else torch.device('cpu')

# dataset = 'MNIST'
# model_name = 'ModelCNNMnist'  # 'ModelCNNMnist', 'LeNet5', 'LeNet5Half'

# dataset = 'cifar10'
# model_name = 'ResNet18'  # 'ModelCNNCifar10', 'ResNet18', 'ResNet34'

# dataset = 'cifar100'
# model_name = 'ResNet34'  # 'ResNet18', 'ResNet34'

dataset = 'SVHN'
model_name = 'WResNet10-1'  # 'WResNet10-1', 'WResNet10-2', 'WResNet16-1'  #'WResNet40-2'


optimizer = 'Adam'
step_size = 0.001  # learning rate of clients, Adam optimizer
batch_size_train = 32
batch_size_eval = 32 #512
max_iter = 9000 #100  # Maximum number of iterations to run
seed = 1
num_iter_one_output = 20
num_of_base_learners = 50 #3
dataset_file_path = os.path.join(os.path.dirname(__file__), 'dataset_data_files')
results_file_path = os.path.join(os.path.dirname(__file__), 'results/')



""" num_samples_bagging = 2000
comments = dataset + "_" + model_name + "_lr" + str(step_size) + "_n" + str(num_of_base_learners) + "_s" + str(num_samples_bagging)
results_file_name = os.path.join(results_file_path, 'rst_' + comments + '.csv')
results_file_name_diversity = os.path.join(results_file_path, 'rst_' + comments + 'div.csv')

 """