# Part of this code is inspired by https://github.com/IBM/adaptive-federated-learning
import certifi
import ssl
from torch.utils.data import Dataset,DataLoader
from config import *
from datasets.dataset import *
from models.get_model import get_model, adjust_learning_rate_cifar10
from models.model import model_in_pool, model_pool
from statistic.collect_stat import CollectStatisticsDEL
import numpy as np
import random
from torch.utils.data.sampler import RandomSampler
from util.utils import DatasetSplit
from util.voting import majority_voting, q_statistics
import copy

random.seed(seed)
np.random.seed(seed)  # numpy
torch.manual_seed(seed)  # cpu
torch.cuda.manual_seed(seed)  # gpu
torch.backends.cudnn.deterministic = True  # cudnn

data_train, data_test = load_data(dataset, dataset_file_path, model_name)
data_train_loader = DataLoader(data_train, batch_size=batch_size_eval, shuffle=True, num_workers=0)
data_test_loader = DataLoader(data_test, batch_size=batch_size_eval, num_workers=0)
img_size, channels, num_classes = get_data_info(dataset, model_name)

# Create distributed dataloader
sample_indices = [i for i in range(len(data_train))]
train_loader_list = []
dataiter_list = []
for n in range(num_of_base_learners):
    indices = random.choices(sample_indices, k=num_samples_bagging)  # random sample with replacement
    train_loader_list.append(DataLoader(DatasetSplit(data_train, indices), batch_size=batch_size_train, shuffle=True))
    dataiter_list.append(iter(train_loader_list[n]))


def sample_minibatch(n):
    try:
        images, labels = dataiter_list[n].next()
        if len(images) < batch_size_train:
            dataiter_list[n] = iter(train_loader_list[n])
            images, labels = dataiter_list[n].next()
    except StopIteration:
        dataiter_list[n] = iter(train_loader_list[n])
        images, labels = dataiter_list[n].next()

    return images, labels


stat = CollectStatisticsDEL(results_file_name=results_file_name)

model_list = []

# Train models
for n in range(num_of_base_learners):
    model = get_model(model_name, dataset, rand_seed=seed, step_size=step_size, device=device)
    num_iter = 0
    last_output = 0
    while True:
        if dataset == 'cifar10':
            adjust_learning_rate_cifar10(model.optimizer, num_iter, step_size)
        images, labels = sample_minibatch(n)
        images, labels = images.to(device), labels.to(device)
        model.optimizer.zero_grad()
        output = model.model(images)
        loss = model.loss_fn(output, labels)
        loss.backward()
        model.optimizer.step()
        num_iter += 1
        if num_iter - last_output >= num_iter_one_output:
            stat.collect_stat_global(n, num_iter, model, data_train_loader, data_test_loader)
            last_output = num_iter
        if num_iter >= max_iter:
            model_list.append(copy.deepcopy(model.model))
            break

# Majority voting
acc = majority_voting(num_of_base_learners, model_list, data_test_loader, num_classes, device)
print("Ensemble test accuracy: ", acc)
stat.write_voting_accuracy(acc)

# Diversity
q, q_avg = q_statistics(num_of_base_learners, model_list, data_test_loader, num_classes, device)
with open(results_file_name_diversity, 'a') as f:
    f.write(str(q) + '\n')
    f.write("Average diversity: " + str(q_avg.item()) + '\n')
    f.close()
print(q_avg)

