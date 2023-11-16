from torch.utils.data import DataLoader
from config import *
from datasets.dataset import *
from statistic.collect_stat import CollectStatistics
import numpy as np
import random
from models.get_model import get_model

if device.type != 'cpu':
    torch.cuda.set_device(device)

random.seed(seed)
np.random.seed(seed)  # numpy
torch.manual_seed(seed)  # cpu
torch.cuda.manual_seed(seed)  # gpu
torch.backends.cudnn.deterministic = True  # cudnn

data_train, data_test = load_data(dataset, dataset_file_path, model_name)
data_train_loader = DataLoader(data_train, batch_size=batch_size_eval, shuffle=True, num_workers=0)
data_test_loader = DataLoader(data_test, batch_size=batch_size_eval, num_workers=0)
dataiter = iter(data_train_loader)
img_size, channels, num_classes = get_data_info(dataset, model_name)
stat = CollectStatistics(results_file_name=results_file_name)

model = get_model(model_name, dataset, rand_seed=seed, step_size=step_size, device=device)


def sample_minibatch(dataiter):
    try:
        images, labels = dataiter.next()
        if len(images) < batch_size_train:
            dataiter = iter(data_train_loader)
            images, labels = dataiter.next()
    except StopIteration:
        dataiter = iter(data_train_loader)
        images, labels = dataiter.next()

    return images, labels


num_iter = 0
last_output = 0
while True:
    model.model.train()
    images, labels = sample_minibatch(dataiter)
    images, labels = images.to(device), labels.to(device)
    model.optimizer.zero_grad()
    output = model.model(images)
    loss = model.loss_fn(output, labels)
    loss.backward()
    model.optimizer.step()

    num_iter = num_iter + 1

    if num_iter - last_output >= num_iter_one_output:
        stat.collect_stat_global(num_iter, model, data_train_loader, data_test_loader)
        last_output = num_iter

    if num_iter >= max_iter:
        break

del model
torch.cuda.empty_cache()

