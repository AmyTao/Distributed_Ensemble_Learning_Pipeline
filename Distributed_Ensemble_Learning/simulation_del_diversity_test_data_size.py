# Part of this code is inspired by https://github.com/IBM/adaptive-federated-learning
import certifi
import ssl
from torch.utils.data import Dataset,DataLoader
from config import *
from datasets.dataset import *
from models.get_model import get_model, adjust_learning_rate_cifar10
from statistic.collect_stat import CollectStatisticsDEL
import numpy as np
import random
from util.sampling import split_data
from torch.utils.data.sampler import RandomSampler
from util.utils import DatasetSplit
from util.voting import majority_voting, q_statistics,mse_calculation
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

data_samples = [10000]

#stat = CollectStatisticsDEL(results_file_name=results_file_name)

for d in range(len(data_samples)):

    # Create distributed dataloader
    # -------------bagging
    sample_indices = [i for i in range(len(data_train))]
    train_loader_list = []
    dataiter_list = []
    for n in range(num_of_base_learners):
        indices = random.choices(sample_indices, k=data_samples[d])  # random sample with replacement
        train_loader_list.append(DataLoader(DatasetSplit(data_train, indices), batch_size=batch_size_train, shuffle=True))
        dataiter_list.append(iter(train_loader_list[n]))

    # # ---------------niid
    # dict_users = split_data(dataset, data_train, num_of_base_learners, 'dirichlet', num_sample=data_samples[d], num_classes=num_classes, alpha=1)
    # train_loader_list = []
    # dataiter_list = []
    # for n in range(num_of_base_learners):
    #     local_Dataset = DatasetSplit(data_train, dict_users[n])
    #     train_loader_list.append(DataLoader(local_Dataset, batch_size=batch_size_train, shuffle=True))
    #     dataiter_list.append(iter(train_loader_list[n]))


    def sample_minibatch(n):
        try:
            images, labels = next(dataiter_list[n])
            if len(images) < batch_size_train:
                dataiter_list[n] = iter(train_loader_list[n])
                images, labels = next(dataiter_list[n])
        except StopIteration:
            dataiter_list[n] = iter(train_loader_list[n])
            images, labels = next(dataiter_list[n])

        return images, labels


    breakpoints = [9000]
    model_list = [[] for i in range(len(breakpoints))]
    num_samples_bagging = data_samples[d]
    comments = dataset + "_" + model_name + "_lr" + str(step_size) + "_n" + str(num_of_base_learners) + "_s" + str(num_samples_bagging)
    results_file_name = os.path.join(results_file_path, 'rst_' + comments + '_breakpoints_'+str(breakpoints[0])+'.csv')
    results_file_name_diversity = os.path.join(results_file_path, 'rst_' + comments + '_breakpoints_'+str(breakpoints[0])+'_div.csv')
    stat = CollectStatisticsDEL(results_file_name=results_file_name)

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
                stat.collect_stat_global(n, num_iter, model, train_loader_list[n], data_test_loader)
                last_output = num_iter
            for i in range(len(breakpoints)):
                if num_iter == breakpoints[i]:
                    model_list[0].append(copy.deepcopy(model.model))
                    model_save=os.path.join(os.path.dirname(__file__)+'/model_records/'+'rst_' + comments + '_breakpoints_'+str(breakpoints[0]))+'_'+str(n)+'.pt'
                    print(model_save)
                    torch.save(model.model,model_save)

            if num_iter >= max_iter:
               # model_list[0].append(copy.deepcopy(model.model))
                break

    # for i in range(len(breakpoints)):
    #     # Majority voting
    #     acc,accuracy_list = majority_voting(num_of_base_learners, model_list[i], data_test_loader, num_classes, device)
    #     print("Ensemble test accuracy: ", acc)
    #     print("Learners' accuracy list: ", accuracy_list)
    #     stat.write_voting_accuracy(acc,accuracy_list)

    #     # Diversity
    #     q, q_avg = q_statistics(num_of_base_learners, model_list[i], data_test_loader, device)
    #     mse=mse_calculation(q,accuracy_list,num_of_base_learners)
    #     with open(results_file_name_diversity, 'a') as f:
    #         f.write(str(q) + '\n')
    #         f.write("Average diversity: " + str(q_avg.item()) + '\n')
    #         for j in range(len(mse)):
    #             f.write("MSE score_ " + str(j+2)+"_:"+str(mse[j]) + '\n')
    #             print(mse[j])
    #         f.close()
    #     print(q_avg)

