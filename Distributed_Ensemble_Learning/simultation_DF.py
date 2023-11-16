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
from util.voting import majority_voting, q_statistics_v2 ,mse_calculation
import copy

random.seed(seed)
np.random.seed(seed)  # numpy
torch.manual_seed(seed)  # cpu
torch.cuda.manual_seed(seed)  # gpu
torch.backends.cudnn.deterministic = True  # cudnn


num_of_learners=[10,20,30,40,50]
learner_list = []
num_samples_bagging = 10000
breakpoints = [5000]

data_train, data_test = load_data(dataset, dataset_file_path, model_name)
data_train_loader = DataLoader(data_train, batch_size=batch_size_eval, num_workers=0)
data_test_loader = DataLoader(data_test, batch_size=batch_size_eval, num_workers=0)
img_size, channels, num_classes = get_data_info(dataset, model_name)

comments = dataset + "_" + model_name + "_lr" + str(step_size) + "_n" + str(num_of_base_learners) + "_s" + str(num_samples_bagging)
results_file_name = os.path.join(results_file_path, 'rst_' + comments + '_breakpoints_'+str(breakpoints[0])+'.csv')



for num in num_of_learners:

    results_file_name_diversity = os.path.join(results_file_path, 'rst_' + dataset + "_" + model_name + "_lr" + str(step_size) + "_s" + str(num_samples_bagging) + '_breakpoints_'+str(breakpoints[0])+'_DF.csv')
    record_used_sample=[]
    sample_indices = [i for i in range(len(data_train))]
    
    # initialize record_used_sample
    for i in range(len(data_train)):
        record_used_sample.append([0]*(num))
    
    # label the samples used to train learner i
    for i in range(num): 
        indices = random.choices(sample_indices, k=num_samples_bagging)  # random sample with replacement
        for j in indices:
            record_used_sample[j][i]=1
    # load model
    for n in range(num):
        learner_name=os.path.join(os.path.dirname(__file__)+'/model_records/'+'rst_' + comments + '_breakpoints_'+str(breakpoints[0]))+'_'+str(n)+'.pt'
        learner_list.append(torch.load(learner_name))

    # Diversity
    q = q_statistics_v2(record_used_sample,num,learner_list,data_train_loader,device)
    print("Diversity measured by DF (" + str(num)+ "): " + str(q) + '\n')

    with open(results_file_name_diversity, 'a') as f:
        f.write("Diversity with " +str(num) + " base leaners: " + str(q) + '\n')
        f.close()
    
    # restart
    learner_list=[]
    record_used_sample=[]




