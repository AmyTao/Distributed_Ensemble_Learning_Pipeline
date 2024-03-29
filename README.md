# Ensemble Learning Piepline Implementation

## Project Structure
```
Distributed_Ensemble_Learning
├── config.py
├── dataset_data_files
│   ├── SVHN
│   ├── cifar-10-batches-py
│   │   ├── batches.meta
│   │   ├── data_batch_1
│   │   ├── data_batch_2
│   │   ├── data_batch_3
│   │   ├── data_batch_4
│   │   ├── data_batch_5
│   │   └── test_batch
│   └── cifar-100-python
│       ├── meta
│       ├── test
│       └── train
├── datasets
│   ├── __pycache__
│   ├── dataset.py
│   ├── dataset_enhanced.py
│   └── test.py
├── model_records
├── models
│   ├── __pycache__
│   ├── cnn_cifar10.py
│   ├── cnn_mnist.py
│   ├── get_model.py
│   ├── lenet.py
│   ├── model.py
│   ├── resnet.py
│   └── wresnet.py
├── read_div_file.py
├── results
├── simulation_CL.py
├── simulation_del.py
├── simulation_del_diversity_test_data_size.py
├── simulation_del_diversity_test_epoch.py
├── simultation_DF.py
├── statistic
│   ├── collect_stat.py
│   
└── util
    ├
    ├── draw_boundary 2.py
    ├── draw_boundary.py
    ├── language_utils 2.py
    ├── language_utils.py
    ├── min_norm_solvers 2.py
    ├── min_norm_solvers.py
    ├── sampling 2.py
    ├── sampling.py
    ├── utils 2.py
    ├── utils.py
    ├── voting 2.py
    └── voting.py
```

# 1.  Problem description

- Brief Intro to Distributed Ensemble Learning:

Distributed ensemble learning aims to use the “wisdom of crowd” to achieve better decision making. Masters, who aggregate the models in this problem, dispatch datasets to its workers and workers give their trained models back to the masters.  Base learners trained with different portion of dataset and with heterogeneous computing capabilities perform differently in the final test set. Aggregating the decisions made by multiple base learners having different perspectives can compensate the errors of the final model.

- Project motivation
    1. The importance of incentives giving to the workers
    
    The project focuses on finding ways to incentivize worker participation in distributed EL. The incentives of masters giving to the workers is important to the accuracy of base learners received by the masters, because workers are self-interested and aim to maximize its gain.  To solve this problem, we take both worker’s gain and master’s gain into consideration:
    
    $Gain_w = incentive - computation Cost-communicationCost$
    
    $Gain_m=qualityOfEmsemble - costsToIncentive-costCommunication$
    
    1. Modeling on the factors above
        - incentive
        - computationCost
        - communicationCost
    2. Estimate worker payoff and server cost
        
        The goal of the reward scheme is to maximize worker’s payoff and minimize server’s cost
        
- My contribution to the project
    
    I involve in building the coding framework in computing the cost of master.
    
    $Cost_M=ensembleModelLoss(error)+totalIncentives$
    
    To minimize the cost of master, we want to figure out what relates with the ensemble model loss.
    
    Two assumptions are:
    
    1. The dependence of base learners
    2. The heterogeneity of base learners (Diversity in the following part of docs)
    
    I build the analyzing pipeline to test how base learners will be different by altering the number of base learners, the size of training sets, and epoch time. The testing is conducted on several different dataset and training models.
    

# 2. Methodology

- Dataset
    
    To simplify the problem, we work on the classification problem on several datasets, including:
    
    - MNIST
    - cifar10/100
    - SVHN
- Model used by workers
    - ModelCNNMnist
    - LeNet
    - ResNet
- Number of base learners
- Ensemble Learning Methods
    
    To simplify the problem, We use Bagging as the ensemble method in this problem. Each learner is trained using data sampled from the original data set and use majority voting for the final decision aggregation.
    

# 3. Coding Framework

*Part of the code is inspired by https://github.com/IBM/adaptive-federated-learning

1. The pipeline description
    
    Step 1: Prepare the training sets and testing sets. Assigning different sampling of datasets to workers.
    
    - Set the seed of random number generator for numpy and pyTorch
    - Choose dataset and models. Load the data for training and testing using pytorch.DataLoader, set the batch_size, and do shuffling for every sampling.
    - Label the samples used to train base learner i
    
    Step 2: Train and preserve models using sampled dataset. During the training process, record the model training logs including learnerID, the number of epoch, loss_value, accuracy on training set and accuracy on testing set in a csv file.
    
    Step 3: Use majority_voting to calculate the prediction accuracy on test dataset and diversity measured among the selected base learners. The diversity calculation is mentioned in the next part.
    
2. Added Helper function
    - majority_voting
        
        Calculate the accuracy made by chosen number of base learners.
        
    - diversity_evaluation_function:
        
        Diversity between two base learners:
        
        ![30511679293058_.pic.jpg](pic/30511679293058_.pic.jpg)
        
        (Kuncheva L I, Whitaker C J. Measures of diversity in classifier ensembles and their relationship with the ensemble accuracy. Machine learning, vol. 51, no. 2, pp.181-207, 2003.)
        
        Two return value:
        
        1. A symmetric matrix consists of diversities between any pairs of base_learners
        2. MSE (mean square error) of the ensemble
        
        ![30521679293495_.pic.jpg](pic/30521679293495_.pic.jpg)
        
    
    (Ko, AH-R., Robert Sabourin, and A. de Souza Britto, Combining diversity and classification accuracy for ensemble selection in random subspaces, IEEE International Joint Conference on Neural Network Proceedings, 2006.)
    
    - Part of the code expression:
    
    ```python
    import torch
    import numpy as np
    from config import *
    device = torch.device('cuda:2') if use_gpu else torch.device('cpu')
    
    def majority_voting(num_based_learners, model_list, data_test_loader, num_classes, device):
        total_correct = 0
        accuracy_list=[0]*num_based_learners
        with torch.no_grad():
            for _, (images, labels) in enumerate(data_test_loader):
                images, labels = images.to(device), labels.to(device)
                voting_result = torch.zeros(images.shape[0], num_classes).to(device)
                voting_local=torch.zeros(images.shape[0], num_classes).to(device)
                for n in range(num_based_learners):
                    model_list[n].eval()
                    output = model_list[n](images)
                    pred = output.data.max(1)[1]
                    for i in range(images.shape[0]):
                        voting_result[i][pred[i]] += 1
                        voting_local[i][pred[i]]+=1
                    pred_local=voting_local.argmax(dim=1)
                    accuracy_list[n]+=pred_local.eq(labels.data.view_as(pred_local)).sum()
                pred = voting_result.argmax(dim=1)
                total_correct += pred.eq(labels.data.view_as(pred)).sum()
    
        acc = float(total_correct) / len(data_test_loader.dataset)
        for i in range(num_based_learners):
            accuracy_list[i]=float(accuracy_list[i]) / len(data_test_loader.dataset)
    
        return acc, accuracy_list
    
    def q_statistics(num_based_learners, model_list, data_test_loader, device):
        pred_results = []
        with torch.no_grad():
            for n in range(num_based_learners):
                correct_list=[]
                model_list[n].eval()
                for _, (images, labels) in enumerate(data_test_loader):
                    images, labels = images.to(device), labels.to(device)
                    output = model_list[n](images)
                    pred = output.data.max(1)[1]
                    correct_list.append(pred.eq(labels.data.view_as(pred)))
                pred_results.append(torch.cat(correct_list))
    
        q = torch.zeros([num_based_learners, num_based_learners]).to(device)
        for i in range(num_based_learners):
            for j in range(num_based_learners):
                if i <= j:
                    n11 = (pred_results[i]*pred_results[j]).sum()
                    n00 = ((~pred_results[i])*(~pred_results[j])).sum()
                    n10 = (pred_results[i]*(~pred_results[j])).sum()
                    n01 = ((~pred_results[i])*pred_results[j]).sum()
                    q[i][j] = (float(n11*n00)-float(n10*n01)) / (float(n11*n00)+float(n10*n01))
                    q[j][i] = q[i][j]
        q_avg = q.sum()/(num_based_learners*num_based_learners)
        return q, q_avg
    
    def q_statistics_v2(record_used_sample,num_based_learners,model_list,data_test_loader, device):
        predict_wrong=0
        result=0
        data_test_size= 0 #len(data_test_loader.dataset)
        pred_results = []
        is_used=False
    
        with torch.no_grad():
            # for n in range(num_based_learners):
            #     model_list[n].eval()
                for idx, (images, labels) in enumerate(data_test_loader):
                    images=images.numpy()
                    labels=labels.numpy()
                    for img_num in range (32):
                        if((idx*32+img_num)>=len(record_used_sample)):
                            break
                        correct_list=[]
                        for n in range(num_based_learners):
                            if(record_used_sample[idx*32+img_num][n]== 1): # data used to train learner n
                                if (is_used==False): 
                                    data_test_size += 1
                                    is_used=True
                                    # single img load to the device
                                    #images=np(size=images[0].shape,dtype = torch.float32)
                                    image=images[img_num]
                                    image=torch.from_numpy(image)
                                    image=torch.unsqueeze(image,0)
                                    label_v=labels[img_num]
                                    label=np.array([0])
                                    label[0]=label_v
                                    label=torch.from_numpy(label)
                                    image, label = image.to(device), label.to(device)
                                model_list[n].to(device)
                                model_list[n].eval()
                                output = model_list[n](image)
                                pred = output.data.max(1)[1]
                                # add single result to the correct_list
                                correct_list.append(pred.eq(label.data.view_as(pred)))
                        is_used=False
                        if (len(correct_list)!= 0):
                            pred_results.append(torch.cat(correct_list))
                            # pred_results [img label][base learner result] 0<=len(base learner result)<=num_of_base_learners
                    if((idx*32+img_num)>=len(record_used_sample)):
                        break
    
        for i in range(len(pred_results)):
            Id = 0
            for j in range(len(pred_results[i])):
                if (~pred_results[i][j]):
                    Id += 1
            predict_wrong += (Id **2)
            Id = 0
        result = float(1/(num_based_learners*(num_based_learners-1)*data_test_size)*predict_wrong)
        return result
    
    def mse_calculation(diversity_matrix,accuracy_list,num_of_base_learners):
        result_d=1
        result_a=1
        count=1
        result=[1]*num_of_base_learners
        for k in range(1,num_of_base_learners+1):
            for i in range (k):
                for j in range (k):
                    if i <= j : break
                    else:
                        result_d*=(1-diversity_matrix[i][j]) #一旦一个模型一样就作废
            for num in accuracy_list:
                if k>1:
                    result_a*=(1-num)
                    count+=1
                    if count==k+1: 
                        count=1
                        break
                else:
                    break
            result[k-1]=result_d**(1/((num_of_base_learners-1)*num_of_base_learners))*result_a**(1/num_of_base_learners)
            result_a=1
            result_d=1
        return result
    ```
    

# 4.Result Analysis

Dataset: SVHN

Model: WResnet

Number of base learners: 3

Sampling size for each base learner: 20000

Epoch: 10000

1. Example of training logs of model

![pic/30541679341496_.pic.jpg](pic/30541679341496_.pic.jpg)

![pic/30551679341513_.pic.jpg](pic/30551679341513_.pic.jpg)

1. Example of diversity matrix and MSE square
    
    ![pic/30561679341868_.pic.jpg](pic/30561679341868_.pic.jpg)
