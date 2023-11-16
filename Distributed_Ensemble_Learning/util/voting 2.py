import torch


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

def q_statistics_v2(num_based_learners,model_list,data_test_loader, device):
    predict_wrong=0
    result=0
    data_test_size=len(data_test_loader.dataset)
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
        
    for i in range(len(pred_results)):
        Id = 0
        for j in range(num_based_learners):
            if (~pred_results[j][i]):
                Id += 1
        predict_wrong += Id **2
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
