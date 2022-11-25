import numpy as np
import pickle
import csv
from sklearn import metrics
import torch
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = "serif"
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
        'axes.labelsize': 'x-large',
        'axes.titlesize':'x-large',
        'xtick.labelsize':'x-large',
        'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)




def write_row(row, name, path="./", round=False):
    if round:
        row = [np.round(i, 2) for i in row]
    f = path + name + ".csv"
    with open(f, "a+") as csvfile:
        filewriter = csv.writer(csvfile, delimiter=",")
        filewriter.writerow(row)

def plot_loss(plt_name,ep,losses,save_path):
    plt.figure(figsize=(15,10))
    plt.plot(range(ep),losses)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(save_path+plt_name, bbox_inches='tight')
    plt.clf()


def multi_label_metric(output, target):
    correct = np.sum(np.mean((output == target), axis=1)==1)
    total = target.shape[0]
    subset_loss = 1-(correct / total)
    subset_acc = 1 - subset_loss
    
    hamming_loss = 1-np.mean((output == target))
    hamming_acc = 1-hamming_loss
    
    preMI = metrics.precision_score(target, output, average='micro')
    preMA = metrics.precision_score(target, output, average='macro')
    
    recallMI = metrics.recall_score(target, output, average='micro')
    recallMA = metrics.recall_score(target, output, average='macro')
    
    f1MI = metrics.f1_score(target, output, average='micro')
    f1MA = metrics.f1_score(target, output, average='macro')
    
    return subset_loss,hamming_loss,preMI,recallMI,f1MI,preMA,recallMA,f1MA


def evaluation(model, x, y, stu_target_lab_order, indexclass_stu):
    logit, hardpred, loss, yhat, mask, last_feature = model(torch.FloatTensor(x), torch.FloatTensor(y))

    new_logit = torch.zeros(logit.size())
    for i,j in enumerate(stu_target_lab_order):
        new_logit[:,:,i]=logit[:,:,indexclass_stu[j]]
    logit = new_logit

    hard_pred = torch.sigmoid(logit)
    hard_pred[hard_pred>0.5] = 1.0
    hard_pred[hard_pred<=0.5] = 0.0
    output = hard_pred.squeeze().data.numpy()
    target = y
    performance = multi_label_metric(output, target)
    return performance


def write_performance(header, csv_filename, performance):
    logpath = './output/'
    row = [header]; write_row(row, csv_filename, path=logpath, round=False)
    row = ['   subset loss    :', round(performance[0],4)]; write_row(row, csv_filename, path=logpath, round=False)
    row = ['   hamming loss   :', round(performance[1],4)]; write_row(row, csv_filename, path=logpath, round=False)
    row = ['   f1 Micro       :', round(performance[4],4)]; write_row(row, csv_filename, path=logpath, round=False)
    row = ['   f1 Macro       :', round(performance[7],4)]; write_row(row, csv_filename, path=logpath, round=False)


def save_model(modeltosave, pathtosave):
    pickle.dump(modeltosave, open(pathtosave, 'wb'))


def load_teacher_model(t_path):
    teachers = [] 
    num_teacher = len(t_path)
    for t in range(num_teacher):
        net_t = pickle.load(open(t_path[t], 'rb'))
        teachers.append(net_t)
    return teachers


def read_file(txt_path):
    x = []
    with open(txt_path,'r') as input_path:  
        for line in input_path.readlines():
            line = line.strip().split(',')
            feature = list(map(float,line))
            x.append(feature)
    return x


def read_data(dataname):
    dataname = dataname[0]
    txt_path = './data/'+dataname+'/xtrain.txt'
    x_train = read_file(txt_path)
    x_train = np.array(x_train)
    txt_path = './data/'+dataname+'/ytrain.txt'
    y_train = read_file(txt_path)
    y_train = np.array(y_train)
    txt_path = './data/'+dataname+'/xtest.txt'
    x_test = read_file(txt_path)
    x_test = np.array(x_test)
    txt_path = './data/'+dataname+'/ytest.txt'
    y_test = read_file(txt_path)
    y_test = np.array(y_test)
    return x_train,y_train,x_test,y_test