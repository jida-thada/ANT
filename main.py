import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from utils import *
from model import *

import matplotlib.pyplot as plt 
plt.switch_backend('agg')

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
                             'axes.labelsize': 'x-large',
                             'axes.titlesize':'x-large',
                             'xtick.labelsize':'x-large',
                             'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)




def prepare_data(t_numclass,t_labels,stu_labels):
    num_teachers = len(t_numclass)
    t_classes = []
    idx = 0
    for i in t_numclass:
        t_classes.append(t_labels[idx:idx+i])
        idx += i
    t_names = []
    for i in range(num_teachers):
        t_name_i = 't'+str(i)
        t_names.append(t_name_i)
    indexclass_stu = dict(zip(stu_labels, [i for i in range(len(stu_labels))]))
        
    common_lab = t_classes[0].copy()
    for i, subclass in enumerate(t_classes):
        if i > 0: 
            for j in common_lab:
                if j not in subclass:
                    common_lab.remove(j)
    
    return t_names,t_classes,indexclass_stu,common_lab
    
    

def train_ANT(x_train,y_train,x_test,y_test,hidden,layer,bs,ep,lr,teachers,t_names,t_classes,stu_labels,indexclass_stu):
    num_classes = len(stu_labels)
    feature_size = x_train.shape[1]
    num_teacher = len(t_names)
    device = 'cpu'
    model = ANT(feature_size, hidden, num_classes, layer, device, 
                    num_teachers=num_teacher,
                    teachers=teachers, 
                    t_names=t_names, 
                    t_classes=t_classes)
    train_data=TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
    train_loader=DataLoader(train_data, batch_size=bs, shuffle=True, drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)
    csv_filename = 'Training'
    model, loss_list = run_train(csv_filename,ep,train_loader,model,device,optimizer,lr_scheduler)
    
    # Training performance
    x = x_train; y = y_train
    performance = evaluation(model,x,y,stu_labels,indexclass_stu); write_performance('Training', csv_filename, performance)
    # Testing performance
    x = x_test; y = y_test
    performance = evaluation(model,x,y,stu_labels,indexclass_stu); write_performance('Testing', csv_filename, performance)
    
    plt_name = 'pltloss'
    logpath = './output/'
    plot_loss(plt_name,ep,loss_list,logpath)
    pathtosave = logpath+model.NAME___+'.sav'
    save_model(model, pathtosave) 



def run_train(csv_filename,num_epochs,train_loader,model,device,optimizer,lr_scheduler):
    loss_list = []
    for epoch in range(num_epochs):    
        final_loss = 0.0
        for i, (X, y) in enumerate(train_loader):
            y = y.to(device)
            # --- Forward pass ---
            logits, yhat, loss, yhat_raw, mask, last_feature = model(X,y,TRAINING = True)
            # --- Backward and optimize ---
            optimizer.zero_grad()
            final_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        if (epoch == 0) or ( (epoch+1) % (num_epochs/10) == 0 ):
            row = ['ep',epoch+1,'/',num_epochs,': Loss ', np.mean(final_loss)];write_row(row, csv_filename, path='./output/', round=False)
        if i > 0: loss_list.append(final_loss/float(i))
        else: loss_list.append(final_loss)
        lr_scheduler.step()
    return model, loss_list



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    
    add_arg('-t_numlabel',  nargs='+',  type=int,   required=True,  help='#labels corresponding to each teacher')
    add_arg('-t_path',      nargs='+',  type=str,   required=True,  help=''' path of each teacher model, e.g., './t0.sav' './t1.sav' ''')
    add_arg('-t_labels',    nargs='+',  type=str,   required=True,  help='concatenated labels of teachers corresponding to t_path, e.g., t0_label: 1 2 3 4 and t1_label: 3 4 5 6, then t_labels: 1 2 3 4 3 4 5 6')
    add_arg('-stu_labels',  nargs='+',  type=str,   required=True,  help='student labels, e.g., 1 2 3 4 5 6')
    add_arg('-dataname',    nargs='+',  type=str,   required=True,  help='dataname, e.g., data1 ')
    add_arg('-lr',                      type=float, default=0.01,   help='Student: learning rate')
    add_arg('-ep',                      type=int,   default=200,    help='Student: total epochs for training')
    add_arg('-bs',                      type=int,   default=8,      help='Student: batch size')
    add_arg('-layer',                  type=int,   default=2,      help='Student: #layers')
    add_arg('-hidden',                  type=int,   default=8,      help='Student: #hidden units')
    
    args = parser.parse_args()
    t_numlabel = args.t_numlabel
    t_path = args.t_path
    t_labels = args.t_labels 
    stu_labels = args.stu_labels
    dataname = args.dataname
    lr = args.lr
    ep = args.ep
    bs = args.bs
    layer = args.layer
    hidden = args.hidden
    
    x_train,y_train,x_test,y_test = read_data(dataname)
    teachers = load_teacher_model(t_path)
    t_names,t_classes,indexclass_stu,common_lab = prepare_data(t_numlabel,t_labels,stu_labels)
    train_ANT(x_train,y_train,x_test,y_test,hidden,layer,bs,ep,lr,teachers,t_names,t_classes,stu_labels,indexclass_stu)
    
    