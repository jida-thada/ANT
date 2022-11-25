import torch
import torch.nn as nn
import numpy as np
import itertools
from itertools import compress

class CC_OrderFree(nn.Module):
    def __init__(self, N_FEATURES, HIDDEN_DIM, N_CLASSES, N_LAYERS=1, DEVICE='cpu', CELL_TYPE='LSTM'):
        super(CC_OrderFree, self).__init__()
        self.NAME___ = 'CC_OrderFree'
        self.HIDDEN_DIM = HIDDEN_DIM
        self.N_FEATURES = N_FEATURES
        self.N_CLASSES = N_CLASSES
        self.CELL_TYPE = CELL_TYPE
        self.device = DEVICE
        self.N_LAYERS = N_LAYERS
        self.input_size = self.N_FEATURES + 2*self.N_CLASSES

        if CELL_TYPE == "RNN":
            self.rnn_cell = nn.RNN(self.input_size, HIDDEN_DIM, self.N_LAYERS, batch_first=False)
        elif CELL_TYPE == "GRU":
            self.rnn_cell = nn.GRU(self.input_size, HIDDEN_DIM, self.N_LAYERS, batch_first=False)
        elif CELL_TYPE == "LSTM":
            self.rnn_cell = nn.LSTM(self.input_size, HIDDEN_DIM, self.N_LAYERS, batch_first=False)

        self.rnn_out = nn.Linear(HIDDEN_DIM, self.N_CLASSES) # Soft confidence P_t: Hidden state -> output space
        self.init_out = nn.Linear(self.N_FEATURES, self.N_CLASSES) # Initialize output v_pred: logits
        

    def initHidden(self, BATCH_SIZE):
        if self.CELL_TYPE == "LSTM":
            return (torch.zeros(self.N_LAYERS, BATCH_SIZE, self.HIDDEN_DIM).to(self.device),
                    torch.zeros(self.N_LAYERS, BATCH_SIZE, self.HIDDEN_DIM).to(self.device))
        else:
            return torch.zeros(self.N_LAYERS, BATCH_SIZE, self.HIDDEN_DIM).to(self.device)

    def getMask(self, predicted_classes):
        T, bs, num_class = predicted_classes.size()
        mask = torch.where(predicted_classes.sum(0) > 0, torch.zeros((bs, num_class)), torch.ones((bs, num_class)))
        return mask    

    def forward(self, X, y, TRAINING = True, init_y = None, init_logits = None, start=0, verbose=False):
        self.bs = X.size()[0]
        if init_logits is not None: 
            self.logits = self.init_out(X)
            self.logits = torch.where(init_logits == 0, self.logits, init_logits)
        else: self.logits = self.init_out(X)
        if init_y is not None: self.yhat = init_y
        else: self.yhat = torch.zeros((self.N_CLASSES, self.bs, self.N_CLASSES)) # Initialize yhat size: time_step * bs * num_classes        
        self.state = self.initHidden(self.bs)
        self.logits_final = torch.zeros((self.bs, self.N_CLASSES))
        self.prob_path = torch.zeros((self.N_CLASSES, self.bs, self.N_CLASSES))
        self.prob_path_score = torch.ones((self.bs))
        self._loss = 0
        self.yhat_final = self.yhat.sum(0)
        
        for t in range(start,self.N_CLASSES):    
            self.X_aug = torch.cat((self.logits.view_as(self.yhat_final), X, self.yhat_final), dim=1)
            self.output, self.state = self.rnn_cell(self.X_aug.unsqueeze(0), self.state)
            self.logits = self.rnn_out(self.output) # New Pt
            self.mask = self.getMask(self.yhat)
            self.mask_logit = self.mask.float() * self.logits
            self.mask_sigmoid = self.mask.float() * torch.sigmoid(self.logits)
            self.argmax_logit = torch.eye(int(self.N_CLASSES))[torch.max(self.mask_sigmoid, 2)[1]]
            
            if t == 0: 
                self.prob_s0 = self.logits
                self.yhat[t,:,:] = torch.where(self.mask_sigmoid > 0.5, self.argmax_logit, self.yhat[t,:,:])
                self.yhat_final = self.yhat.sum(0)
                self.logits_final = self.logits
            elif t > 0: 
                if (self.prob_path_score > 0.5).sum() > 0:
                    self.maskout = (self.prob_path_score > 0.5).unsqueeze(2).expand_as(self.mask_sigmoid)
                    self.yhat_temp = torch.where(self.mask_sigmoid > 0.5, self.argmax_logit, torch.zeros((self.bs, self.N_CLASSES)))
                    self.yhat[t,:,:] = torch.where(self.maskout == True, self.yhat_temp, torch.zeros((self.bs, self.N_CLASSES)))
                    self.yhat_final = self.yhat.sum(0)
                    self.logits_final = torch.where(self.yhat[t,:,:] == 1, self.logits, self.logits_final)
                else:
                    self.yhat[t,:,:] = torch.zeros((self.bs, self.N_CLASSES))
                    self.yhat_final = self.yhat.sum(0)
                    self.logits_final = torch.where(self.yhat[t,:,:] == 1, self.logits, self.logits_final)
            self.prob_path_final = self.argmax_logit*self.mask_sigmoid
            self.prob_path_score = self.prob_path_score * self.prob_path_final.sum(2)
    
            if TRAINING:
                self.loss = self.computeLoss(self.logits.squeeze(), y)
                self._loss += self.loss
            
        return self.logits_final, self.yhat_final, self._loss, self.yhat, self.prob_s0, self.output

    def computeLoss(self, logits, labels): 
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(logits, labels)
        return loss





class ANT(nn.Module):
    def __init__(self, N_FEATURES, HIDDEN_DIM, N_CLASSES, N_LAYERS=1, DEVICE='cpu', num_teachers=2,
                teachers=None, t_names=None, t_classes=None,CELL_TYPE='LSTM'):
        super(ANT, self).__init__()
        self.NAME___ = 'ANT'
        self.teachers = teachers
        self.t_names = t_names
        self.t_classes = t_classes
        self.HIDDEN_DIM = HIDDEN_DIM
        self.N_FEATURES = N_FEATURES
        self.N_CLASSES = N_CLASSES
        self.CELL_TYPE = CELL_TYPE
        self.device = DEVICE
        self.N_LAYERS = N_LAYERS
        self.input_size = self.N_FEATURES + 2*self.N_CLASSES
        self.num_teachers = num_teachers

        if CELL_TYPE == "RNN":
            self.rnn_cell = nn.RNN(self.input_size, HIDDEN_DIM, self.N_LAYERS, batch_first=False)
        elif CELL_TYPE == "GRU":
            self.rnn_cell = nn.GRU(self.input_size, HIDDEN_DIM, self.N_LAYERS, batch_first=False)
        elif CELL_TYPE == "LSTM":
            self.rnn_cell = nn.LSTM(self.input_size, HIDDEN_DIM, self.N_LAYERS, batch_first=False)

        self.rnn_out = nn.Linear(HIDDEN_DIM, self.N_CLASSES) # Soft confidence P_t: Hidden state -> output space
        self.init_out = nn.Linear(self.N_FEATURES, self.N_CLASSES) # Initialize output v_pred: logits

    def initHidden(self, BATCH_SIZE):
        if self.CELL_TYPE == "LSTM":
            return (torch.zeros(self.N_LAYERS, BATCH_SIZE, self.HIDDEN_DIM).to(self.device),
                    torch.zeros(self.N_LAYERS, BATCH_SIZE, self.HIDDEN_DIM).to(self.device))
        else:
            return torch.zeros(self.N_LAYERS, BATCH_SIZE, self.HIDDEN_DIM).to(self.device)

    def getMask(self, predicted_classes):
        T, bs, num_class = predicted_classes.size()
        mask = torch.where(predicted_classes.sum(0) > 0, torch.zeros((bs, num_class)), torch.ones((bs, num_class)))
        return mask

    def forward(self, X, y, TRAINING = True, init_y = None, init_logits = None, start=0, verbose=False):
        y = self.extract_teacher(self.num_teachers, X, y, self.t_names, self.t_classes, self.teachers)
        
        self.bs = X.size()[0]
        if init_logits is not None:
            self.logits = self.init_out(X)
            self.logits = torch.where(init_logits == 0, self.logits, init_logits)
        else: self.logits = self.init_out(X)
        if init_y is not None: self.yhat = init_y
        else: self.yhat = torch.zeros((self.N_CLASSES, self.bs, self.N_CLASSES)) # Initialize yhat size: time_step * bs * num_classes        
        self.state = self.initHidden(self.bs)
        self.logits_final = torch.zeros((self.bs, self.N_CLASSES))
        self.prob_path = torch.zeros((self.N_CLASSES, self.bs, self.N_CLASSES))
        self.prob_path_score = torch.ones((self.bs))
        self._loss = 0
        self.yhat_final = self.yhat.sum(0)

        for t in range(start,self.N_CLASSES):
            self.X_aug = torch.cat((self.logits.view_as(self.yhat_final), X, self.yhat_final), dim=1)
            self.output, self.state = self.rnn_cell(self.X_aug.unsqueeze(0), self.state)
            self.logits = self.rnn_out(self.output) # New Pt
            self.mask = self.getMask(self.yhat)
            self.mask_logit = self.mask.float() * self.logits
            self.mask_sigmoid = self.mask.float() * torch.sigmoid(self.logits)
            self.argmax_logit = torch.eye(int(self.N_CLASSES))[torch.max(self.mask_sigmoid, 2)[1]]    
            if t == 0:  
                self.prob_s0 = self.logits
                self.yhat[t,:,:] = torch.where(self.mask_sigmoid > 0.5, self.argmax_logit, self.yhat[t,:,:])
                self.yhat_final = self.yhat.sum(0)
                self.logits_final = self.logits
            elif t > 0:  
                if (self.prob_path_score > 0.5).sum() > 0:
                    self.maskout = (self.prob_path_score > 0.5).unsqueeze(2).expand_as(self.mask_sigmoid)
                    self.yhat_temp = torch.where(self.mask_sigmoid > 0.5, self.argmax_logit, torch.zeros((self.bs, self.N_CLASSES)))
                    self.yhat[t,:,:] = torch.where(self.maskout == True, self.yhat_temp, torch.zeros((self.bs, self.N_CLASSES)))
                    self.yhat_final = self.yhat.sum(0)
                    self.logits_final = torch.where(self.yhat[t,:,:] == 1, self.logits, self.logits_final)
                else:
                    self.yhat[t,:,:] = torch.zeros((self.bs, self.N_CLASSES))
                    self.yhat_final = self.yhat.sum(0)
                    self.logits_final = torch.where(self.yhat[t,:,:] == 1, self.logits, self.logits_final)

    
            self.prob_path_final = self.argmax_logit*self.mask_sigmoid
            self.prob_path_score = self.prob_path_score * self.prob_path_final.sum(2)
    
            if TRAINING:
                self.loss = self.soft_BCE(self.logits.squeeze(), y)
                self._loss += self.loss
            
        return self.logits_final, self.yhat_final, self._loss, self.yhat, self.prob_s0, self.output


    def soft_BCE(self, logits, target, T=1.0, size_average=True, target_is_prob=False):
        criterion = nn.BCELoss()
        if target_is_prob: p_target = target
        else: p_target = torch.sigmoid(target/T)
        logits = torch.sigmoid(logits/T)        
        loss = criterion(logits, p_target)
        return loss 


    def extract_teacher(self, num_teachers, X, y, t_names, t_classes, teachers):
        # --- Extract teachers' info ---
        t_logits = []
        t_hardpreds = []
        t_yhats = []
        logit_s0 = []
        for t in range(num_teachers):
            t_logit_t, t_hardpred_t, loss_t, t_yhat_t, logit_s0_t, t_last_feature = self.teachers[t](X,y,False)
            t_logits.append(t_logit_t)
            t_hardpreds.append(t_hardpred_t)
            t_yhats.append(t_yhat_t)
            logit_s0.append(logit_s0_t)
            
        indexclass_t = {}
        t_logit = {}
        t_lab = {}; T = {}; k=0
        indexclass_stu = {}; j=0
        
        for t in range(num_teachers):
            # create dict for teachers
            t_logit[t_names[t]]=t_logits[t]
            T[t_names[t]]=teachers[t]
            t_lab[t_names[t]]=t_classes[t]
            k+=1
            
            indexclass_tmp = {}
            i=0
            for c in t_classes[t]:
                indexclass_tmp[c]=i
                i+=1
                # create dict for student
                if c not in indexclass_stu.keys():
                    indexclass_stu[c]=j
                    j+=1
            indexclass_t[t_names[t]] = indexclass_tmp
        
        stu_lab =  list(indexclass_stu.keys())
        stu_nclass = len(stu_lab)
        logit = torch.zeros((1,X.size()[0],stu_nclass))
        logit[:,:,:] = -999
        
        
        # check global common classes
        commonclass_check = [False]*len(stu_lab)
        t_rela = {}
        
        for j,lab in enumerate(stu_lab):
            cnt_t = 0
            t_experts =[]
            for nt in range(num_teachers):
                if lab in t_lab[t_names[nt]]: 
                    cnt_t += 1
                    t_experts.append(t_names[nt])
            if cnt_t > 1: 
                commonclass_check[j] = True
                t_rela_tmp = []
                for pair in itertools.permutations(t_experts):
                    p_append = []
                    for p in pair:
                        p_append.append(t_names.index(p)) 
                    t_rela_tmp.append(p_append)
                t_rela[lab]=t_rela_tmp 
        commonclass = list(compress(stu_lab, commonclass_check))

        # run over each instance
        for ins in range(X.size()[0]):
            ins_common_label = []
            ins_rela_tranfer = {}
            
            #Extract initial commonlabel
            hardpred = []
            for t in range(num_teachers):
                logit_tmp = t_logit[t_names[t]][:,ins,:]
                index_positive = torch.where(torch.sigmoid(logit_tmp)>0.5)[1]
                hardpred_tmp = [t_lab[t_names[t]][p_index] for p_index in index_positive]
                hardpred.append(hardpred_tmp)
            label_set = []
            for l in hardpred:
                label_set.extend(l)
            ins_common_label.extend(list(set(commonclass) & set(label_set)))
            
            # Extract initial teacher relation for transfer knowledge
            for lab in ins_common_label: 
                rela_tmp = []
                rela_tmp_tmp = []
                for t in range(num_teachers):
                    if lab in hardpred[t]: rela_tmp_tmp.append(t)
                rela_tmp.append(rela_tmp_tmp)
                
                rela_tranfer = []
                for r in rela_tmp:
                    if len(r)==1: 
                        t_seed = r[0]
                        for rela in t_rela[lab]:
                            if rela[0]==t_seed and rela not in rela_tmp:
                                rela_tranfer.append(rela)
                
                if len(rela_tranfer)>0: ins_rela_tranfer[lab] = rela_tranfer
              
            logit_imp = torch.zeros((1,stu_nclass))
            logit_imp[:,:] = -999
            
            t_logit_ins = {}
            for t in range(num_teachers):
                t_logit_ins[t_names[t]] = t_logit[t_names[t]][:,ins,:]
            if len(ins_rela_tranfer) == 0: # No transfer knowledge
                logit_imp = torch.tensor(combine_output(ins,stu_lab,num_teachers,t_lab,t_names,t_logit_ins,indexclass_t))
            else: # Transfer knowledge
                running_rela = {}
                list_logit_imp = []
                label_for_transfer = list(ins_rela_tranfer.keys())
                logit_imp_list = transfer(label_for_transfer, ins_rela_tranfer,stu_nclass,t_logit,T,ins,X,y,indexclass_t,
                                          indexclass_stu,stu_lab,t_lab,logit_imp,ins_common_label,commonclass,t_logit_ins,
                                          running_rela,list_logit_imp,t_names,num_teachers,t_rela,ins_rela_tranfer)
                #convert to hardpred
                ins_logit = []
                new_logit_list = []
                hardpred_list = []
                for n, l in enumerate(logit_imp_list):
                    logit_tmp = l.squeeze().numpy().tolist()
                    new_logit_list.append(logit_tmp)
                    hardpred_tmp = torch.sigmoid(logit_imp_list[n])
                    hardpred_tmp[hardpred_tmp>0.5] = 1.0
                    hardpred_tmp[hardpred_tmp<=0.5] = 0.0
                    hardpred_list.append(hardpred_tmp.squeeze().numpy().tolist())
                sum_lab = np.sum(np.array(hardpred_list),axis=0)
                new_logit_array = np.array(new_logit_list)
                for n in range(stu_nclass):
                    if sum_lab[n] == 0: final_logit = np.min(new_logit_array[:,n])
                    else: final_logit = np.max(new_logit_array[:,n])
                    ins_logit.append(final_logit)
                logit_imp = torch.tensor(ins_logit)
            logit[:,ins,:] = logit_imp
        
        y = torch.FloatTensor(logit.squeeze())
        
        return y


def impute_logit(logit_imp,ins_list_infer,ins_list,stu_lab,T_lab,target_T,from_T,indexclass_stu,t_logit,indexclass,common_label_list,X,y,T):
    i = 0
    for ins in ins_list_infer:
        
        if ins not in ins_list:
            for lab in stu_lab:
                for t in T.keys(): 
                    if lab in T_lab[t]: 
                        logit_compare_list = [logit_imp[:,ins,indexclass_stu[lab]].detach().numpy(),t_logit[t][:,ins,indexclass[t][lab]].detach().numpy()]
                        logit_imp[:,ins,indexclass_stu[lab]] = torch.tensor(np.max(logit_compare_list, axis = 0))
               
        else:
            common_label = common_label_list[i]
            x_test = torch.Tensor([X[ins].numpy().tolist(),X[ins].numpy().tolist()])
            y_test = torch.Tensor([y[ins].numpy().tolist(),y[ins].numpy().tolist()])
            nclass = len(indexclass[target_T])
            
            init_logits_t = torch.zeros((y_test.size()[0],nclass))
            init_logits_t[:,indexclass[target_T][common_label]] = t_logit[from_T][:,ins,indexclass[from_T][common_label]]
            init_l = torch.where(init_logits_t == 0, t_logit[target_T][:,ins], init_logits_t)
            t_logit_imp, t_hardpred_imp, loss_imp, t_yhat_imp, prob_s0_t_imp = T[target_T](x_test,y_test,False,init_logits=init_l)
            
            for lab in stu_lab:
                if lab in T_lab[target_T]:
                    logit_imp[:,ins,indexclass_stu[lab]] = t_logit_imp[0][0][indexclass[target_T][lab]]
                else:
                    for t in T.keys(): 
                        if lab in T_lab[t]: 
                            logit_compare_list = [logit_imp[:,ins,indexclass_stu[lab]].detach().numpy(),t_logit[t][:,ins,indexclass[t][lab]].detach().numpy()]
                            logit_imp[:,ins,indexclass_stu[lab]] = torch.tensor(np.max(logit_compare_list, axis = 0))
            i += 1
    return logit_imp


def combine_output(ins,stu_lab,num_teachers,t_lab,t_names,t_logit_ins,indexclass_t):  
    ins_logit = []
    for i,lab in enumerate(stu_lab):
        #print(lab)
        
        # Extract all hardpreds from all teachers
        lab_logits = np.array([])
        for t in range(num_teachers):
            if lab in t_lab[t_names[t]]: 
                t_logit_for_lab = t_logit_ins[t_names[t]][:,indexclass_t[t_names[t]][lab]].item()
            else: 
                t_logit_for_lab = np.nan
            lab_logits = np.append(lab_logits, t_logit_for_lab)
        lab_hardpred = torch.sigmoid(torch.tensor(lab_logits))
        lab_hardpred[lab_hardpred>0.5] = 1.0
        lab_hardpred[lab_hardpred<=0.5] = 0.0
        #print(lab_hardpred)
    
        # If all negative, min. If some positive, max.
        if np.nansum(lab_hardpred) == 0: # All negative
            final_logit = np.nanmin(lab_logits)
        elif np.nansum(lab_hardpred) >= 1: # Some positive
            final_logit = np.nanmax(lab_logits)
        ins_logit.append(final_logit)
    return ins_logit


def extract_transfer_rela(label_list,t_rela,t_names,origin_t):
    for lab in label_list: 
        rela = {}
        rela_tranfer = []
        rela_tmp = t_rela[lab]
        for r in rela_tmp:
            if r[0] == t_names.index(origin_t): rela_tranfer.append(r)
        rela[lab] = rela_tranfer
    return rela
                        

def transfer(common_label,rela_tranfer,stu_nclass,t_logit,T,ins,X,y,indexclass_t,indexclass_stu,stu_lab,t_lab,logit_imp,
             ins_common_label,commonclass,t_logit_ins,running_rela,list_logit_imp,t_names,num_teachers,t_rela,ins_rela_tranfer, verbose=False):
    for lab in common_label: 
        if verbose: print(lab)
        
        # TRANSFER
        for r in rela_tranfer[lab]: 
            if lab in running_rela.keys(): running_rela[lab].append(r)
            else: running_rela[lab]=[r]
            if verbose: print('rela',rela_tranfer)
            if verbose: print('running_rela',running_rela)
            from_T = t_names[r[0]]
            to_T = t_names[r[1]]
            if verbose: print(from_T, '-->', to_T)         
                
            t_logit_imp = impute_logit_new(ins,lab,X,y,indexclass_t,from_T,to_T,t_logit,T,indexclass_stu)
            if verbose: print('sigmoid',torch.sigmoid(t_logit_imp))
            
            t_logit_new = t_logit_ins.copy()
            t_logit_new[to_T] = t_logit_imp.view_as(t_logit_ins[to_T])
            
            logit_imp = torch.tensor(combine_output(ins,stu_lab,num_teachers,t_lab,t_names,t_logit_new,indexclass_t)).view_as(logit_imp)
            if verbose: print('logit_imp',torch.sigmoid(logit_imp))
            list_logit_imp.append(logit_imp)
            
            # Check if recursive process is needed
            
            hardpred_ori = torch.sigmoid(t_logit_ins[to_T]).squeeze()
            hardpred_ori[hardpred_ori>0.5] = 1.0
            hardpred_ori[hardpred_ori<=0.5] = 0.0
            
            hardpred_imp = torch.sigmoid(t_logit_imp).squeeze()
            hardpred_imp[hardpred_imp>0.5] = 1.0
            hardpred_imp[hardpred_imp<=0.5] = 0.0
            
            new_positive = []
            for pred_lab in range(len(hardpred_imp)):
                if hardpred_imp[pred_lab] > hardpred_ori[pred_lab]:
                    new_positive.append(t_lab[to_T][pred_lab])
            
            # check if new_positive in common label
            new_common_label = set(new_positive) & set(commonclass)
            if verbose: print('new_common_label',new_common_label)
            
            if len(new_common_label)>0:
                if verbose: print('cont')
                
                # extract next rela
                new_rela_tranfer = {}
                new_rela_tranfer_all = extract_transfer_rela(list(new_common_label),t_rela,t_names,to_T)   
                if verbose: print('new_rela_tranfer_all',new_rela_tranfer_all)
                for k,vset in new_rela_tranfer_all.items(): 
                    
                    for v in vset:
                        reverse_rela = v.copy()
                        reverse_rela.reverse()
                        
                        cond1 = False
                        if k not in ins_common_label: cond1 = True
                        elif reverse_rela not in ins_rela_tranfer[k]: cond1 = True
                        
                        cond2 = False
                        if k not in running_rela.keys(): cond2 = True
                        elif v not in running_rela[k] and reverse_rela not in running_rela[k]: cond2 = True
                        
                        if cond1 and cond2: 
                            if k not in new_rela_tranfer.keys(): new_rela_tranfer[k]=[v]
                            else: new_rela_tranfer[k].append(v)
                if verbose: print('new_rela_tranfer',new_rela_tranfer)
                if len(new_rela_tranfer) > 0:
                    new_common_label = list(new_rela_tranfer.keys())
                    list_logit_imp = transfer(new_common_label, new_rela_tranfer,stu_nclass,t_logit,T,ins,X,y,indexclass_t,indexclass_stu,
                                              stu_lab,t_lab,logit_imp,ins_common_label,commonclass,t_logit_new,running_rela,list_logit_imp,t_names,num_teachers,t_rela,ins_rela_tranfer)
            
    return list_logit_imp

                                
def impute_logit_new(ins,lab,X,y,indexclass_t,from_T,to_T,t_logit,T,indexclass_stu):
    x_test = torch.Tensor([X[ins].numpy().tolist()])
    y_test = torch.Tensor([y[ins].numpy().tolist()])
    
    nclass = len(indexclass_t[to_T])
    init_logits_t = torch.zeros((y_test.size()[0],nclass))
    init_logits_t[:,indexclass_t[to_T][lab]] = t_logit[from_T][:,ins,indexclass_t[from_T][lab]]
    init_l = torch.where(init_logits_t == 0, t_logit[to_T][:,ins], init_logits_t)
    t_logit_imp, t_hardpred_imp, loss_imp, t_yhat_imp, prob_s0_t_imp, t_last_feature = T[to_T](x_test,y_test,False,init_logits=init_l)
    
    return t_logit_imp





        
