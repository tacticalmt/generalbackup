import numpy as np
import torch.nn as nn
import torch
import os
import Liblinks.utis as f_s
import random
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


mat=np.zeros((3,4))
conv1 = nn.Linear(2, 3)
input_l = torch.randn(3, 2)
print(input_l)
out_res = conv1(input_l)
print(out_res)
input_l = input_l.cuda()
act1 = nn.ReLU(inplace=True)
# act1.cuda()
after_res = act1(1 - input_l)
after_res += act1(1 + input_l)
out = after_res
seman_path = os.path.abspath('./')
dim=300
a=torch.randn(32,85)
b=torch.randn(85,40)

c=torch.mm(a,b)

#f,(ax1,ax2)=plt.subplots(2,2)
ran_mat=a.numpy()
#sns.set()
#ax=sns.heatmap(ran_mat,cmap='rainbow')
#ax.tick_params(labelsize=8)
#string_l=ax.get_yticklabels()
#ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
#ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
#ax.set_xlabel('true label')
#ax1=sns.heatmap(ran_mat,cmap='rainbow')
#ax1.tick_params(labelsize=8)
#ax1.set_yticklabels(ax.get_yticklabels(), rotation=0)
#ax1.set_xticklabels(ax.get_xticklabels(), rotation=0)
#ax1.set_ylabel('true label')


ax1=plt.subplot(2,2,1)
sns.heatmap(ran_mat,ax=ax1,vmax=10.0,vmin=0.0,cmap='rainbow')
ax1.tick_params(labelsize=8)
ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
ax1.set_xlabel('true label')
ax2=plt.subplot(2,2,2)
sns.heatmap(ran_mat,ax=ax2,vmax=20.0,vmin=0,cmap='rainbow')
ax2.tick_params(labelsize=8)
ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
ax2.set_ylabel('true label')

plt.show()
counvec=torch.zeros(6)
counvec2=torch.zeros(3, 6)
vec_res1=torch.tensor([[1,1,3,4,4,4,2,3,1,6]]).type(torch.LongTensor)
unk,cnum=torch.unique(vec_res1,return_counts=True)
inde_unk=unk-1
counvec[inde_unk]+=cnum
counvec2[1]+=counvec
np_coun2=counvec2.numpy()
#ssl_signal=[i for i in range(4)]
#random.shuffle(ssl_signal)
#print(ssl_signal)
mat1=torch.tensor([[1.0,1.0,2.,3.],[2.,3.,4.,2.]]).type(torch.float32)
mat2=torch.tensor([[2.,2.,2.,2.],[1.,1.,1.,1.]])
m_v_m=torch.mul(mat1,mat2)
los=torch.sum(m_v_m)

ensemble1=torch.tensor([[1,1,3,4]]).type(torch.LongTensor)
ensemble2=torch.tensor([[2,2,3,3]]).type(torch.LongTensor)
ensemble3=torch.tensor([[1,2,4,4]]).type(torch.LongTensor)
ensemble=torch.tensor([[1,1,3,4],[2,2,3,3],[1,2,4,4],[2,3,4,6]]).type(torch.float32)
div=ensemble/ensemble1.T
oppp=ensemble[2][3]
w,h=ensemble.size()
same_list=[]
for i_c in range(h):
    done_list = []
    for i_r in range(w):
        if ensemble[i_r][i_c] not in done_list:
            done_list.append(ensemble[i_r][i_c])
            num_non=len(done_list)
    same_list.append(num_non)
np_same=np.array(same_list)
mask_ok=np_same<3
oukankan=torch.unique_consecutive(ensemble,dim=0)
#outa,cou=torch.unique(ensemble,return_counts=True,dim=1)
res_en=torch.eq(ensemble1,ensemble2)




predicted_true_cls_test_pesudo_clssfer = torch.empty((0,)).type(torch.LongTensor)
tensor2= torch.empty((0,)).type(torch.LongTensor)
dasd=predicted_true_cls_test_pesudo_clssfer-tensor2
o=dasd.size()
num=torch.tensor(51).type(torch.LongTensor)
predicted_true_cls_test_pesudo_clssfer=torch.cat([predicted_true_cls_test_pesudo_clssfer, num.unsqueeze(0)], (0))
num2=torch.tensor(51).type(torch.LongTensor)
predicted_true_cls_test_pesudo_clssfer=torch.cat([predicted_true_cls_test_pesudo_clssfer, num2.unsqueeze(0)], (0))
mask51=predicted_true_cls_test_pesudo_clssfer==51
predicted_true_cls_test_pesudo_clssfer[mask51]=1
cluster_pesudo_label=np.load('./mediate/selfZSL/sslpretrained_FINCUB80ep vFINCUBbackBone  s01 Cluster 1000pesudo_label.npy')
CUB_pseudo_label=np.load('./mediate/selfZSL/sslpretrained_CUB80ep vCompareResnetCUBbackBone  s01 Cluster 1000pesudo_label.npy')
pesudo_label = torch.from_numpy(cluster_pesudo_label).type(torch.LongTensor)
CUB_label=torch.from_numpy(CUB_pseudo_label).type(torch.LongTensor)
CUB_unseen_set=CUB_label[-2967:]
unseen_set=pesudo_label[-6985:]
seen_set=pesudo_label[:30337]
num_unseen=unseen_set[unseen_set>39].size(0)
num_CUB_unseen=CUB_unseen_set[CUB_unseen_set>149].size(0)
num_err_seen=seen_set[seen_set>39].size(0)
lab=np.array([1,3,4,5,6,8])
lab_py=torch.from_numpy(lab)
lab_num=lab_py[-3:]

num=lab_py[lab_py>4].size(0)
forlabel=torch.empty((0,)).type(torch.LongTensor)
for i, index in enumerate(lab_py):
    forlabel = torch.cat([forlabel, lab_py[i].unsqueeze(0)],0)

#forlabel=torch.cat([forlabel, lab_py[index]], (0))
shuzu1 = np.array([[2, 3, 1], [1, 1, 1], [34, 2, 2]])
duizhao1 = np.array([1, 1, 1,3,5,6,6,8,8])
mask_than5=duizhao1>10
duizhao1[mask_than5]=51
tensor_shuzu1 = torch.tensor(shuzu1)
tensor_duizhao1 = torch.tensor(duizhao1)
outshu = []
#for i in range(shuzu1.shape[0]):
#    if not torch.all(torch.eq(tensor_duizhao1, tensor_shuzu1[i])):
#        print(shuzu1[i])
#        outshu.append(shuzu1[i])
#print(outshu)

mask_n=[0,3,4]
label1=['a','b','c','d',777]
np_label1=np.array(label1)
np_mask_n=np.array(mask_n)
new_lab1=np_label1[np_mask_n]
label2=['a','a','fafaf','c',456]
np_label1=np.array(label1)
np_label2=np.array(label2)
mask4=np_label1==np_label2
mask3=np_label2=='a'
label3=[False,False,1,2,False,4,False]
label3.append(False)
np_label3=np.array(label3)
py_label3=torch.from_numpy(np_label3)
mask2=np_label3==None
label4=[[1,2,4,5],[3,4,5,6],[0],[0]]
np_label4=np.array([1,2,3,4,5,6,7,50,50,50,50])
py_label4=torch.from_numpy(np_label4)
mask34=py_label4==50
mask=label2.index(456)
tag=label1[mask]
#ceshi=np.array([1,2,3,4,5,6,7])
#for i, item in enumerate(ceshi):
#    if i<3:
#        continue
#    print(i)
#    print(item)


all_seman_dict = np.load(seman_path + '/' + 'semantic' + str(dim) + '.npy', allow_pickle=True).item()
dict=all_seman_dict
data_path = os.path.abspath('../../dataset/GZSL_for_AWA/Animals_with_Attributes2/')
label_dict,label_name_list=f_s.true_label_list(data_path)
mask=label_name_list.index('zebra')
name1=label_name_list[mask]