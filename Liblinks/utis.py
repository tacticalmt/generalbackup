import os
import numpy as np
import torch
from torch.nn import functional as F
from scipy.io import loadmat

# from scipy import linalg

def data_pack(path_list, class_info,
              seman_dict,class_label=None):  # 输入类名文件夹的地址和类名信息list，model是读取semantic的函数地址，函数内对每个文件夹的内容遍历，抽出jpg格式的样本  seman_dict是语义字典 4
    samp_path = []  # 样本的读取路径
    label_true = []  # 样本类标
    samp_seman = []  # 样本的语义内嵌
    label_train = []
    lenth_info = len(class_info[0])  # 每个样本包含多少个信息
    #print(len(path_list))
    # print(path_list)
    for i, class_root in enumerate(path_list):  # i为第i类
        for x in os.scandir(class_root):
            if x.name.endswith(".jpg"):
                if class_label is not None:
                    index_label=class_label
                else:
                    index_label=i
                samp_path.append(x.path)
                #try:
                float_label = float(class_info[i][0])  # 0为类标
                #except:
                #    print('第{0}条数据处理失败'.format(i))
                # float_label=float(class_info[i][0])#0为类标
                # print(float_label)
                label_true.append(float_label)
                samp_seman.append(seman_dict[class_info[i][1]])  # 1为类名字符串model[class_info[i][1]]
                label_train.append(index_label)
        #if i == (len(class_info) - 1):
        #    break
            # for i_info,info in enumerate(class_info[i]):
            #   a.append(info)
    return samp_path, label_true, samp_seman, label_train

def data_pack_multi(path_list, class_info,
              seman_dict,class_label):  # 输入类名文件夹的地址和类名信息list，model是读取semantic的函数地址，函数内对每个文件夹的内容遍历，抽出jpg格式的样本  seman_dict是语义字典 4
    samp_path = []  # 样本的读取路径
    label_true = []  # 样本类标
    samp_seman = []  # 样本的语义内嵌
    label_train = []
    lenth_info = len(class_info[0])  # 每个样本包含多少个信息
    print(len(path_list))
    # print(path_list)
    for i, class_root in enumerate(path_list):  # i为第i类
        for x in os.scandir(class_root):
            if x.name.endswith(".jpg"):
                samp_path.append(x.path)
                #try:
                float_label = float(class_info[class_label][0])  # 0为类标
                #except:
                #    print('第{0}条数据处理失败'.format(i))
                # float_label=float(class_info[i][0])#0为类标
                # print(float_label)
                label_true.append(float_label)
                samp_seman.append(seman_dict[class_info[class_label][1]])  # 1为类名字符串model[class_info[i][1]]
                label_train.append(class_label)
        #if i == (len(class_info) - 1):
        #    break
            # for i_info,info in enumerate(class_info[i]):
            #   a.append(info)
    return samp_path, label_true, samp_seman, label_train


def get_path_info(name_list, name_dict, root,
                  cate='trainingset'):  # namelist是类名的列表， name是字典，每个键值都是一个包含类标和类名的list  tar_path为文件夹地址 3  训练文件夹trainingset,测试集文件夹testingset
    path_dir = []
    cl_info = []
    done_list = []
    # print(len(path_dir))
    # print(name_list)
    tar_path = os.path.join(root, cate)
    for i, word in enumerate(name_list):
        # [path_dir.append(x.path) for x in os.scandir(tar_path) if x.name.endswith(word)]
        for x in os.scandir(tar_path):
            if x.name.endswith(word) and (word not in done_list):
                path_dir.append(x.path)
                done_list.append(word)
        cl_info.append(name_dict[word])
        # print(i)
        # print(word)
        if i == (len(name_list) - 1):
            break
        # print(path_dir)
    # print(name_list)
    # print(len(name_list))
    #print(len(path_dir))
    # print(path_dir)

    return path_dir, cl_info  # path_dir每个类的样本的文件夹路径，cl_info是对应的真实类标


def get_name_list(root, cls='trainclasses'):  # 在指定目录下，找指定文件名的地址    1 #cls可以是trainclasses 或者 testclasses或者gzslclasses
    if cls == 'trainclasses' or cls == 'testclasses' or cls == 'gzslclasses':
        print(cls)
    else:
        print('error type')
        return -1

    label_path = [x.path for x in os.scandir(root) if cls in x.path]  # cls指定的文件名目录里面找类名 搜索根目录下带有类标名的txt文档
    word = []
    for i, file_label in enumerate(label_path):
        f = open(file_label, 'r')
        while True:
            temp_w = f.readline().strip()
            if not temp_w:
                break
            word.append(temp_w)
        f.close()
    return word  # 返回不同的name list给get_path_info函数来控制gzsl还是zsl


def get_name_list_fea(root, cls='trainvalclasses'):
    if cls == 'trainvalclasses' or cls == 'testclasses' or cls == 'gzslclasses':
        print(cls)
    else:
        print('error type')
        return -1
    label_path = [x.path for x in os.scandir(root) if cls in x.path]  # cls指定的文件名目录里面找类名 搜索根目录下带有类标名的txt文档
    word = []
    for i, file_label in enumerate(label_path):
        f = open(file_label, 'r')
        while True:
            temp_w = f.readline().strip()
            if not temp_w:
                break
            word.append(temp_w)
        f.close()
    return word  # 返回不同的name list给get_path_info函数来控制gzsl还是zsl


def get_dict_name(root):  # 获取name字典  从包含类标和类名的文件获取  root为文件夹路径 每个词条里包含一个列表，这个列表包含了真实类标和对应的自然语言名字 2
    # 返回name
    label_path = os.path.join(root, 'classes.txt')
    dict_name = {}
    wordl_l = []
    cl_f = open(label_path, 'r')

    while True:
        temp_cl = cl_f.readline().strip()
        if not temp_cl:
            break

        wordl_l.append(temp_cl)
    cl_f.close()
    temp_label = []

    for key_word in wordl_l:
        # print(key_word)
        v = key_word.split('\t', 1)
        # print(v)
        # print(len(v))
        temp_label.append(v)

    # else:
    #   v=[key_word]
    #   print(v)

    for i_dict in temp_label:
        dict_name[i_dict[1]] = i_dict

    # print(temp_label)
    # print(label_dict)

    return dict_name

#获取label list
def true_label_list(root):
    label_path = os.path.join(root, 'classes.txt')
    dict_name = {}
    wordl_l = []
    true_name_list=[]
    cl_f = open(label_path, 'r')

    while True:
        temp_cl = cl_f.readline().strip()
        if not temp_cl:
            break

        wordl_l.append(temp_cl)
    cl_f.close()
    temp_label = []

    for key_word in wordl_l:
        # print(key_word)
        v = key_word.split('\t', 1)
        #v = key_word.split(' ')
        # print(v)
        # print(len(v))
        temp_label.append(v)

    # else:
    #   v=[key_word]
    #   print(v)

    for i_dict in temp_label:
        dict_name[i_dict[1]] = i_dict
        true_name_list.append(i_dict[1])


    # print(temp_label)
    # print(label_dict)

    return dict_name,true_name_list

def CUBtrue_label_list(root):
    label_path = os.path.join(root, 'classes.txt')
    dict_name = {}
    wordl_l = []
    true_name_list=[]
    cl_f = open(label_path, 'r')

    while True:
        temp_cl = cl_f.readline().strip()
        if not temp_cl:
            break

        wordl_l.append(temp_cl)
    cl_f.close()
    temp_label = []

    for key_word in wordl_l:
        # print(key_word)
        #v = key_word.split('\t', 1)
        v = key_word.split(' ')
        # print(v)
        # print(len(v))
        temp_label.append(v)

    # else:
    #   v=[key_word]
    #   print(v)

    for i_dict in temp_label:
        dict_name[i_dict[1]] = i_dict
        true_name_list.append(i_dict[1])


    # print(temp_label)
    # print(label_dict)

    return dict_name,true_name_list


def zsl_el(S, S_tesamp, label, true_label_index, p=5):  # S_tesamp是测试样本(一行一个样本)，跟每个类的semantic算cosine距离,label是真实类标向量
    n = 0
    # print(S.shape)
    # print(S_tesamp.shape)
    dist_cos = np.dot(S_tesamp, S.T) / (np.linalg.norm(S) * np.linalg.norm(S_tesamp))
    # print(dist_cos)
    hit_mat = np.zeros([S_tesamp.shape[0], p])  # 前p命中矩阵
    test_label_mat = np.zeros([S_tesamp.shape[0], p])
    sort_mat = np.argsort(-dist_cos, axis=1)
    for i in range(S_tesamp.shape[0]):
        hit_mat[i, :] = sort_mat[i, 0:p]  # 对应现在训练类标的索引，需要转化成对应的真实类标
        hit_mat = hit_mat.astype(int)
    # print(hit_mat)
    for i in range(S_tesamp.shape[0]):
        for l_i in range(p):
            test_label_mat[i, l_i] = true_label_index[hit_mat[i, l_i]]  # true label是tensor
    test_label_mat = test_label_mat.astype(int)
    # print(test_label_mat)
    # print(label)
    for i in range(S_tesamp.shape[0]):
        # print(test_label_mat[i,:])
        # print(label[i])
        if label[i] in test_label_mat[i, :]:
            n = n + 1
            # print(n)

    acc = n / S_tesamp.shape[0]
    return acc, n, S_tesamp.shape[0]


def get_semantic(word_list, name_dict, semantic_dict):  # 给出对应的类标名字，和语义字典，输出语义的列表
    temp = []
    name_list = []
    for word in word_list:
        temp.append(semantic_dict[word])
        temp_name = int(name_dict[word][0])
        name_list.append(temp_name)
    semantic_mat = np.array(temp)
    true_label_index = np.array(name_list)
    return semantic_mat, true_label_index


def get_attri_label(root):  #读取attribute list  先用这个1
    a_path = os.path.join(root, 'predicate-matrix-binary.txt')
    word_att = []
    att = open(a_path, 'r')  # 打开attribute文件
    while True:
        temp_att = att.readline().strip()
        if not temp_att:
            break
        word_att.append(temp_att)
    att.close()
    divided_att = []
    for key_word in word_att:
        v = key_word.split(' ')
        divided_att.append(v)
    divided_np = np.array(divided_att)
    attri_list = divided_np.astype(int)
    return attri_list

def get_CUBattri_label(root):  #读取attribute list  先用这个1
    a_path = os.path.join(root, 'attributes/class_attribute_labels_continuous.txt')
    word_att = []
    att = open(a_path, 'r')  # 打开attribute文件
    while True:
        temp_att = att.readline().strip()
        if not temp_att:
            break
        word_att.append(temp_att)
    att.close()
    divided_att = []
    for key_word in word_att:
        v = key_word.split(' ')
        divided_att.append(v)
    divided_np = np.array(divided_att)
    attri_list = divided_np.astype(float)
    return attri_list


def create_name_to_attri(name_list, attri_list):  # name list要包含全部类名的lists,返回一个包含全部类的attribute的字典     这个构造属性prototype2
    attri_dict = {}
    attribute_list=[]
    true_label=[]
    for i, word_i in enumerate(name_list):
        attri_dict[word_i] = attri_list[i]
        attribute_list.append(attri_list[i])
        true_label.append(i+1)
    return name_list,attri_dict,attribute_list,true_label

#------------------------------------------------------
#------------------------------------------------------
#------------------

def create_attri_semantic_prototype(root,name_list):  #root为类标文件地址
    att_list=get_attri_label(root)
    cls_name_word,attri_prototype_dict,attri_prototype_list,true_label=create_name_to_attri(name_list,att_list)
    return cls_name_word,attri_prototype_dict,attri_prototype_list,true_label



def create_embed_semantic_prototype(root,name_list):  #root为semantic文件地址
    semantic_dict=np.load(root, allow_pickle=True).item()
    true_label=[]
    semantic_prototype_list=[]
    for i,word in enumerate(name_list):
        semantic_prototype_list.append(semantic_dict[word])
        true_label.append(i + 1)
    return name_list,semantic_dict,semantic_prototype_list,true_label


def create_CUBattri_semantic_prototype(root,name_list):  #root为类标文件地址
    tar_path = root
    mat_root = os.path.join(tar_path, 'res101.mat')
    mask_info_root = os.path.join(tar_path, 'att_splits.mat')
    mask_info = loadmat(mask_info_root)
    mat_in = loadmat(mat_root)
    matdata_label = np.squeeze(mat_in['labels'])
    true_label=np.unique(matdata_label)

    training_mask = np.squeeze(mask_info['trainval_loc'] - 1)
    test_seen_mask = np.squeeze(mask_info['test_seen_loc'] - 1)
    test_unseen_mask = np.squeeze(mask_info['test_unseen_loc'] - 1)
    all_cls_name = np.squeeze(mask_info['allclasses_names'])
    att_mat = np.squeeze(mask_info['original_att'].T)
    attri_prototype_list=att_mat
    cls_name_word=all_cls_name
    attri_prototype_dict=mask_info

    #att_list=get_CUBattri_label(root)
    #cls_name_word,attri_prototype_dict,attri_prototype_list,true_label=create_name_to_attri(name_list,att_list)
    return cls_name_word,attri_prototype_dict,attri_prototype_list,true_label

#------------------
#------------------------------------------------------
#------------------------------------------------------


def data_pack_attr(path_list, class_info, attri_dict):
    # 获取属性类标
    samp_path = []  # 样本的读取路径
    label_true = []  # 样本类标
    attri_label = []  # 训练时用的类标
    lenth_info = len(class_info[0])  # 每个样本包含多少个信息
    label_train = []
    # print(len(path_list))
    for i, class_root in enumerate(path_list):  # i为第i类
        for x in os.scandir(class_root):
            if x.name.endswith(".jpg"):
                samp_path.append(x.path)
                try:
                    float_label = float(class_info[i][0])  # 0为类标
                except:
                    print('第{0}条数据处理失败'.format(i))
                # float_label=float(class_info[i][0])#0为类标
                # print(float_label)
                # print(class_info[i][0])
                print(float_label)
                label_true.append(float_label)
                # print(attri_dict[class_info[i][1]])
                attri_label.append(attri_dict[class_info[i][1]])  # 1为类名字符串model[class_info[i][1]]
                label_train.append(i)  # softmax层用的类标,embed层也要用
        if i == (len(class_info) - 1):
            break
    return samp_path, label_true, attri_label, label_train


def max_singular_value(W, u=None, Ip=1):  # IP为迭代次数   W为tensor
    if not Ip >= 1:
        raise ValueError("The number of power iterations should be positive integer")
    with torch.no_grad():
        if u is None:
            u = F.normalize(torch.ones(()).new_empty(W.size(0)).normal_(0, 1), dim=0, eps=1e-12)
        _u = u
        for _ in range(Ip):
            _v = F.normalize(torch.mv(W.t(), _u), dim=0, eps=1e-12)
            _u = F.normalize(torch.mv(W, _v), dim=0, eps=1e-12)
    sigma = torch.dot(_u, torch.mv(W, _v))

    # W_np = W.detach().cpu().numpy()
    # if u is None:
    #    u = np.random.normal(size=(1, W.shape[0])).astype(np.float32)
    # _u = u
    # for _ in range(Ip):
    #    _v = _l2normalize(np.dot(_u, W_np), eps=1e-12)
    #    _u = _l2normalize(np.dot(_v, W_np.T), eps=1e-12)

    # sigma = np.dot(_u, np.dot(W_np, _v.T)).sum()
    return sigma, _u, _v


def _l2normalize(vect, eps=1e-12):  # 1e-12=1*10^(-12)
    l2norm = np.linalg.norm(vect)
    return vect / (l2norm + eps)


def get_sample_label(name_dict, tar_list):  # name_dict存储类标与类名对应关系，tar_list存储对应训练集或者测试集类名，返回类标数组
    obj_label_list = []

    for key_word in tar_list:
        temp = int(name_dict[key_word][0])
        obj_label_list.append(temp)

    return obj_label_list


def get_tar_set(tar_class_index, all_labels, all_samples, seman_list, seman_index):  # sample都传tensor
    tar_label = []
    tar_data = []
    tar_seman = []
    for i_label, label in enumerate(all_labels):
        if label in tar_class_index:
            index_sm = np.argwhere(seman_index == label)  # 用于找内嵌的索引 np.argwhere
            tar_seman.append(seman_list[index_sm[0][0]])
            tar_label.append(label)
            tar_data.append((all_samples[i_label, :]))  # 取出对应位置的样本
    # np_tar_data = np.array(tar_data)
    np_tar_seman = np.array(tar_seman)
    np_tar_label = np.array(tar_label)
    return tar_data, np_tar_seman, np_tar_label


def get_tar_set_np(tar_class_index, all_labels, all_samples, seman_feature, seman_index,train_ratio=0.8):
    tar_label = []
    tar_data = []
    tar_seman = []
    test_data=[]
    test_seman=[]
    test_label=[]
    check_list=[]
    samp_count = 0
    train_num=0
    for i_label, label in enumerate(all_labels):
        if label in tar_class_index:
            if label not in check_list: #判断是否是第一次见这个类
                check_list.append(label)
                num_sample=countX(all_labels,label)
                train_num=int(num_sample*train_ratio)
                samp_count=0
            index_sm = np.argwhere(seman_index == label)  # 用于找内嵌的索引 np.argwhere
            samp_count = samp_count + 1
            if samp_count<=train_num:
                tar_seman.append(seman_feature[index_sm[0][0]])
                tar_label.append(label)
                tar_data.append((all_samples[i_label, :]))  # 取出对应位置的样本
            else:
                test_seman.append(seman_feature[index_sm[0][0]])
                test_label.append(label)
                test_data.append((all_samples[i_label, :]))



    np_tar_data = np.array(tar_data)
    np_tar_seman = np.array(tar_seman)
    np_tar_label = np.array(tar_label)
    np_test_data=np.array(test_data)
    np_test_seman=np.array(test_seman)
    np_test_label=np.array(test_label)

    return np_tar_data, np_tar_seman, np_tar_label,np_test_data,np_test_seman,np_test_label


def gen_seman_label(label_dict,label_name,bat_size,label_dim):#label_dim语义类标的长度  #需要类标名字和类标字典
    dict_len=len(label_dict)
    name_len=len(label_name)
    label_mat=np.zeros((bat_size,label_dim))
    for i_bat in range(bat_size):
        random_index=np.random.randint(name_len)
        label_mat[i_bat,:]=label_dict[label_name[random_index]]

    return np.array(label_mat)

#def countX(lst, x):
#    return lst.count(x)
def countX(lst, x):
    count = 0
    for ele in lst:
        if (ele == x):
            count = count + 1
    return count

#获取测试类的prototype
def get_test_prototype_list(target_list,all_cls_name_list,all_semantic_prototype_matrix):
    target_semantic_matrix=[]
    target_true_label=[]
    for i,word in enumerate(target_list):
        mask=all_cls_name_list.index(word)
        target_semantic_matrix.append(all_semantic_prototype_matrix[mask])
        target_true_label.append(mask+1)

    return  target_semantic_matrix,target_true_label

def get_ps_test_prototype_list(target_list,all_cls_name_list,all_semantic_prototype_matrix,true_label_list):
    target_semantic_matrix=[]
    target_true_label=[]
    for i,word in enumerate(target_list):
        mask=all_cls_name_list==word
        target_semantic_matrix.append(all_semantic_prototype_matrix[mask])
        target_true_label.append(true_label_list[mask])

    return  target_semantic_matrix,target_true_label

#获取训练类与真是类标对照表
def get_refer_table(ref_word,all_word,true_label):
    mask_list=[]
    for key_word in ref_word:
        mask=all_word.index(key_word)
        mask_list.append(mask)
    np_mask_list=np.array(mask_list)
    np_true_labl=np.array(true_label)
    ref_table=np_true_labl[np_mask_list]
    return ref_table

def get_ps_refer_table(ref_word,all_word,true_label):
    mask_list=[]
    ref_table=[]
    for key_word in ref_word:
        mask=all_word==key_word
        ref_table.append(true_label[mask])
    #np_mask_list=np.array(mask_list)
    #np_true_labl=np.array(true_label)
    #ref_table=np_true_labl[np_mask_list]
    ref_table=np.squeeze(ref_table)
    return ref_table



#获取训练样本
def get_training_sample(data_root,seen_ref_list,all_cls_name_list,semantic_prototype_mat,true_label_list,train_data=True):
    if train_data == True:
        dir_class = 'trainingset'
        # root_label=os.path.abspath('../dataset/Animals_with_Attributes2')
    else:
        dir_class = 'testingset'
    path_dir = []
    mask_list=[]
    done_list = []
    for key_word in seen_ref_list:
        mask = all_cls_name_list.index(key_word)
        for x in os.scandir(data_root + '/' + dir_class + '/'):
            if x.name.endswith(key_word) and (key_word not in done_list):
                path_dir.append(x.path)
                mask_list.append(mask)
                done_list.append(key_word)
        # [path_dir.append(x.path) for x in os.scandir(root + '/' + dir_class + '/') if x.name.endswith(key_word)]

    pic_path = []
    semantic_label=[]
    sam_true_label=[]
    label_t = []

    for l, train_root in enumerate(path_dir):  # 对训练集文件夹遍历
        # [pic_path.append(x.path) for x in os.scandir(train_root) if x.name.endswith(".jpg")]#每个文件夹里的图片地址
        mask_for_label=mask_list[l]
        for x in os.scandir(train_root):
            if x.name.endswith(".jpg"):
                pic_path.append(x.path)  #data path
                semantic_label.append(semantic_prototype_mat[mask_for_label])# semantic label
                sam_true_label.append(true_label_list[mask_for_label])#true label
                label_t.append(l)                                     #training label
                
    return pic_path,semantic_label,sam_true_label,label_t

"""
accul_sample 计算累计样本个数，确定返回的样本类标
accul_folder 计算累计扫描过的文件夹个数
"""
def obtain_samples(data_root,tar_cls_ref_list,all_cls_name_list,semantic_prototype_mat,true_label_list,data_folder=None,label_folder_len=None,accul_sample=0,accul_folder=0,class_termin_flag=None):
    dir_cls=data_folder
    path_dir = []
    mask_list = []
    done_list = []
    for key_word in tar_cls_ref_list:
        mask = all_cls_name_list.index(key_word)
        for x in os.scandir(data_root + '/' + dir_cls + '/'):
            if x.name.endswith(key_word) and (key_word not in done_list):
                path_dir.append(x.path)
                mask_list.append(mask)
                done_list.append(key_word)
    pic_path = []
    semantic_label = []
    sam_true_label = []
    label_t = []
    for l, train_root in enumerate(path_dir):  # 对训练集文件夹遍历
        if accul_folder<(label_folder_len+1):
            class_termin_flag.append(accul_sample)
        mask_for_label=mask_list[l]
        for x in os.scandir(train_root):
            if x.name.endswith(".jpg"):
                pic_path.append(x.path)  #data path
                sam_true_label.append(true_label_list[mask_for_label])
                semantic_label.append(semantic_prototype_mat[mask_for_label])
                accul_sample=accul_sample+1
                if accul_folder<label_folder_len:
                    label_t.append(accul_folder)  #训练用类标
                else:
                    label_t.append(50)  #labelless label
        accul_folder = accul_folder + 1

                #semantic_label.append(semantic_prototype_mat[mask_for_label])# semantic label
                #sam_true_label.append(true_label_list[mask_for_label])#true label
                #label_t.append(l)                                     #training label



    return pic_path,semantic_label,sam_true_label,label_t,accul_sample,accul_folder,class_termin_flag


def obtain_ps_samples(data_root,ps_data_root,data_set,training_data=True,Test_seen=False,Test_unseen=False):
    #dir_cls=data_folder
    pic_path = []
    semantic_label = []
    sam_true_label = []
    label_t = []
    class_termin_flag=[]
    tar_list=[]
    tar_path = ps_data_root
    mat_root = os.path.join(tar_path, 'res101.mat')
    mask_info_root = os.path.join(tar_path, 'att_splits.mat')
    mask_info = loadmat(mask_info_root)
    mat_in = loadmat(mat_root)
    img_path = np.squeeze(mat_in['image_files'])
    matdata_label = np.squeeze(mat_in['labels'])
    training_mask = np.squeeze(mask_info['trainval_loc'] - 1)
    test_seen_mask = np.squeeze(mask_info['test_seen_loc'] - 1)
    test_unseen_mask = np.squeeze(mask_info['test_unseen_loc'] - 1)
    all_cls_name = np.squeeze(mask_info['allclasses_names'])
    att_mat = np.squeeze(mask_info['original_att'].T)
    training_num=matdata_label[training_mask].size
    cls_size=np.unique(matdata_label).size
    training_label = matdata_label[training_mask]
    training_img=img_path[training_mask]
    seen_label = matdata_label[test_seen_mask]
    unseen_label = matdata_label[test_unseen_mask]
    if data_set=='CUB':
        delete_tar_str = '/BS/Deep_Fragments/work/MSc/CUB_200_2011/'
        #tar_path='/home/zhaozhi/dataset/xlsa17/data/CUB'
    elif data_set=='SUN':
        delete_tar_str='/BS/Deep_Fragments/work/MSc/data/SUN/'
        #tar_path = '/home/zhaozhi/dataset/xlsa17/data/SUN'
    else:
        delete_tar_str=None
        #tar_path = None
    acc_instance=0
    acc_cls=0
    seen_training_true_label = np.unique(training_label)
    seen_cls_size=seen_training_true_label.size
    unseen_true_label = np.unique(unseen_label)
    if not training_data and not Test_unseen and not Test_seen:
        print('error')
        tar_list.append(-1)
    if training_data:
        tar_list.append(training_mask)
    if Test_seen:
        tar_list.append(test_seen_mask)
    if Test_unseen:
        tar_list.append(test_unseen_mask)


    #tar_list=[training_mask,test_seen_mask,test_unseen_mask]#重新构造
    for i_mask in tar_list:
        tar_img=img_path[i_mask]
        tar_true_label=matdata_label[i_mask]
        i_tar_true_label = np.unique(tar_true_label)
        for i,i_label in enumerate(i_tar_true_label):
            if acc_cls<(seen_cls_size+1):
                class_termin_flag.append(acc_instance)
            mask=tar_true_label==i_label
            cls_img=tar_img[mask]
            for i_path in cls_img:
                cut_path=np.char.replace(i_path,delete_tar_str,"")
                temp_path = os.path.join(data_root, cut_path[0])
                pic_path.append(temp_path)
                semantic_label.append(att_mat[i_label-1])
                sam_true_label.append(i_label)
                if acc_cls<seen_cls_size:
                    label_t.append(i)
                else:
                    label_t.append(cls_size)
                acc_instance+=1
            acc_cls+=1
    class_termin_flag.append(acc_instance)
    #for i_path in img_path:
    #    cut_path=np.char.replace(i_path,delete_tar_str,"")
        #i_path.char.replace(delete_tar_str,"",1)
    #    temp_path=os.path.join(data_root, cut_path[0])
    #    pic_path.append(temp_path)




    return pic_path,semantic_label,sam_true_label,label_t,class_termin_flag

#def onesampdist(x,semantic_prototype):
#    y=x
class dist_cal(torch.nn.Module):#用于输出batch中每个样本与prototype的距离矩阵
    def __init__(self,semantic_prototype_mat,gpu=True,cls_num=50):
        super(dist_cal, self).__init__()
        self.protomat=semantic_prototype_mat
        self.gpu=gpu
        self.cls_num=cls_num

    def getonedist(self,x):
        dist = torch.sum(torch.mul(x - self.protomat, x - self.protomat), (1))
        return dist

    def getalldist(self,x):
        bat_size=x.size(0)
        dists_mat = torch.empty((0, self.cls_num))
        if self.gpu:
            dists_mat=dists_mat.cuda()
        for i in range(bat_size):
            dist_vec=self.getonedist(x[i])
            dists_mat = torch.cat([dists_mat, dist_vec.unsqueeze(0)], (0))
        return dists_mat

    def forward(self,x):
        dist_mat=self.getalldist(x)
        return dist_mat



class entropyCal(torch.nn.Module):
    def __init__(self):
        super(entropyCal, self).__init__()
        self.e_mat=None
        self.frac=None
        self.softmax=torch.nn.Softmax(dim=1)
        self.logsoftmax=torch.nn.LogSoftmax(dim=1)
    def forward(self,x):
        q_mat=self.softmax(x)
        #q_mat_test=torch.sum(q_mat,dim=1)
        log_q_mat=self.logsoftmax(x)
        entropy_mat=torch.mul(q_mat,log_q_mat)
        entropy_loss=-torch.sum(entropy_mat)
        return entropy_loss

def stableexp(x):
    return torch.where(x < 50,torch.exp(x),x)

def log1pexp(x):
    # more stable version of log(1 + exp(x))
    return torch.where(x < 50, torch.log1p(torch.exp(x)), x)

def get_CUBname_list(data_root,data_set='CUB',cls_type='seen'):  # 在指定目录下，找指定文件名的地址    1 #cls可以是trainclasses 或者 testclasses或者gzslclasses
    if cls_type=='seen':
        cls_signal=1
    else:
        cls_signal=0
    tar_path=data_root

    if data_set=='CUB':
        delete_tar_str = '/BS/Deep_Fragments/work/MSc/CUB_200_2011/'
        #tar_path='/home/zhaozhi/dataset/xlsa17/data/CUB'
    elif data_set=='SUN':
        delete_tar_str='/BS/Deep_Fragments/work/MSc/data/SUN/'
        #tar_path = '/home/zhaozhi/dataset/xlsa17/data/SUN'
    else:
        delete_tar_str=None
        #tar_path = None

    mat_root = os.path.join(tar_path, 'res101.mat')
    mask_info_root = os.path.join(tar_path, 'att_splits.mat')
    mask_info = loadmat(mask_info_root)
    mat_in = loadmat(mat_root)
    img_path = mat_in['image_files']
    matdata_label = np.squeeze(mat_in['labels'])

    training_mask = np.squeeze(mask_info['trainval_loc']-1)
    test_seen_mask = np.squeeze(mask_info['test_seen_loc']-1)
    test_unseen_mask = np.squeeze(mask_info['test_unseen_loc']-1)
    all_cls_name = np.squeeze(mask_info['allclasses_names'])
    att_mat = mask_info['original_att']

    training_label=matdata_label[training_mask]
    seen_label=matdata_label[test_seen_mask]
    unseen_label=matdata_label[test_unseen_mask]

    seen_true_label=np.unique(seen_label)
    seen_training_true_label=np.unique(training_label)
    unseen_true_label=np.unique(unseen_label)

    if cls_signal==1:
        tar_label=seen_true_label
    else:
        tar_label=unseen_true_label

    mask_cls_name=tar_label-1
    tar_cls_name=all_cls_name[mask_cls_name]

    return tar_cls_name  # 返回不同的name list给get_path_info函数来控制gzsl还是zsl

def get_ps_true_label_list(tar_path):
    mat_root = os.path.join(tar_path, 'res101.mat')
    mask_info_root = os.path.join(tar_path, 'att_splits.mat')
    mask_info = loadmat(mask_info_root)
    mat_in = loadmat(mat_root)
    img_path = mat_in['image_files']
    matdata_label = mat_in['labels']
    training_mask = mask_info['trainval_loc']
    test_seen_mask = mask_info['test_seen_loc']
    test_unseen_mask = mask_info['test_unseen_loc']
    all_cls_name = np.squeeze(mask_info['allclasses_names'])
    att_mat = mask_info['original_att']
    return  all_cls_name

#tar_str='/BS/Deep_Fragments/work/MSc/CUB_200_2011/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0081_426.jpg'
#tar_path=os.path.abspath('../../dataset/CUB/CUB_200_2011/')
#delete_tar_str='/BS/Deep_Fragments/work/MSc/CUB_200_2011/'
#delete_tarSUN_str='/BS/Deep_Fragments/work/MSc/data/SUN/'
#tarSUN_str='/BS/Deep_Fragments/work/MSc/data/SUN/images/a/abbey/sun_abegcweqnetpdlrh.jpg'
#mat_root = os.path.join(tar_path, 'res101.mat')
#mask_info_root=os.path.join(tar_path, 'att_splits.mat')
#mask_info=loadmat(mask_info_root)
#mat_in = loadmat(mat_root)
#img_path=mat_in['image_files']
#matdata_label = mat_in['labels']
#training_mask=mask_info['trainval_loc']
#test_seen_mask=mask_info['test_seen_loc']
#test_unseen_mask=mask_info['test_unseen_loc']
#all_cls_name=mask_info['allclasses_names']
#att_mat=mask_info['original_attc']