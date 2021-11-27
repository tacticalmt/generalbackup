import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import *
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as tf
import os
#from torch.optim import RMSprop
from torchvision.utils import make_grid
from pylab import plt
# import Mod.res_block as resblock
import Libm.imporfunc as f_semantic

data_path = os.path.abspath('../../../dataset/GZSL_for_AWA/Animals_with_Attributes2/')
path_model = os.path.abspath('../../../dataset/wiki.en.text.model')
seman_path = os.path.abspath('../')
# vec_model=Word2Vec.load(path_model)#载入语义模型
bat_size = 20
img_size = 224
gpu = True
tr_epoch = 5
worker = 2
dim = 300
mode = 'gzsl'

transform1 = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])


# 数据集载入
class SeenData(Dataset):
    def __init__(self, data_root, seman_dict, transforms=None, data_type='training'):  # data_root为总文件夹路径
        if data_type == 'training':
            cls_type = 'trainclasses'
            category = 'trainingset'
        elif data_type == 'zsl':
            cls_type = 'testclasses'
            category = 'testingset'
        elif data_type == 'gzsl':
            cls_type = 'gzslclasses'
            category = 'testingset'
        else:
            print('no model type')

        word = f_semantic.get_name_list(data_root, cls_type)  # 地址为包含类名文件txt的文件夹的地址
        label_dict = f_semantic.get_dict_name(data_root)  #
        cls_dir, info = f_semantic.get_path_info(word, label_dict, data_root, category)  # 是读取训练还是测试集
        self.images_files, label_t, self.seman_list, self.label_train = f_semantic.data_pack(cls_dir, info, seman_dict)
        label_np = np.array(label_t)
        # print(label_t)
        # print(label_np)
        label_tensor = torch.from_numpy(label_np).type(torch.LongTensor)
        #
        # self.images_files=pic_path #图片目录
        self.transforms = transforms
        self.true_label = label_tensor

    def __getitem__(self, index):
        try:
            pic_data = self.transforms(Image.open(self.images_files[index]))

        except RuntimeError:
            print(self.images_files[index])
            return self.transforms(Image.open(self.images_files[index - 1])), self.label_train[index - 1], \
                   self.seman_list[index - 1], self.true_label[index - 1]  # 返回类名

        else:
            return pic_data, self.label_train[index], self.seman_list[index], self.true_label[index]

    def __len__(self):
        return len(self.images_files)


class MuiltClassData(Dataset):
    def __init__(self, data_root, seman_dict, cls_name, transforms=None, data_type='training'):
        self.root = data_root
        self.seman_dict = seman_dict
        self.tar_cls_name = cls_name
        if data_type == 'training':
            self.cls_type = 'trainclasses'
            self.category = 'trainingset'
        elif data_type == 'zsl':
            self.cls_type = 'testclasses'
            self.category = 'testingset'
        elif data_type == 'gzsl':
            self.cls_type = 'gzslclasses'
            self.category = 'testingset'
        else:
            print('no model type')
        # word = f_semantic.get_name_list(data_root, self.cls_type)#跟cls_name绑定
        label_dict = f_semantic.get_dict_name(data_root)  #
        cls_dir, info = f_semantic.get_path_info(self.tar_cls_name, label_dict, data_root, self.category)  # 是读取训练还是测试集
        self.images_files, label_t, self.seman_list, self.label_train = f_semantic.data_pack(cls_dir, info, seman_dict)
        label_np = np.array(label_t)
        label_tensor = torch.from_numpy(label_np).type(torch.LongTensor)
        self.transforms = transforms
        self.true_label = label_tensor

    def __getitem__(self, index):
        try:
            pic_data = self.transforms(Image.open(self.images_files[index]))

        except RuntimeError:
            print(self.images_files[index])
            return self.transforms(Image.open(self.images_files[index - 1])), self.label_train[index - 1], \
                   self.seman_list[index - 1], self.true_label[index - 1]  # 返回类名

        else:
            return pic_data, self.label_train[index], self.seman_list[index], self.true_label[index]

    def __len__(self):
        return len(self.images_files)


# 网络
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes, out_channels=places, kernel_size=7, stride=stride, padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )


class Bottleneck(nn.Module):
    def __init__(self, in_places, places, stride=1, downsampling=False, expansion=4):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places * self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places * self.expansion, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(places * self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, blocks, num_classes=1000, expansion=4):
        super(ResNet, self).__init__()
        self.expansion = expansion

        self.conv1 = Conv1(in_planes=3, places=64)
        self.layer1 = self.make_layer(in_places=64, places=64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places=256, places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512, places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024, places=512, block=blocks[3], stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        #self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places, stride, downsampling=True))
        for i in range(1, block):
            layers.append(Bottleneck(places * self.expansion, places))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


net = ResNet([3, 4, 23, 3], num_classes=40)

# 载入网络和通用变量
alex = AlexNet(40)
alex.load_state_dict(torch.load('../models/99seenCNNremakev10a.pkl'))
net.load_state_dict(torch.load('../models/v11143RetrainedseenResNeta.pkl'))

all_seman_dict = np.load(seman_path + '/' + 'semantic' + str(dim) + '.npy', allow_pickle=True).item()
# indata = SeenData(data_path, all_seman_dict, transform1, data_type='gzsl')  # 改zsl类型时要改
tar_name_list = 1  # 为全部测试集类名list
#cls = 0  # 为类名list
# indata=MuiltClassData(data_path,all_seman_dict,cls,transform1,data_type='gzsl')
# testingdata = DataLoader(indata, bat_size, shuffle=False, num_workers=worker)
if mode == 'gzsl':
    class_type = 'gzslclasses'
    loaddata_type = 'gzsl'
elif mode == 'zsl':
    class_type = 'testclasses'
    loaddata_type = 'zsl'
else:
    print('trianing mode')

if gpu:
    net.cuda()


tlabel_name_list = f_semantic.get_dict_name(data_path)
test_label_list = f_semantic.get_name_list(data_path)
seman_temp, _ = f_semantic.get_semantic(test_label_list, tlabel_name_list, all_seman_dict)  # 训练类标对应的semantic
seman_mat = torch.from_numpy(seman_temp).cuda()
all_test = f_semantic.get_name_list(data_path, cls=class_type)  # zsl的时候testclasses  gzsl为gzslclasses
all_seman, tl_index = f_semantic.get_semantic(all_test, tlabel_name_list, all_seman_dict)  # 测试用的全部的类标对应的semantic
# all_seman=torch.from_numpy(all_seman_temp)
#tar_name_list = f_semantic.get_name_list(data_path, class_type)

count_macc = 0
count_cls = 0
T = 10  # 取前几名
total_cls = 40
set_zero = total_cls - T  # 置零的位数
softm = nn.Softmax(dim=1)
for i_word, word in enumerate(all_test):  #改一下all test就好
    error_rate = 0
    total_samp = 0
    error_sam = 0
    indata = MuiltClassData(data_path, all_seman_dict, [word], transform1, data_type=loaddata_type)
    testingdata = DataLoader(indata, bat_size, shuffle=False, num_workers=worker)
    for i_batch, batch in enumerate(testingdata):
        data = batch[0].cuda()
        train_label = batch[1]
        semantic_label = batch[2]
        true_label = batch[3]
        true_sample_label = true_label.numpy()
        #    if gpu:
        #        data=data.cuda()
        #output = alex(data)
        output=net(data)
        soft_out = softm(output)
        # print(output)#缺softmax
        # print(soft_out)
        index_soft = torch.argsort(soft_out, dim=1)
        for i_samp in range(index_soft.size()[0]):
            for i_dim in range(set_zero):  # 把小的值都屏蔽了
                soft_out[i_samp, index_soft[i_samp, i_dim]] = 0
        # print(soft_out)
        result_semanti = torch.matmul(soft_out,
                                      seman_mat)  # output*semantic_vec#semantic_label是样本类标的semantic，每一列维度个数为训练时类别的个数，应该为bat_size*40  40* 300的两个矩阵

        ##########
        np_semantic = result_semanti.detach().cpu().numpy()
        loc_acc, correct_t, num_s_temp = f_semantic.zsl_el(all_seman, np_semantic, true_sample_label, tl_index,
                                                           p=1)  # 结果与全部的semantic做cos距离计算。
        # print(loc_acc)
        error_sam = error_sam + correct_t  # 计算累计错误样本
        total_samp = total_samp + num_s_temp  # 总样本个数
    acc = error_sam / total_samp
    count_macc = count_macc + acc
    count_cls = count_cls + 1

avg_acc = count_macc / count_cls
print(avg_acc)

