"""
Adopted from https://github.com/Megvii-Nanjing/BBN
Customized by Kaihua Tang
"""

import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import random
import os
import dill

class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, phase, imbalance_ratio, root = '../data/cifar_100/', imb_type='exp',
        test_imb_ratio=None, reverse=False):

        train = True if phase == "train" else False#phase指带是训练阶段还是测试阶段
        super(IMBALANCECIFAR10, self).__init__(root, train, transform=None, target_transform=None, download=True)#原始代码是download = False
        self.train = train
        if self.train:
            self.img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imbalance_ratio, reverse=reverse)#获得每个类的数量
            self.gen_imbalanced_data_train(self.img_num_list)#创建了不平衡的数据集
            # self.transform = transforms.Compose([
            #     transforms.RandomCrop(32, padding=4),
            #     transforms.RandomHorizontalFlip(),
            #     #transforms.Resize(224),
            #     transforms.ToTensor(),
            #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # ])
            self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
        else:
            if test_imb_ratio:
                # if test imbalance ratio is explicitly given, test dataset should be imbalanced.
                self.img_num_list = self.get_img_num_per_cls(
                    self.cls_num, imb_type, test_imb_ratio, reverse=reverse)
            else:
                self.img_num_list = self.get_img_num_per_cls(
                    self.cls_num, imb_type, imb_factor=imbalance_ratio, reverse=reverse)
            self.gen_imbalanced_data(self.img_num_list)
            self.transform = transforms.Compose([
                            #transforms.Resize(224),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                            ])
                            
        self.labels = self.targets  # self.targets is inherited from torchvision.datasets.CIFAR10
        print("{} Mode, {} images".format(phase, len(self.data)))
    
    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.get_annotations()):
            cat_id = anno["category_id"]
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)#返回的是cat_id类别图像的索引
        return class_dict

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor, reverse=False):#imb_type指的是指数型的不平衡还是阶梯型的不平衡
        img_max = len(self.data) / cls_num# 计算不平衡状态下每个类的数量
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                if reverse:
                    num = img_max * (imb_factor**((cls_num - 1 - cls_idx) / (cls_num - 1.0)))#**代表指数运算
                else:
                    num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':#前一半类具有较多的图片，后一半数量较少
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls,):
        new_data = []
        new_targets = []#类别标签
        targets_np = np.array(self.targets, dtype=np.int64)#self.targets是返回的类别数组举个例子=[1,2,5,6,7,7,8,2]
        classes = np.unique(targets_np)#去除重复的元素，返回值是从小到大排序

        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]#在原数据集中找出这个类的所有图片的下标索引
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]#取得了the_img_num个该类的索引
            new_data.append(self.data[selec_idx, ...])#选择了self.data中为selec_idx的图像，其中...表示取剩余维度(32*32*3)增加可读性
            new_targets.extend([the_class, ] * the_img_num)#假设类别是2,the_img_num = 5,那么new_targets = [2,2,2,2,2]
        new_data = np.vstack(new_data)#垂直堆叠，保证new_data是n*32*32*3
        self.data = new_data            # n*32*32*3
        self.targets = new_targets      # list of length n

    def gen_imbalanced_data_train(self, img_num_per_cls,):
        new_data = []
        new_targets = []#类别标签
        targets_np = np.array(self.targets, dtype=np.int64)#self.targets是返回的类别数组举个例子=[1,2,5,6,7,7,8,2]
        classes = np.unique(targets_np)#去除重复的元素，返回值是从小到大排序
        list_indice = []
        file_path = os.path.join("transfer_dataset", "cifar10-LT_index_list_seed7_trainingIF_0.02")
        with open(file_path, 'rb') as f:
            list_indice = dill.load(f)
    
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]#在原数据集中找出这个类的所有图片的下标索引
            np.random.shuffle(idx)
            #selec_idx = idx[:the_img_num]#取得了the_img_num个该类的索引
            selec_idx = list_indice[the_class]
            new_data.append(self.data[selec_idx, ...])#选择了self.data中为selec_idx的图像，其中...表示取剩余维度(32*32*3)增加可读性
            new_targets.extend([the_class, ] * the_img_num)#假设类别是2,the_img_num = 5,那么new_targets = [2,2,2,2,2]
        new_data = np.vstack(new_data)#垂直堆叠，保证new_data是n*32*32*3
        self.data = new_data            # n*32*32*3
        self.targets = new_targets      # list of length n
            

    def __getitem__(self, index):
        
        img, label = self.data[index], self.labels[index]
        # ensure consistency with all other datasets to return a PIL Image
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, index

    def __len__(self):
        return len(self.labels)

    def get_num_classes(self):
        return self.cls_num

    def get_annotations(self):
        annos = []
        for label in self.labels:
            annos.append({'category_id': int(label)})
        return annos

    def get_cls_num_list(self):#返回每个类的数量
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


class IMBALANCECIFAR100(IMBALANCECIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    cls_num = 100
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
