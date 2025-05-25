import numpy as np
from medmnist import TissueMNIST
import torch
import random
import os
import dill
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
class IMBALANCETissueMNIST(TissueMNIST):
    cls_num = 8  # TissueMnisT有8类
    original_training_counts = [
        53075,  # 类别0
        7814,  # 类别1
        5866,  # 类别2
        15406,  # 类别3
        11789,  # 类别4
        7705,  # 类别5
        39203,  # 类别6
        24608,  # 类别7
    ]
    original_test_counts = [
        15165,  # 类别0
        2233,  # 类别1
        1677,  # 类别2
        4402,  # 类别3
        3369,  # 类别4
        2202,  # 类别5
        11201,  # 类别6
        7031,  # 类别7
    ]
    
    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, split='train',
                 transform=None, target_transform=None, download=False, reverse=False):
        super(IMBALANCETissueMNIST, self).__init__(root=root, split=split, transform=transform, target_transform=target_transform, download=download)
        np.random.seed(rand_number)
        self.rank = sorted(range(len(self.original_training_counts)), 
                      key=lambda i: self.original_training_counts[i], 
                      reverse=True)
        if split == 'train':
            img_max = 53075
            self.img_num_list = self.get_img_num_per_cls(self.cls_num, img_max, imb_type, imb_factor, reverse)
            self.gen_imbalanced_data_train(self.img_num_list)  # 使用train专用方法
            self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=(0.5), std=(0.5))
            ])
        else :
            img_max = 1677
            self.img_num_list = self.get_img_num_per_cls(self.cls_num, img_max, imb_type, imb_factor, reverse)
            self.gen_imbalanced_data(self.img_num_list)
            self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=(0.5), std=(0.5))
            ])
        
        # 创建新的 info 字典
        self.info = {
            'n_samples': {
                'train': len(self.imgs),
                'val': len(self.imgs),
                'test': len(self.imgs)
            }
        }
        
        # self.gen_imbalanced_data(img_num_list)
        self.reverse = reverse
        self.data = self.imgs
        self.targets = self.labels.squeeze().tolist() if isinstance(self.labels, np.ndarray) else self.labels


    def gen_imbalanced_data_train(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.labels, dtype=np.int64).squeeze()
        classes = np.unique(targets_np)
        list_indice = []
        
        # 尝试加载预定义的索引列表
        file_path = os.path.join("transfer_dataset", "TissueMNIST-LT_index_list_seed7_trainingIF_0.02")
        try:
            with open(file_path, 'rb') as f:
                list_indice = dill.load(f)
        except:
            print("Warning: Could not load predefined indices, using random sampling instead")
            list_indice = None

        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            
            if list_indice is not None and the_class in list_indice:
                selec_idx = list_indice[the_class]
            else:
                selec_idx = idx[:the_img_num]
                
            new_data.append(self.imgs[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
            
        new_data = np.vstack(new_data)
        self.imgs = new_data
        self.labels = np.array(new_targets).reshape(-1, 1)

    def get_img_num_per_cls(self, cls_num,img_max, imb_type, imb_factor, reverse):
        
        img_num_per_cls = []
        print("get_img_num_per_clas:{}".format(imb_type))
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                if reverse:
                    num = img_max * (imb_factor ** ((cls_num - 1 - cls_idx) / (cls_num - 1.0)))
                    img_num_per_cls.append(int(num))
                else:
                    num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                    img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2, cls_num):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
            
        final_img_num_per_cls = [None] * cls_num
        for i in range(len(self.rank)):
            final_img_num_per_cls[self.rank[i]] = (img_num_per_cls[i])
        return final_img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.labels, dtype=np.int64).squeeze()
        classes = np.unique(targets_np)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.imgs[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.imgs = new_data
        self.labels = np.array(new_targets).reshape(-1, 1)
        
        # 更新 info 字典
        self.info['n_samples'][self.split] = len(self.imgs)
        print(f"Updated info['n_samples'][{self.split}] to {len(self.imgs)}")

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict.get(i, 0))
        return cls_num_list

if __name__ == '__main__':
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    trainset = IMBALANCETissueMNIST(root='datasets/TissueMnist', split='train', download=True, transform=transform)
    print(trainset.get_cls_num_list())