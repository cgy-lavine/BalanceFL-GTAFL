import numpy as np
from medmnist import PathMNIST
import torch
import random
import os
import dill
from PIL import Image
from torchvision import transforms
class IMBALANCEPATHMNIST(PathMNIST):
    cls_num = 9  # PathMNIST有9类
    original_training_counts = [
        9366,  # 类别0
        9509,  # 类别1
        10360,  # 类别2
        10401,  # 类别3
        8006,  # 类别4
        12182,  # 类别5
        7886,  # 类别6
        9401,  # 类别7
        12885   # 类别8
    ]
    original_test_counts = [
        1338,  # 类别0
        847,  # 类别1
        339,  # 类别2
        634,  # 类别3
        1035,  # 类别4
        592,  # 类别5
        741,  # 类别6
        421,  # 类别7
        1233   # 类别8
    ]
    
    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, split='train',
                 transform=None, target_transform=None, download=False, reverse=False):
        super(IMBALANCEPATHMNIST, self).__init__(root=root, split=split, transform=transform, target_transform=target_transform, download=download, as_rgb=True)
        np.random.seed(rand_number)
        self.rank = sorted(range(len(self.original_training_counts)), 
                      key=lambda i: self.original_training_counts[i], 
                      reverse=True)
        if split == 'train':
            img_max = 12885
            self.img_num_list = self.get_img_num_per_cls(self.cls_num, img_max, imb_type, imb_factor, reverse)
            self.gen_imbalanced_data_train(self.img_num_list)  # 使用train专用方法
            self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
            ])
        else :
            img_max = 339
            self.img_num_list = self.get_img_num_per_cls(self.cls_num, img_max, imb_type, imb_factor, reverse)
            self.gen_imbalanced_data(self.img_num_list)
            self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
            ])
        # 创建新的 info 字典
        self.info = {
            'n_samples': {
                'train': len(self.imgs),
                'val': len(self.imgs),
                'test': len(self.imgs)
            }
        }
        self.reverse = reverse
        self.data = self.imgs
        self.targets = self.labels.squeeze().tolist() if isinstance(self.labels, np.ndarray) else self.labels

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

    def gen_imbalanced_data_train(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.labels, dtype=np.int64).squeeze()
        classes = np.unique(targets_np)
        list_indice = []
        
        # 尝试加载预定义的索引列表
        file_path = os.path.join("transfer_dataset", "Pathmnist-LT_index_list_seed7_trainingIF_0.02")
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

    def __getitem__(self, index):
        img, label = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label, index
    def __len__(self):
        return len(self.targets)

    def get_num_classes(self):
        return self.cls_num

    def get_annotations(self):
        annos = []
        for label in self.targets:
            annos.append({'category_id': int(label)})
        return annos
    
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict.get(i, 0))
        return cls_num_list

if __name__ == '__main__':
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    trainset = IMBALANCEPATHMNIST(root='./data/pathmnist', split='train', download=True, transform=transform)
    print(trainset.get_cls_num_list())