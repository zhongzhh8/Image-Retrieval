import time
from functools import wraps
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets.cifar import CIFAR10
import torchvision.transforms as transforms
import itertools
import torch.nn.functional as F
from PIL import Image
import torch.utils.data as torchdata
from torch.utils.data.sampler import Sampler
import random
import os
import logging
from logging.handlers import TimedRotatingFileHandler


def default_loader(path):
    return Image.open(path).convert('RGB')

class MyDataset(torchdata.Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            if line == '':
                break
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split(' ')
            imgs.append(words)
        self.imgs = imgs


    def __getitem__(self, index):
        words = self.imgs[index]
        img = self.loader('/home/disk1/yangyifan/cub/CUB_200_2011/images/'+words[0])
        # img = self.loader('../../disk2/yangyifan/cifar10/'+words[0])
        if self.transform is not None:
            img = self.transform(img)
        label = int(words[1])
        return img,label

    def __len__(self):
        return len(self.imgs)

class categoryRandomSampler_CUB(Sampler):
    def __init__(self, numBatchCategory, targets, batch_size):
        """
        This sampler will sample numBatchCategory categories in each batch.
        """
        self.targets = list(targets)
        self.batch_size = batch_size
        self.num_samples = len(targets)
        self.numBatchCategory = numBatchCategory
        self.num_categories = max(targets)
        self.category_idxs = {}
        self.categorys = list(range(1, self.num_categories+1))

        for i in range(1, self.num_categories+1):
            self.category_idxs[i] = []

        for i in range(self.num_samples):
            self.category_idxs[int(targets[i])].append(i)

    def __iter__(self):
        num_batches = self.num_samples // self.batch_size
        selected = []

        for i in range(num_batches):
            batch = []
            random.shuffle(self.categorys)
            categories_selcted = self.categorys[:self.numBatchCategory]
            # categories_selcted = np.random.randint(self.num_categories, size=self.numBatchCategory)

            for j in categories_selcted:
                random.shuffle(self.category_idxs[j])
                batch.extend(self.category_idxs[j][:int(self.batch_size // self.numBatchCategory)])

            random.shuffle(batch)

            selected.extend(batch)

        # print('--------------------------------------------', countn / (countp + countn) * 1.0)

        return iter(torch.LongTensor(selected))

    def __len__(self):
        return self.num_samples

class categoryRandomSampler_dog(Sampler):
    def __init__(self, numBatchCategory, targets, batch_size):
        """
        This sampler will sample numBatchCategory categories in each batch.
        """
        self.targets = list(targets)
        self.batch_size = batch_size
        self.num_samples = len(targets)
        self.numBatchCategory = numBatchCategory
        self.num_categories = max(targets)
        self.category_idxs = {}
        self.categorys = list(range( self.num_categories+1))

        for i in range( self.num_categories+1):
            self.category_idxs[i] = []

        for i in range(self.num_samples):
            self.category_idxs[int(targets[i])].append(i)

    def __iter__(self):
        num_batches = self.num_samples // self.batch_size
        selected = []

        for i in range(num_batches):
            batch = []
            random.shuffle(self.categorys)
            categories_selcted = self.categorys[:self.numBatchCategory]
            # categories_selcted = np.random.randint(self.num_categories, size=self.numBatchCategory)

            for j in categories_selcted:
                random.shuffle(self.category_idxs[j])
                batch.extend(self.category_idxs[j][:int(self.batch_size // self.numBatchCategory)])

            random.shuffle(batch)

            selected.extend(batch)

        # print('--------------------------------------------', countn / (countp + countn) * 1.0)

        return iter(torch.LongTensor(selected))

    def __len__(self):
        return self.num_samples

def init_cifar_dataloader(batchSize):
    # root_dir = '../../disk2/yangyifan/cifar10/'
    root_dir = '/home/disk1/yangyifan/cub/CUB_200_2011/'

    # transform_train = transforms.Compose(
    #     [transforms.Resize((256, 256)), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(),
    #      transforms.ToTensor()])
    # transform_test = transforms.Compose(
    #     [transforms.Resize((256, 256)), transforms.CenterCrop(224), transforms.ToTensor()])
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    # transform_test = transforms.Compose([transforms.ToTensor()])

    test_dir = root_dir+'test.txt'
    train_dir = root_dir+'train.txt'
    DB_dir = root_dir+'train.txt'

    testset = MyDataset(txt=test_dir, transform=transform_test)
    test_loader = torchdata.DataLoader(testset, batch_size=batchSize,shuffle=False,
                                       num_workers=2, pin_memory=True)

    train_set = MyDataset(txt=train_dir, transform=transform_train)
    train_loader_ = torchdata.DataLoader(train_set, batch_size=batchSize, shuffle=True,
                                       num_workers=2, pin_memory=True)
    labels = []
    for batch_idx, (data, target) in enumerate(train_loader_):
        labels.extend(target)
    casampler = categoryRandomSampler(numBatchCategory=10, targets=labels, batch_size=batchSize)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batchSize, shuffle=False, num_workers=2,
                                               sampler=casampler)

    DB_set = MyDataset(txt=DB_dir, transform=transform_train)
    DB_loader = torchdata.DataLoader(DB_set, batch_size=batchSize, shuffle=True,
                                       num_workers=2, pin_memory=True)

    return DB_loader, train_loader, test_loader



def combinations(iterable, r):
    pool = list(iterable)
    n = len(pool)
    for indices in itertools.permutations(range(n), r):
        if sorted(indices) == list(indices):
            yield list(pool[i] for i in indices)

def get_triplets(labels):
    labels = labels.cpu().data.numpy() #一个batch_size维的向量，对应每张图片的类别数字(0~199)
    triplets = []
    for label in set(labels): #set(labels)不包含重复数字，相当于逐个遍历labels中包含的类别
        label_mask = (labels == label)
        label_indices = np.where(label_mask)[0] #当前类别所在的索引位置（可能有多个位置）
        if len(label_indices) < 2:  #当前类别只存在一个样本，无法构造三元组
            continue
        negative_indices = np.where(np.logical_not(label_mask))[0] #其他类别所在的索引位置
        anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs

        # Add all negatives for all positive pairs
        temp_triplets = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                         for neg_ind in negative_indices]
        triplets += temp_triplets

    return torch.LongTensor(np.array(triplets))

def set_logger(logger, info):
    logger.setLevel(logging.INFO)

    os.makedirs('log', exist_ok=True)
    filename = './log/' + info + '.log'

    ch = TimedRotatingFileHandler(filename, when='D', encoding="utf-8")
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def timing(f):
    """print time used for function f"""

    @wraps(f)
    def wrapper(*args, **kwargs):
        time_start = time.time()
        ret = f(*args, **kwargs)
        print(f'total time = {time.time() - time_start:.4f}')
        return ret

    return wrapper


@timing
def compute_mAP(trn_binary, tst_binary, trn_label, tst_label):
    """
    compute mAP by searching testset from trainset
    https://github.com/flyingpot/pytorch_deephash
    """
    for x in trn_binary, tst_binary, trn_label, tst_label: x.long()

    AP = []
    Ns = torch.arange(1, trn_binary.size(0) + 1)
    Ns = Ns.type(torch.FloatTensor)
    for i in range(tst_binary.size(0)):
        query_label, query_binary = tst_label[i], tst_binary[i]
        _, query_result = torch.sum((query_binary != trn_binary).long(), dim=1).sort()
        correct = (query_label == trn_label[query_result]).float()
        P = torch.cumsum(correct.type(torch.FloatTensor), dim=0) / Ns
        AP.append(torch.sum(P * correct) / torch.sum(correct))
    mAP = torch.mean(torch.Tensor(AP))
    return mAP


def choose_gpu(i_gpu):
    """choose current CUDA device"""
    torch.cuda.device(i_gpu).__enter__()
    cudnn.benchmark = True


def feed_random_seed(seed=np.random.randint(1, 10000)):
    """feed random seed"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
