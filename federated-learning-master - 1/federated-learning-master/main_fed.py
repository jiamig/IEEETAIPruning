#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar10_iid,emnist_noniid,emnist_iid, fmnist_iid, fmnist_noniid, celeba_iid, celeba_noniid, cifar10_noniid, cifar100_noniid, cifar100_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, VGG9, ResNet18
from models.Fed import FedAvg
from models.test import test_img


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'emnist':
        trans_emnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.EMNIST('../data/emnist/', split='balanced', train=True, download=True,
                                        transform=trans_emnist)
        dataset_test = datasets.EMNIST('../data/emnist/', split='balanced', train=False, download=True,
                                       transform=trans_emnist)
        if args.iid:
            dict_users = emnist_iid(dataset_train, args.num_users)
        else:
            dict_users = emnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'fmnist':
        trans_fmnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset_train = datasets.FashionMNIST('../data/fmnist/', train=True, download=True, transform=trans_fmnist)
        dataset_test = datasets.FashionMNIST('../data/fmnist/', train=False, download=True, transform=trans_fmnist)
        if args.iid:
            dict_users = fmnist_iid(dataset_train, args.num_users)
        else:
            dict_users = fmnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'celeba':
        trans_celeba = transforms.Compose(
            [transforms.Resize(64), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CelebA(root='../data/celeba/', split='train', download=True, transform=trans_celeba)
        dataset_test = datasets.CelebA(root='../data/celeba/', split='test', download=True, transform=trans_celeba)
        if args.iid:
            dict_users = celeba_iid(dataset_train, args.num_users)
        else:
            dict_users = celeba_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar10':
        trans_cifar10 = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar10', train=True, download=True, transform=trans_cifar10)
        dataset_test = datasets.CIFAR10('../data/cifar10', train=False, download=True, transform=trans_cifar10)
        if args.iid:
            dict_users = cifar10_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar10_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar100':
        trans_cifar100 = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR100('../data/cifar100', train=True, download=True, transform=trans_cifar100)
        dataset_test = datasets.CIFAR100('../data/cifar100', train=False, download=True, transform=trans_cifar100)
        if args.iid:
            dict_users = cifar100_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar100_noniid(dataset_train, args.num_users)
    else:
        exit('Error: unrecognized dataset')

    print("Dataset loaded successfully!")
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'mlp' and args.dataset == 'mnist':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    elif args.model == 'cnn' and args.dataset in ['emnist', 'fmnist']:
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'vgg9' and args.dataset in ['cifar10', 'celeba']:
        net_glob = VGG9(num_classes=args.num_classes).to(args.device)
    elif args.model == 'resnet18' and args.dataset == 'cifar100':
        net_glob = ResNet18(num_classes=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model or dataset combination')
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    import time
    import matplotlib.pyplot as plt
    import copy


    # 函数计算模型大小
    def model_size(model):
        """计算模型的参数量"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_all_mb = (param_size + buffer_size) / 1024 ** 2
        return size_all_mb


    # 初始化变量
    communication_rounds = 0
    total_communication_time = 0
    total_data_transferred = 0
    total_training_time = 0

    # 训练部分
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]

    for iter in range(args.epochs):
        start_training_time = time.time()
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))

            # 记录通信开销
            start_communication_time = time.time()
            data_transferred = model_size(w)
            total_data_transferred += data_transferred
            end_communication_time = time.time()
            communication_time = end_communication_time - start_communication_time
            total_communication_time += communication_time

            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

        # 更新全局权重
        w_glob = FedAvg(w_locals)
        net_glob.load_state_dict(w_glob)

        # 打印损失
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

        end_training_time = time.time()
        training_time = end_training_time - start_training_time
        total_training_time += training_time

        print(
            f"Round: {iter}, Training time: {training_time:.2f}s, Communication time: {communication_time:.2f}s, Data transferred: {data_transferred:.2f}MB")

    # 绘制损失曲线
    #plt.figure()
    #plt.plot(range(len(loss_train)), loss_train)
    #plt.ylabel('train_loss')
    #plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # 测试部分
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

    # 打印总的通信开销和训练时间
    print(f"Total Communication Rounds: {communication_rounds}")
    #print(f"Total Communication Time: {total_communication_time:.6f}s")
    print(f"Total Data Transferred: {total_data_transferred:.6f}MB")
    print(f"Total Training Time: {total_training_time:.6f}s")


