import os
import torch
import random
import time
import os
from tqdm import tqdm
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from models.a_llmrec_model import *
from pre_train.sasrec.utils import data_partition, SeqDataset, SeqDataset_Inference

# 设置分布式训练的初始化
def setup_ddp(rank, world_size):
    os.environ ["MASTER_ADDR"] = "localhost"  # 设置主节点的地址
    os.environ ["MASTER_PORT"] = "12355"  # 设置主节点的端口
    init_process_group(backend="nccl", rank=rank, world_size=world_size)  # 初始化进程组
    torch.cuda.set_device(rank)  # 设置每个进程使用的GPU

# 训练阶段1
def train_model_phase1(args):
    print('A-LLMRec start train phase-1\n')
    if args.multi_gpu:
        world_size = torch.cuda.device_count()  # 获取GPU的数量
        mp.spawn(train_model_phase1_, args=(world_size, args), nprocs=world_size)  # 使用多GPU训练
    else:
        train_model_phase1_(0, 0, args)  # 单GPU或CPU训练
        
# 训练阶段2
def train_model_phase2(args):
    print('A-LLMRec start train phase-2\n')
    if args.multi_gpu:
        world_size = torch.cuda.device_count()  # 获取GPU数量
        mp.spawn(train_model_phase2_, args=(world_size, args), nprocs=world_size)  # 使用多GPU训练
    else:
        train_model_phase2_(0, 0, args)  # 单GPU或CPU训练

# 推理阶段
def inference(args):
    print('A-LLMRec start inference\n')
    if args.multi_gpu:
        world_size = torch.cuda.device_count()  # 获取GPU数量
        mp.spawn(inference_, args=(world_size, args), nprocs=world_size)  # 使用多GPU推理
    else:
        inference_(0,0,args)  # 单GPU或CPU推理

# 训练阶段1的具体实现
def train_model_phase1_(rank, world_size, args):
    if args.multi_gpu:
        setup_ddp(rank, world_size)  # 初始化分布式训练
        args.device = 'cuda:' + str(rank)  # 设置当前进程的设备
        
    model = A_llmrec_model(args).to(args.device)  # 加载模型并转到对应设备
    
    # 数据预处理
    dataset = data_partition(args.rec_pre_trained_data, path=f'./data/amazon/{args.rec_pre_trained_data}.txt')
    [user_train, user_valid, user_test, usernum, itemnum] = dataset  # 获取训练、验证、测试数据
    print('user num:', usernum, 'item num:', itemnum)
    
    num_batch = len(user_train) // args.batch_size1  # 计算批次数
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))  # 打印平均序列长度
    
    # 初始化Dataloader, Model, Optimizer
    train_data_set = SeqDataset(user_train, usernum, itemnum, args.maxlen)
    if args.multi_gpu:
        train_data_loader = DataLoader(train_data_set, batch_size = args.batch_size1, sampler=DistributedSampler(train_data_set, shuffle=True), pin_memory=True)
        model = DDP(model, device_ids = [args.device], static_graph=True)  # 分布式训练
    else:
        train_data_loader = DataLoader(train_data_set, batch_size = args.batch_size1, pin_memory=True)  # 单机训练        
        
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.stage1_lr, betas=(0.9, 0.98))  # 优化器
    
    epoch_start_idx = 1
    model.train()
    t0 = time.time()
    for epoch in tqdm(range(epoch_start_idx, args.num_epochs + 1)):
        if args.multi_gpu:
            train_data_loader.sampler.set_epoch(epoch)  # 设置数据加载的epoch
        for step, data in enumerate(train_data_loader):
            u, seq, pos, neg = data
            u, seq, pos, neg = u.numpy(), seq.numpy(), pos.numpy(), neg.numpy()
            model([u, seq, pos, neg], optimizer=adam_optimizer, batch_iter=[epoch, args.num_epochs + 1, step, num_batch], mode='phase1')  # 训练
            if step % max(10, num_batch//100) == 0:
                if rank == 0:
                    if args.multi_gpu:
                        model.module.save_model(args, epoch1=epoch)  # 保存模型
                    else:
                        model.save_model(args, epoch1=epoch)  # 保存模型
        if rank == 0:
            if args.multi_gpu:
                model.module.save_model(args, epoch1=epoch)  # 保存模型
            else:
                model.save_model(args, epoch1=epoch)  # 保存模型

    print('train time :', time.time() - t0)
    if args.multi_gpu:
        destroy_process_group()  # 结束分布式训练
    return 

# 训练阶段2的具体实现
def train_model_phase2_(rank, world_size, args):
    if args.multi_gpu:
        setup_ddp(rank, world_size)
        args.device = 'cuda:' + str(rank)  # 设置当前进程的设备
    random.seed(0)

    model = A_llmrec_model(args).to(args.device)
    phase1_epoch = 10
    model.load_model(args, phase1_epoch=phase1_epoch)  # 加载阶段1的模型

    dataset = data_partition(args.rec_pre_trained_data, path=f'./data/amazon/{args.rec_pre_trained_data}.txt')
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    print('user num:', usernum, 'item num:', itemnum)
    
    num_batch = len(user_train) // args.batch_size2
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))  # 打印平均序列长度
    
    # 初始化Dataloader, Model, Optimizer
    train_data_set = SeqDataset(user_train, usernum, itemnum, args.maxlen)
    if args.multi_gpu:
        train_data_loader = DataLoader(train_data_set, batch_size = args.batch_size2, sampler=DistributedSampler(train_data_set, shuffle=True), pin_memory=True)
        model = DDP(model, device_ids = [args.device], static_graph=True)  # 分布式训练
    else:
        train_data_loader = DataLoader(train_data_set, batch_size = args.batch_size2, pin_memory=True, shuffle=True)  # 单机训练
    
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.stage2_lr, betas=(0.9, 0.98))  # 优化器
    
    epoch_start_idx = 1
    model.train()
    t0 = time.time()
    for epoch in tqdm(range(epoch_start_idx, args.num_epochs + 1)):
        if args.multi_gpu:
            train_data_loader.sampler.set_epoch(epoch)  # 设置数据加载的epoch
        for step, data in enumerate(train_data_loader):
            u, seq, pos, neg = data
            u, seq, pos, neg = u.numpy(), seq.numpy(), pos.numpy(), neg.numpy()
            model([u, seq, pos, neg], optimizer=adam_optimizer, batch_iter=[epoch, args.num_epochs + 1, step, num_batch], mode='phase2')  # 训练
            if step % max(10, num_batch//100) == 0:
                if rank == 0:
                    if args.multi_gpu:
                        model.module.save_model(args, epoch1=phase1_epoch, epoch2=epoch)  # 保存模型
                    else:
                        model.save_model(args, epoch1=phase1_epoch, epoch2=epoch)  # 保存模型
        if rank == 0:
            if args.multi_gpu:
                model.module.save_model(args, epoch1=phase1_epoch, epoch2=epoch)  # 保存模型
            else:
                model.save_model(args, epoch1=phase1_epoch, epoch2=epoch)  # 保存模型
    
    print('phase2 train time :', time.time() - t0)
    if args.multi_gpu:
        destroy_process_group()  # 结束分布式训练
    return

# 推理阶段的具体实现
def inference_(rank, world_size, args):
    if args.multi_gpu:
        setup_ddp(rank, world_size)
        args.device = 'cuda:' + str(rank)  # 设置当前进程的设备
        
    model = A_llmrec_model(args).to(args.device)
    phase1_epoch = 10
    phase2_epoch = 5
    model.load_model(args, phase1_epoch=phase1_epoch, phase2_epoch=phase2_epoch)  # 加载训练好的模型

    dataset = data_partition(args.rec_pre_trained_data, path=f'./data/amazon/{args.rec_pre_trained_data}.txt')
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    print('user num:', usernum, 'item num:', itemnum)
    
    num_batch = len(user_train) // args.batch_size_infer
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))  # 打印平均序列长度
    model.eval()  # 设置为推理模式
    
    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)  # 随机选择10000个用户进行推理
    else:
        users = range(1, usernum + 1)  # 如果用户数小于10000，则全部选择
        
    user_list = []
    for u in users:
        if len(user_train[u]) < 1 or len(user_test[u]) < 1: continue
        user_list.append(u)

    inference_data_set = SeqDataset_Inference(user_train, user_valid, user_test, user_list, itemnum, args.maxlen)
    
    if args.multi_gpu:
        inference_data_loader = DataLoader(inference_data_set, batch_size = args.batch_size_infer, sampler=DistributedSampler(inference_data_set, shuffle=True), pin_memory=True)
        model = DDP(model, device_ids = [args.device], static_graph=True)  # 分布式推理
    else:
        inference_data_loader = DataLoader(inference_data_set, batch_size = args.batch_size_infer, pin_memory=True)  # 单机推理
    
    for _, data in enumerate(inference_data_loader):
        u, seq, pos, neg = data
        u, seq, pos, neg = u.numpy(), seq.numpy(), pos.numpy(), neg.numpy()
        model([u, seq, pos, neg, rank], mode='generate')  # 执行推理
