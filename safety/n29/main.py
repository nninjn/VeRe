
import os
from collections import defaultdict
import torch
import torch.nn as nn
import torchvision
import time
from network import FNN, FNNspilt6
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from torch.utils.data import DataLoader, Dataset, Subset
import random
import numpy as np
from repair import Repair
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import argparse

parse = argparse.ArgumentParser(description='Safety repair')    
parse.add_argument('--interval', type=int, help='interval nums', default=5)
parse.add_argument('--localepoch', type=int, help='epoch for local repair', default=5)
parse.add_argument('--N', type=int, help='poi_input_num for repair', default=10000)
parse.add_argument('--N_clean', type=int, help='clean_input_num for repair', default=10000)
parse.add_argument('--layer_round', type=int, help='repair round for a layer', default=20)
parse.add_argument('--seed', type=int, default=0)
args = parse.parse_args() 
print(args)
interval_num = args.interval
local_epoch = args.localepoch
N = args.N
N_clean = args.N_clean
layer_round = args.layer_round

buggy_nn = FNN()
n_classes = 5
BATCH_SIZE = 32

DRAWNDOWN_SIZE = BATCH_SIZE * 313
COUNTEREG_SIZE = BATCH_SIZE * 313
buggy_nn.load_state_dict(torch.load('n29/model/n29.pth'))
buggy_nn.eval()


BATCH_SIZE = 64

correct_data = torch.load('n29/data/drawdown.pt')
correct_data_test = torch.load('n29/data/drawdown_test.pt')
mis_data = torch.load('n29/data/counterexample.pt')
mis_data_test = torch.load('n29/data/counterexample_test.pt')

random.seed(args.seed)

l = random.randint(0, len(correct_data) - N_clean)
clean_data_for_repair = Subset(dataset=correct_data, indices=list(range(l, l + N_clean)))
correct_data_loader = DataLoader(dataset=clean_data_for_repair, batch_size=BATCH_SIZE, shuffle=False)
print(l, ' - ', l + N_clean, ' clean data for repair')

# l = random.randint(0, (len(bd_test_dataset) // 2) - N)
poi_data_for_repair = Subset(dataset=mis_data, indices=list(range(l, l + N)))
mis_data_loader = DataLoader(dataset=poi_data_for_repair, batch_size=BATCH_SIZE, shuffle=False)
print(l, ' - ', l + N, ' poisoned data for repair')

# correct_data_loader = DataLoader(dataset=correct_data, batch_size=BATCH_SIZE, shuffle=False)
correct_data_test_loader = DataLoader(dataset=correct_data_test, batch_size=BATCH_SIZE, shuffle=False)
# mis_data_loader = DataLoader(dataset=mis_data, batch_size=BATCH_SIZE, shuffle=False)
mis_data_test_loader = DataLoader(dataset=mis_data_test, batch_size=BATCH_SIZE, shuffle=False)

approximate_method = ['backward (CROWN)']

interval_times = 1
safety_reapir = Repair(BATCH_SIZE=BATCH_SIZE, n_classes=n_classes,  target=None, buggy_model=buggy_nn, \
                    normal_data=correct_data, normal_data_loader=correct_data_loader, normal_data_test=correct_data_test, \
                    normal_data_test_loader=correct_data_test_loader, mis_data=mis_data, mis_data_loader=mis_data_loader, \
                    mis_data_test=mis_data_test, mis_data_test_loader=mis_data_test_loader, approximate_method=approximate_method, \
                    interval_times=interval_times, local_epoch=local_epoch)

acc_test = safety_reapir.Test_acc(buggy_nn, safety_reapir.normal_data_test_loader)
sr_test= safety_reapir.Test_SR(buggy_nn, safety_reapir.mis_data_test_loader)
print(acc_test, sr_test)

acc = safety_reapir.Test_acc(buggy_nn, safety_reapir.normal_data_loader)
acc = 1 - acc
sr = safety_reapir.Test_SR(buggy_nn, safety_reapir.mis_data_loader)


record = []
record.append([acc, sr])
best = len(correct_data_loader.dataset) * acc + len(mis_data_loader.dataset) * (1 - sr)
print(best) 
for check_layer in range(4, 5):
# for check_layer in range(5, 6):
    if check_layer == 5:
        model = FNNspilt6()
    elif check_layer == 4:
        from network import FNNspilt5
        model = FNNspilt5()

    analyze_neuron_num = 50
    print(model.state_dict().keys())

    torch.save(safety_reapir.buggy_model.state_dict(), 'n29/model/best model.pth')
    no_eff = 0
    for round in range(layer_round):
        safety_reapir.repair_data_classification()
        # sample_input = list(range(1, 2))
        # backdoor_reapir.sample_nn([879], check_layer, model, 3000, 0.03, project, sample_input)

        spilt_para = {}
        for key in model.state_dict().keys():
            spilt_para[key] = safety_reapir.buggy_model.state_dict()[key]
        model.load_state_dict(spilt_para)
            
        print('-' * 100, 'Round: ', round)
        normal_vio, mis_vio = safety_reapir.compute_max_effect(check_layer=check_layer, round=round,\
                                                    model=model, analyze_neuron_num=analyze_neuron_num, N=N, interval_num=interval_num)
        if normal_vio == 0 and mis_vio == 0:
            print('Model is perfect, stop repair!')
            print('--' * 20)
            break


print('loc time = ', safety_reapir.loc_time, 'repair time = ', safety_reapir.repair_time)

acc_test = safety_reapir.Test_acc(buggy_nn, safety_reapir.normal_data_test_loader)
sr_test= safety_reapir.Test_SR(buggy_nn, safety_reapir.mis_data_test_loader)
print(acc_test, sr_test)


