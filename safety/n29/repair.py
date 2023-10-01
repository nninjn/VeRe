import os
from collections import defaultdict
import torch
import torch.nn as nn
import torchvision
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import Flatten
import time
from dataprocess import Data
from torch.utils.data import DataLoader, Dataset, Subset
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pyswarms as ps
from tqdm import tqdm
import random

class Repair():


    def __init__(self, BATCH_SIZE, n_classes, target, buggy_model, normal_data, normal_data_loader, \
                    normal_data_test, normal_data_test_loader, mis_data, mis_data_loader, \
                    mis_data_test, mis_data_test_loader, 
                    approximate_method, interval_times, local_epoch) -> None:

        self.BATCH_SIZE = BATCH_SIZE
        self.target = target
        self.buggy_model = buggy_model
        self.buggy_model.eval()


        self.normal_data = normal_data
        self.normal_data_loader = normal_data_loader

        self.normal_data_test = normal_data_test
        self.normal_data_test_loader = normal_data_test_loader

        self.mis_data = mis_data
        self.mis_data_loader = mis_data_loader

        self.mis_data_test = mis_data_test
        self.mis_data_test_loader = mis_data_test_loader
        
        self.n_classes = n_classes

        self.approximate_method = approximate_method

        self.alpha = 0.1
        self.repair_n = 20
        self.pso_repair_way = []

        self.interval_times = interval_times
        self.local_epoch = local_epoch

        self.repair_time = {}
        self.loc_time = {}
        self.cor_data = []
        self.incor_data = []
        self.cor_data_num = len(self.cor_data)
        self.incor_data_num = len(self.incor_data)


    def Test_acc(self, nn_test, loader):
        print('Test Start! Test Type = Normal data dd!')
        nn_test.eval()
        nn_test = nn_test.cuda()
        vio = 0
        for step, (x) in enumerate(loader):
            train_x = x
            train_x = train_x.cuda()
            train_output = nn_test(train_x)
            train_output = train_output[-1]
            pred_y = torch.min(train_output.cpu(), 1)[1].numpy()
            vio += (pred_y == 2).sum() + (pred_y == 3).sum() + (pred_y == 4).sum()
        print('-' * 20, 'Normal data VR = ', vio / len(loader.dataset) * 100)
        return vio / len(loader.dataset)

    def Test_SR(self, nn_test, loader):
        print('Test Start! Test Type = Counterexample !')
        nn_test.eval()
        nn_test = nn_test.cuda()
        vio = 0

        for step, (x) in enumerate(loader):
            train_x = x
            train_x = train_x.cuda()
            train_output = nn_test(train_x)
            train_output = train_output[-1]
            pred_y = torch.min(train_output.cpu(), 1)[1].numpy()
            # print(pred_y)

            vio += (pred_y == 2).sum() + (pred_y == 3).sum() + (pred_y == 4).sum()


        print('-' * 20, 'Counterexample VR = ', vio / len(loader.dataset) * 100)
        return vio / len(loader.dataset)


    def repair_data_classification(self):
        self.cor_data = []
        self.incor_data = []

        if torch.cuda.is_available():
            self.buggy_model.cuda()
        for step, (x) in enumerate(self.normal_data_loader):
            train_x = x
            train_x = train_x.cuda()
            train_output = self.buggy_model(train_x)
            train_output = train_output[-1]
            pred_y = torch.min(train_output.cpu(), 1)[1].numpy()

            for i in range(len(pred_y)): 
                if pred_y[i] == 0 or pred_y[i] == 1:
                    self.cor_data.append(x[i])
                else:
                    self.incor_data.append(x[i])
        for step, (x) in enumerate(self.mis_data_loader):
            train_x = x
            train_x = train_x.cuda()
            train_output = self.buggy_model(train_x)
            train_output = train_output[-1]
            pred_y = torch.min(train_output.cpu(), 1)[1].numpy()

            for i in range(len(pred_y)): 
                if pred_y[i] != 0 and pred_y[i] != 1:
                    self.incor_data.append(x[i])
                if len(self.incor_data) > 1023:
                    break
        self.cor_data_num = len(self.cor_data)
        self.incor_data_num = len(self.incor_data)


    def analyze_neuron_action(self, check_layer, analyze_neuron_num):
        print('Now we start analyze all neurons in check_layer, to compute max/min activation value of all input')
        neuron_act_bound = torch.zeros((analyze_neuron_num, 2))
        
        image = []
        for i in range(len(self.mis_data_loader.dataset)):
            x = self.mis_data[i]
            image.append(x)

        for i in range(len(self.normal_data_loader.dataset)):
            x = self.normal_data[i]
            image.append(x)
        image = torch.stack(image)
        print(image.shape)

        if torch.cuda.is_available():
            image = image.cuda()
            self.buggy_model.cuda()
            
        all_output = self.buggy_model(image)

        check_value = all_output[check_layer]
        if torch.cuda.is_available():
            check_value = check_value.cuda()
        neuron_act_bound[:, 0], index = torch.min(check_value, dim=0)
        neuron_act_bound[:, 1], index = torch.max(check_value, dim=0)

        return neuron_act_bound

    def compute_max_effect(self, round, check_layer, model, analyze_neuron_num, N, interval_num):
        print('Now we start analyze the model, compute max effect for every neuron.')
        
        all_input_effect = torch.zeros((analyze_neuron_num)).cuda()
        all_neuron_effect = torch.zeros((analyze_neuron_num, self.incor_data_num + 1)).cuda()
        # shape = neuron_num * N * interval_num * 2
        all_neuron_eps_effect = torch.zeros((analyze_neuron_num, self.incor_data_num, interval_num, 2)).cuda()
        # print(all_input_effect.device, all_neuron_effect.device, all_neuron_eps_effect.device)
        t = time.time()

        image = []
        true_label = []
        if len(self.incor_data) == 0:
            return 0, 0
        image = torch.stack(self.incor_data)
        # print(image.shape)

        if torch.cuda.is_available():
            image = image.cuda()
            model = model.cuda()
            self.buggy_model.cuda()
            
        all_output = self.buggy_model(image)
        pred_y = torch.min(all_output[-1].cpu(), 1)[1]
        true_input = all_output[check_layer]
        if torch.cuda.is_available():
            true_input = true_input.cuda()

        lirpa_model = BoundedModule(model, torch.empty_like(true_input), device=true_input.device)
        print('Running on', true_input.device)
        neuron_num = len(true_input[0])
        neuron_act_bound = self.analyze_neuron_action(check_layer=check_layer, analyze_neuron_num=analyze_neuron_num)

        eps_record = torch.zeros((analyze_neuron_num, interval_num, 2)).cuda()
        for neuron in range(neuron_num):
            diff = neuron_act_bound[neuron][1] - neuron_act_bound[neuron][0]
            st = neuron_act_bound[neuron][0] - diff 
            st = (-self.interval_times) * neuron_act_bound[neuron][1]
            diff = neuron_act_bound[neuron][1] * self.interval_times * 2
            steps = diff / interval_num
            for i in range(interval_num):
                eps_record[neuron][i][0] = st + steps * i
                eps_record[neuron][i][1] = st + steps * (i + 1)

        C = torch.zeros(size=(self.incor_data_num, 1, self.n_classes), device=true_input.device)
        for i in range(self.incor_data_num):
            C[i][0][0] = 1.0
            C[i][0][1] = 1.0
 

        ori = torch.zeros((self.incor_data_num)).cuda()
        for i in range(self.incor_data_num):
            ori[i] = all_output[-1][i][0] + all_output[-1][i][1]


        for neuron in range(neuron_num):
            eps = eps_record[neuron]
            if sum(eps[:, 0]) == sum(eps[:, 1]):
                all_neuron_eps_effect[neuron, :, :, :] = 0
            else:
                true_input_L = true_input.detach().clone()
                true_input_U = true_input.detach().clone()
                for interval in range(len(eps)):
                    true_input_L[:, neuron] = eps[interval][0]
                    true_input_U[:, neuron] = eps[interval][1]
                    ptb = PerturbationLpNorm(x_L=true_input_L, x_U=true_input_U)
                    true_input = BoundedTensor(true_input, ptb)
                            
                    required_A = defaultdict(set)
                    required_A[lirpa_model.output_name[0]].add(lirpa_model.input_name[0])
                    for method in self.approximate_method:
                        if 'Optimized' in method:
                            lirpa_model.set_bound_opts({'optimize_bound_args': {'iteration': 20, 'lr_alpha': 0.1, }})

                                
                        lb, ub, A_dict = lirpa_model.compute_bounds(x=(true_input,), method=method.split()[0], return_A=True,
                                                                        needed_A_dict=required_A, C=C)
                        l_A, l_bias = A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['lA'], \
                                                A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['lbias']


                            
                        # interval_l_bound_L interval_l_bound_U shape ---- [self.incor_data_num]
                        interval_l_bound_L = torch.sum(true_input_L * l_A[:, 0, :], dim=1) + l_bias[:, 0]
                        interval_l_bound_U = torch.sum(true_input_U * l_A[:, 0, :], dim=1) + l_bias[:, 0]
                        # shape = neuron_num * self.incor_data_num * interval_num * 2
                        all_neuron_eps_effect[neuron, :, interval, 0] = (ori - interval_l_bound_L).detach().clone()
                        all_neuron_eps_effect[neuron, :, interval, 1] = (ori - interval_l_bound_U).detach().clone()

            
                for i in range(self.incor_data_num):
                    all_neuron_effect[neuron][i] = max(0, torch.max(all_neuron_eps_effect[neuron][i]))
        for i in range(self.incor_data_num):
            # eff = all_neuron_effect[:, i].detach().clone()
            max_effect, index = torch.max(all_neuron_effect[:, i], dim=0)
            min_effect, index = torch.min(all_neuron_effect[:, i], dim=0)
            if max_effect - min_effect == 0:
                all_neuron_effect[:, i] = 0
            else:
                all_neuron_effect[:, i] -= min_effect
                all_neuron_effect[:, i] /= (max_effect - min_effect)

        for neuron in range(neuron_num):
            all_neuron_effect[neuron][self.incor_data_num] = torch.sum(all_neuron_effect[neuron][0: N])
            all_input_effect[neuron] = all_neuron_effect[neuron][self.incor_data_num]
        sorted_effect, sorted_index = torch.sort(all_input_effect, descending=True)
        all_neuron_effect = all_neuron_effect.tolist()
        all_neuron_effect.sort(key=lambda x:x[-1], reverse=True)
        if check_layer not in self.loc_time:
            self.loc_time[check_layer] = 0
        self.loc_time[check_layer] += time.time() - t

        repair_neuron = sorted_index[0]

        vio = self.local_repair(check_layer=check_layer, model=model, repair_interval=all_neuron_eps_effect[repair_neuron],\
                                analyze_neuron=repair_neuron, ori=ori, eps_record=eps_record[repair_neuron], local_epoch=self.local_epoch)
        return vio[0], vio[1]

    def local_repair(self, check_layer, model, repair_interval, analyze_neuron, ori, eps_record, local_epoch):
        print('Now we start to repair the model locally')

        t = time.time()
        image = []
        true_label = []
        for data in self.incor_data:
            image.append(data)

        for data in self.cor_data:
            image.append(data)

        image = torch.stack(image)
        input_num = image.shape[0]

        if torch.cuda.is_available():
            image = image.cuda()
            model = model.cuda()
            self.buggy_model.cuda()
            
        all_output = self.buggy_model(image)
        pre_layer_value = all_output[check_layer - 1]
        check_layer_value = all_output[check_layer]
        # before repair
        check_neuron_value = check_layer_value[:, analyze_neuron]
        in_weight_num = len(pre_layer_value[0])

        mini_label = torch.empty([self.cor_data_num + self.incor_data_num], requires_grad=False).cuda()
        for i in range(self.incor_data_num):
            m = 0
            mini_label[i] = check_neuron_value[i]
            f = 'not entirely'
            point_set = []

            m1, index1 = torch.max(repair_interval[i, :, 0], dim=0)
            m2, index2 = torch.max(repair_interval[i, :, 1], dim=0)

            if m1 > m2 and m1 > 0:
                mini_label[i] = eps_record[index1][0]
            elif m2 > m1 and m2 > 0:
                mini_label[i] = eps_record[index2][1]
            if m1 > m2:
                index = index1
                m = m1
            else:
                index = index2
                m = m2


            # another strategy: ideal interval
            '''
            for j1 in range(repair_interval[i].shape[0]):
                for j2 in range(repair_interval[i].shape[1]):
                    if repair_interval[i][j1][j2] + ori[i] > 0:
                        f = 'entirely'
                        point_set.append(eps_record[j1][j2])
            # if m + ori[i] > 0:
                
            if f == 'not entirely':
                for j in range(repair_interval[i].shape[0]):
                    m1, index = torch.max(repair_interval[i][j], dim=0)
                    if m1 > m:
                        m = m1
                        mini_label[i] = eps_record[j][index]
            else:
                distance = torch.Tensor(point_set).cuda()
                distance -= check_neuron_value[i]
                distance = torch.abs(distance)
                index = torch.argmin(distance)
                mini_label[i] = point_set[index]
            '''

        mini_label[self.incor_data_num:] = check_neuron_value[self.incor_data_num:].detach()

        d = len(self.buggy_model.state_dict().keys()) - len(model.state_dict().keys())

        para_name = list(self.buggy_model.state_dict().keys())
        pre_check_wegiht = self.buggy_model.state_dict()[para_name[d - 2]]
        pre_check_bias = self.buggy_model.state_dict()[para_name[d - 1]]
        
        post_check_wegiht = self.buggy_model.state_dict()[para_name[d]]
        post_check_bias = self.buggy_model.state_dict()[para_name[d + 1]]


        mini_input = torch.empty([self.cor_data_num + self.incor_data_num, in_weight_num], requires_grad=False).cuda()
        mini_input = pre_layer_value.detach()

        mini_data = Data(mini_input, mini_label.detach())
        mini_loader = DataLoader(dataset=mini_data, batch_size=self.BATCH_SIZE, shuffle=True)

        check_neuron_in_weight = pre_check_wegiht[analyze_neuron]
        check_neuron_bias = pre_check_bias[analyze_neuron]
        
        mini_nn = nn.Sequential(
            nn.Linear(in_weight_num, 1),
            nn.ReLU(),
            nn.Linear(1, 1)
        )
        para = {}
        para['0.weight'] = check_neuron_in_weight.view(1, in_weight_num)
        para['0.bias'] = check_neuron_bias.view(1)
        para['2.weight'] = torch.tensor(1).view(1, 1)
        para['2.bias'] = torch.tensor(0).view(1)
        mini_nn.load_state_dict(para)
        mini_nn.cuda()
        mini_nn.train()
        optim = torch.optim.Adam(mini_nn.parameters(), lr=0.01)
        loss_fcn = nn.MSELoss()
        start = time.time()


        for epoch in range(local_epoch):
            for step,(x,y) in enumerate(mini_loader):
                output = mini_nn(x)
                output = output.squeeze(1)
                loss = loss_fcn(output, y)
                optim.zero_grad()
                loss.backward()
                optim.step()
        

        repaired_para = {}
        for key in self.buggy_model.state_dict().keys():
            repaired_para[key] = self.buggy_model.state_dict()[key]
        b = mini_nn.state_dict()['2.bias']
        w = mini_nn.state_dict()['2.weight'][0]

        repaired_para[para_name[d - 2]][analyze_neuron] = mini_nn.state_dict()['0.weight']
        repaired_para[para_name[d - 1]][analyze_neuron] = mini_nn.state_dict()['0.bias']
        repaired_para[para_name[d + 1]] += b * post_check_wegiht[:, analyze_neuron]
        repaired_para[para_name[d]][:, analyze_neuron] *= w
            
        if check_layer not in self.repair_time:
            self.repair_time[check_layer] = 0
        self.repair_time[check_layer] += time.time() - t
        self.buggy_model.load_state_dict(repaired_para)
        print('-' * 20, 'One repair round end, using data (for repair) to test!')
        normal_vio = self.Test_acc(self.buggy_model, self.normal_data_loader)
        mis_vio = self.Test_SR(self.buggy_model, self.mis_data_loader)      
        return normal_vio, mis_vio

   
