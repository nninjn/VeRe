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
# from sklearn import linear_model
import random
import glob
import logging

class Repair():


    def __init__(self, BATCH_SIZE, n_classes, \
                    target, clean_model, poisoned_model, clean_test_data, poisoned_test_data,\
                    approximate_method, N, N_clean, interval_times, local_epoch, seed, imagenet=False) -> None:

        self.BATCH_SIZE = BATCH_SIZE
        self.target = target
        self.clean_model = clean_model
        self.poisoned_model = poisoned_model
        self.poisoned_model.eval()
        # self.poisoned_model.cuda()

        self.clean_test_data = clean_test_data
        self.poisoned_test_data = poisoned_test_data
        self.n_classes = n_classes

        self.approximate_method = approximate_method

        self.alpha = 0.1
        self.repair_n = 20
        self.way = []

        self.N = N
        self.N_clean = N_clean
        self.interval_times = interval_times
        self.local_epoch = local_epoch

        self.repair_time = {}
        self.loc_time = {}
        self.imagenet = imagenet
        if not imagenet:
            # random shuffle has done! 
            random.seed(seed)
            l = random.randint(0, (len(self.clean_test_data) // 2) - 1000)
            self.repair_test_acc_dataset = Subset(dataset=self.clean_test_data, indices=list(range(len(self.clean_test_data) // 2, len(self.clean_test_data))))
            self.repair_test_sr_dataset = Subset(dataset=self.poisoned_test_data, indices=list(range(len(self.poisoned_test_data) // 2, len(self.poisoned_test_data))))
            self.repair_test_acc_loader = DataLoader(dataset=self.repair_test_acc_dataset, batch_size=BATCH_SIZE, shuffle=False)
            self.repair_test_sr_loader = DataLoader(dataset=self.repair_test_sr_dataset, batch_size=BATCH_SIZE, shuffle=False)
            print(len(self.clean_test_data) // 2, ' - ', len(self.clean_test_data), ' data for test acc and sr')

            # divide test set to repair set and generalization set
            l = random.randint(0, (len(self.clean_test_data) // 2) - N_clean)
            self.clean_data_for_repair = Subset(dataset=self.clean_test_data, indices=list(range(l, l + N_clean)))
            self.clean_data_for_repair_loader = DataLoader(dataset=self.clean_data_for_repair, batch_size=N_clean, shuffle=False)
            print(l, ' - ', l + N_clean, ' clean data for repair')

            l = random.randint(0, (len(self.poisoned_test_data) // 2) - N)
            self.poi_data_for_repair = Subset(dataset=self.poisoned_test_data, indices=list(range(l, l + N)))
            self.poi_data_for_repair_loader = DataLoader(dataset=self.poi_data_for_repair, batch_size=N, shuffle=False)
            print(l, ' - ', l + N, ' poisoned data for repair')
        else:
            random.seed(seed)
            ind = random.sample(list(range(len(self.clean_test_data))), len(self.clean_test_data) // 2)
            self.repair_test_acc_dataset = Subset(dataset=self.clean_test_data, indices=ind)
            self.repair_test_acc_loader = DataLoader(dataset=self.repair_test_acc_dataset, batch_size=BATCH_SIZE, shuffle=False)
            self.repair_test_sr_dataset = Subset(dataset=self.poisoned_test_data, indices=ind)
            self.repair_test_sr_loader = DataLoader(dataset=self.repair_test_sr_dataset, batch_size=BATCH_SIZE, shuffle=False)
            print(len(self.clean_test_data) // 2, ' data for test acc and sr')

            # divide test set to repair set and generalization set
            ind1 = []
            while len(ind1) != self.N_clean:
                u = random.sample(list(range(len(self.clean_test_data))), 1)
                if u not in ind1 and u not in ind:
                    ind1.append(u[0])
            self.clean_data_for_repair = Subset(dataset=self.clean_test_data, indices=ind1)
            self.clean_data_for_repair_loader = DataLoader(dataset=self.clean_data_for_repair, batch_size=16, shuffle=False)
            print(ind1, ' clean data for repair')

            ind2 = []
            while len(ind2) != self.N:
                u = random.sample(list(range(len(self.poisoned_test_data))), 1)
                if u not in ind2 and u not in ind:
                    ind2.append(u[0])
            self.poi_data_for_repair = Subset(dataset=self.poisoned_test_data, indices=ind2)
            self.poi_data_for_repair_loader = DataLoader(dataset=self.poi_data_for_repair, batch_size=16, shuffle=False)
            print(ind2, ' poisoned data for repair')

        self.cor_data = []
        self.incor_data = []
        self.cor_data_num = len(self.cor_data)
        self.incor_data_num = len(self.incor_data)

        self.classifi = 0
        
    def Test_acc(self, nn_test, loader, model_type):
        print('Test Start! Test Type = Clean data Accuracy!')
        nn_test.eval()
        nn_test = nn_test.cuda()
        accuracy = 0
        for step, (x, y) in enumerate(loader):
            train_output = nn_test(x.cuda())[-1]
            pred_y = torch.max(train_output.cpu(), 1)[1].numpy()
            label_y = y.cpu().numpy()
            accuracy += (pred_y == label_y).sum()
        print(accuracy, len(loader.dataset))
        print('-' * 20, 'Acc = ', accuracy / len(loader.dataset) * 100)
        return accuracy / len(loader.dataset)

    def Test_SR(self, nn_test, loader, model_type):
        # print('Test Start! Test Type = Poisoned data SR!', ' Model Type = ', model_type, 'target:', self.target)
        nn_test.eval()
        nn_test = nn_test.cuda()
        success = 0
        nocount = 0
        for step, (x, y) in enumerate(loader):
            train_output = nn_test(x.cuda())[-1]
            pred_y = torch.max(train_output.cpu(), 1)[1].numpy()
            label_y = y.cpu().numpy()
            success += ((pred_y == self.target) & (label_y != self.target)).sum()
            nocount += (label_y == self.target * np.ones(pred_y.shape)).sum()
        print(nocount, success, len(loader.dataset))
        sr = success / (len(loader.dataset) - nocount)
        print('-' * 20, 'SR = ', sr * 100)
        return sr, len(loader.dataset) - nocount - success

    def show(self, x, project):
        print('Show a input:')
        image_array = x
        if 'mnist' in project:
            image_array = image_array.reshape(28, 28)
        elif 'cifar' in project:
            image_array = image_array.reshape(3, 32, 32)
        plt.imshow(image_array)
        plt.savefig('./1.jpg')
        plt.show()

    def repair_data_classification(self):
        self.cor_data = []
        self.incor_data = []

        if torch.cuda.is_available():
            self.poisoned_model.cuda()
        for step, (x, y) in enumerate(self.clean_data_for_repair_loader):
            train_x, train_y = x, y
            train_x = train_x.cuda()
            train_output = self.poisoned_model(train_x)
            train_output = train_output[-1]
            pred_y = torch.max(train_output.cpu(), 1)[1].numpy()
            label_y = train_y.cpu().numpy()
            for i in range(len(pred_y)): 
                if pred_y[i] == label_y[i]:
                    self.cor_data.append(self.clean_data_for_repair[i])
                # else:
                    # pass
                    # self.incor_data.append(self.clean_data_for_repair[i])
        for step, (x, y) in enumerate(self.poi_data_for_repair_loader):
            train_x, train_y = x, y
            train_x = train_x.cuda()
            train_output = self.poisoned_model(train_x)
            train_output = train_output[-1]
            pred_y = torch.max(train_output.cpu(), 1)[1].numpy()
            for i in range(len(pred_y)): 
                if pred_y[i] == self.target and y[i] != self.target:
                    self.incor_data.append(self.poi_data_for_repair[i])
                if len(self.incor_data) > 127:
                    break
                if self.imagenet and len(self.incor_data) > 15:
                    break
                # elif pred_y[i] == y[i]:
                #     self.cor_data.append(self.poi_data_for_repair[i])
        print('cor data : ', len(self.cor_data))
        print('incor data : ', len(self.incor_data))
        self.cor_data_num = len(self.cor_data)
        self.incor_data_num = len(self.incor_data)

    def new_classification(self):
        self.cor_to_mis = []
        self.mis_to_mis = []
        for step, (x, y) in enumerate(self.clean_data_for_repair_loader):
            train_x, train_y = x, y
            train_x = train_x.cuda()
            train_output = self.poisoned_model(train_x)
            train_output = train_output[-1]
            pred_y = torch.max(train_output.cpu(), 1)[1].numpy()
            label_y = train_y.cpu().numpy()
        for i in range(len(pred_y)): 
            if pred_y[i] != label_y[i]:
                self.cor_to_mis.append(self.clean_data_for_repair[i])
        
        for step, (x, y) in enumerate(self.poi_data_for_repair_loader):
            train_x, train_y = x, y
            train_x = train_x.cuda()
            train_output = self.poisoned_model(train_x)
            train_output = train_output[-1]
            pred_y = torch.max(train_output.cpu(), 1)[1].numpy()
        for i in range(len(pred_y)): 
            if pred_y[i] == self.target and y[i] != self.target:
                self.mis_to_mis.append(self.poi_data_for_repair[i])
        if self.classifi == 0:
            self.cor_to_mis = []
            self.classifi = 1
        print('cor to mis data : ', len(self.cor_to_mis))
        print('mis to mis data : ', len(self.mis_to_mis))
        # self.cor_data_num = len(self.cor_data)
        self.incor_data_num = len(self.mis_to_mis)
        
    def analyze_neuron_action(self, check_layer, analyze_neuron_num):

        neuron_act_bound = torch.zeros((analyze_neuron_num, 2))
        
        image = []
        for i in range(self.N):
            x, y = self.poi_data_for_repair[i]
            image.append(x)

        for i in range(self.N_clean):
            x, y = self.clean_data_for_repair[i]
            image.append(x)
        image = torch.stack(image)
        print(image.shape)


        if torch.cuda.is_available():
            image = image.cuda()
            self.poisoned_model.cuda()

        all_output = self.poisoned_model(image)

        check_value = all_output[check_layer]
        if torch.cuda.is_available():
            check_value = check_value.cuda()
        neuron_act_bound[:, 0], index = torch.min(check_value, dim=0)
        neuron_act_bound[:, 1], index = torch.max(check_value, dim=0)

        return neuron_act_bound


    def compute_max_effect(self, round, check_layer, model, analyze_neuron_num, N, interval_num):
        with torch.no_grad():
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
            for i in range(len(self.incor_data)):
                x, y = self.incor_data[i]
                image.append(x)
                true_label.append(y)
            # for i in range(len(self.mis_to_mis)):
            #     x, y = self.mis_to_mis[i]
            #     image.append(x)
            #     true_label.append(y)
            image = torch.stack(image)
            # print(image.shape)


            if torch.cuda.is_available():
                image = image.cuda()
                model = model.cuda()
                self.poisoned_model.cuda()
                
            
            all_output = self.poisoned_model(image)

            image = []
            # print(all_output)
            pred_y = torch.max(all_output[-1].cpu(), 1)[1]
            # print(true_label, pred_y, self.target)
            # print(all_output[-1])
            true_input = all_output[check_layer]
            # print(true_input)
            if torch.cuda.is_available():
                true_input = true_input.cuda()

            lirpa_model = BoundedModule(model, torch.empty_like(true_input), device=true_input.device)
            # print('Running on', true_input.device)
            neuron_num = len(true_input[0])
            neuron_act_bound = self.analyze_neuron_action(check_layer=check_layer, analyze_neuron_num=analyze_neuron_num)

            eps_record = torch.zeros((analyze_neuron_num, interval_num, 2)).cuda()
            for neuron in range(neuron_num):
                diff = neuron_act_bound[neuron][1] - neuron_act_bound[neuron][0]
                st = neuron_act_bound[neuron][0] - diff
                if self.interval_times == 1:
                    st = neuron_act_bound[neuron][0] - diff


                diff *= (self.interval_times)
                
                # st = -self.interval_times * neuron_act_bound[neuron][1]
                # diff = self.interval_times * neuron_act_bound[neuron][1] * 2
                steps = diff / interval_num

                if self.imagenet:
                    st = neuron_act_bound[neuron][0] 
                for i in range(interval_num):
                    eps_record[neuron][i][0] = st + steps * i
                    eps_record[neuron][i][1] = st + steps * (i + 1)

            C = torch.zeros(size=(self.incor_data_num, 1, self.n_classes), device=true_input.device)
            for i in range(self.incor_data_num):
                C[i][0][true_label[i]] = 1.0
                C[i][0][pred_y[i]] = -1.0
                
            ori = torch.zeros((self.incor_data_num)).cuda()
            for i in range(self.incor_data_num):
                ori[i] = all_output[-1][i][true_label[i]] - all_output[-1][i][pred_y[i]]
            # print(ori)

            for neuron in tqdm(range(neuron_num)):
                # print('Now analyze neuron:', neuron)
                # print('-' * 70, 'one neuron start ', time.time() - t)
                
                eps = eps_record[neuron]
                if sum(eps_record[neuron][:, 0]) == sum(eps_record[neuron][:, 1]) and False:
                    all_neuron_eps_effect[neuron, :, :, :] = 0
                else:
                    true_input_L = true_input.detach().clone()
                    true_input_U = true_input.detach().clone()
                    # print(eps)
                    for interval in range(len(eps)):
                        
                        # true_input_L[:, neuron] = true_input[:, neuron] + eps[interval][0]
                        # true_input_U[:, neuron] = true_input[:, neuron] + eps[interval][1]
                        true_input_L[:, neuron] = eps[interval][0]
                        true_input_U[:, neuron] = eps[interval][1]
                        ptb = PerturbationLpNorm(x_L=true_input_L.detach().clone(), x_U=true_input_U.detach().clone())
                        true_input = BoundedTensor(true_input.detach().clone(), ptb)
                        required_A = defaultdict(set)
                        required_A[lirpa_model.output_name[0]].add(lirpa_model.input_name[0])
                        for method in self.approximate_method:
                            if 'Optimized' in method:
                                lirpa_model.set_bound_opts({'optimize_bound_args': {'iteration': 20, 'lr_alpha': 0.1, }})

                                    
                            lb, ub, A_dict = lirpa_model.compute_bounds(x=(true_input,), method=method.split()[0], return_A=True,
                                                                            needed_A_dict=required_A, C=C)
                            l_A, l_bias = A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['lA'], \
                                                    A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['lbias']

                            # u_A, u_bias = A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['uA'], \
                            #                         A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['ubias']
                            # l_A shape:  self.incor_data_num * 1 * neurons
                            # print(l_A.shape)
                            # l_bias shape:  self.incor_data_num * 1
                            # print(l_bias.shape)

                                
                            # interval_l_bound_L interval_l_bound_U shape ---- [self.incor_data_num]
                            interval_l_bound_L = torch.sum(true_input_L * l_A[:, 0, :], dim=1) + l_bias[:, 0]
                            interval_l_bound_U = torch.sum(true_input_U * l_A[:, 0, :], dim=1) + l_bias[:, 0]
                            # shape = neuron_num * self.incor_data_num * interval_num * 2
                            all_neuron_eps_effect[neuron, :, interval, 0] = (interval_l_bound_L - ori).detach().clone()
                            all_neuron_eps_effect[neuron, :, interval, 1] = (interval_l_bound_U - ori).detach().clone()

                
                    for i in range(self.incor_data_num):
                        all_neuron_effect[neuron][i] = max(0, torch.max(all_neuron_eps_effect[neuron][i]))
                # print('-' * 70, 'one neuron end ', time.time() - t)
            # print('-' * 70, 'all neuron end ', time.time() - t)
            for i in range(self.incor_data_num):
                # eff = all_neuron_effect[:, i].detach().clone()

                max_effect, index = torch.max(all_neuron_effect[:, i], dim=0)
                # print(max_effect, index)
                min_effect, index = torch.min(all_neuron_effect[:, i], dim=0)
                # print(min_effect, index)
                if max_effect - min_effect == 0:
                    all_neuron_effect[:, i] = 0
                else:
                    all_neuron_effect[:, i] -= min_effect
                    all_neuron_effect[:, i] /= (max_effect - min_effect)

                        
            # print('-' * 70, 'normalize end ', time.time() - t)
            for neuron in range(neuron_num):
                all_neuron_effect[neuron][self.incor_data_num] = torch.sum(all_neuron_effect[neuron][0: N])
                all_input_effect[neuron] = all_neuron_effect[neuron][self.incor_data_num]
            # print(all_input_effect)
            sorted_effect, sorted_index = torch.sort(all_input_effect, descending=True)
            all_neuron_effect = all_neuron_effect.tolist()
            all_neuron_effect.sort(key=lambda x:x[-1], reverse=True)
            # print('-' * 70, 'sort end ', time.time() - t)
            if check_layer not in self.loc_time:
                self.loc_time[check_layer] = 0
            self.loc_time[check_layer] += time.time() - t

        repair_neuron = sorted_index[0]
        self.way.append(repair_neuron)
        self.new_repair(check_layer=check_layer, model=model, repair_interval=all_neuron_eps_effect[repair_neuron],\
                                analyze_neuron=repair_neuron, ori=ori, eps_record=eps_record[repair_neuron], local_epoch=self.local_epoch)
        acc = self.Test_acc(self.poisoned_model, self.clean_data_for_repair_loader, 'Poisoned Model')
        sr, succ = self.Test_SR(self.poisoned_model, self.poi_data_for_repair_loader, 'Poisoned Model')      

        print('After repair {} '.format(repair_neuron), 'the model acc : ', acc, ' sr : ', sr)
        # return sorted_effect, sorted_index, all_neuron_effect, all_neuron_eps_effect, eps_record
        return acc, sr, succ



    def compute_max_effect_batch(self, round, check_layer, model, analyze_neuron_num, N, interval_num):
        with torch.no_grad():
            # print('Now we start analyze the model, compute max effect for every neuron.')
            
            all_input_effect = torch.zeros((analyze_neuron_num)).cuda()
            all_neuron_effect = torch.zeros((analyze_neuron_num, self.incor_data_num + 1)).cuda()
            # shape = neuron_num * N * interval_num * 2
            all_neuron_eps_effect = torch.zeros((analyze_neuron_num, self.incor_data_num, interval_num, 2)).cuda()
            all_ori = torch.zeros((self.incor_data_num)).cuda()
            # print(all_input_effect.device, all_neuron_effect.device, all_neuron_eps_effect.device)
            t = time.time()
            neuron_act_bound = self.analyze_neuron_action(check_layer=check_layer, analyze_neuron_num=analyze_neuron_num)

            if len(self.incor_data) == 0:
                return 0, 0
            Batch = 20
            round = self.incor_data_num // Batch
            if self.incor_data_num % Batch != 0:
                round += 1
            for step in range(round):
                image = []
                true_label = []
                for i in range(Batch):
                    # print(step * Batch + i)
                    if step * Batch + i >= len(self.incor_data):
                        break
                    x, y = self.incor_data[step * Batch + i]
                    image.append(x)
                    true_label.append(y)
                # for i in range(len(self.mis_to_mis)):
                #     x, y = self.mis_to_mis[i]
                #     image.append(x)
                #     true_label.append(y)
                image = torch.stack(image)
                # print(image.shape)
                if torch.cuda.is_available():
                    image = image.cuda()
                    model = model.cuda()
                    self.poisoned_model.cuda()
                
                all_output = self.poisoned_model(image)


                # print(all_output)
                pred_y = torch.max(all_output[-1].cpu(), 1)[1]
                # print(true_label, pred_y, self.target)
                # print(all_output[-1])
                true_input = all_output[check_layer]
                # print(true_input)
                if torch.cuda.is_available():
                    true_input = true_input.cuda()

                lirpa_model = BoundedModule(model, torch.empty_like(true_input), device=true_input.device)
                print('Running on', true_input.device)
                neuron_num = len(true_input[0])

                eps_record = torch.zeros((analyze_neuron_num, interval_num, 2)).cuda()
                for neuron in range(neuron_num):
                    diff = neuron_act_bound[neuron][1] - neuron_act_bound[neuron][0]
                    st = neuron_act_bound[neuron][0] - diff
                    diff *= (self.interval_times)
                    steps = diff / interval_num
                    # cifar
                    # if self.interval_times == 1:
                    #     st = neuron_act_bound[neuron][0] - diff
                    # else:
                    #     st = neuron_act_bound[neuron][0] 
                    for i in range(interval_num):
                        eps_record[neuron][i][0] = st + steps * i
                        eps_record[neuron][i][1] = st + steps * (i + 1)
                C = torch.zeros(size=(len(image), 1, self.n_classes), device=true_input.device)
                for i in range(len(image)):
                    C[i][0][true_label[i]] = 1.0
                    C[i][0][pred_y[i]] = -1.0
                    # print('-'*100, i, true_label[i], pred_y[i])

                ori = torch.zeros((len(image))).cuda()
                for i in range(len(image)):
                    ori[i] = all_output[-1][i][true_label[i]] - all_output[-1][i][pred_y[i]]
                    all_ori[step * Batch + i] = all_output[-1][i][true_label[i]] - all_output[-1][i][pred_y[i]]
                # print(ori)

                for neuron in tqdm(range(neuron_num)):
                    # print('Now analyze neuron:', neuron)
                    # print('-' * 70, 'one neuron start ', time.time() - t)
                    
                    eps = eps_record[neuron]
                    if sum(eps_record[neuron][:, 0]) == sum(eps_record[neuron][:, 1]) and False:
                        all_neuron_eps_effect[neuron, :, :, :] = 0
                    else:
                        true_input_L = true_input.detach().clone()
                        true_input_U = true_input.detach().clone()
                        # print(eps)
                        for interval in range(len(eps)):
                            true_input_L[:, neuron] = eps[interval][0]
                            true_input_U[:, neuron] = eps[interval][1]
                            ptb = PerturbationLpNorm(x_L=true_input_L.detach().clone(), x_U=true_input_U.detach().clone())
                            true_input = BoundedTensor(true_input.detach().clone(), ptb)
                            required_A = defaultdict(set)
                            required_A[lirpa_model.output_name[0]].add(lirpa_model.input_name[0])
                            for method in self.approximate_method:
                                if 'Optimized' in method:
                                    lirpa_model.set_bound_opts({'optimize_bound_args': {'iteration': 20, 'lr_alpha': 0.1, }})


                                lb, ub, A_dict = lirpa_model.compute_bounds(x=(true_input,), method=method.split()[0], return_A=True,
                                                                                needed_A_dict=required_A, C=C)
                                l_A, l_bias = A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['lA'], \
                                                        A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['lbias']

                                # u_A, u_bias = A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['uA'], \
                                #                         A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['ubias']
                                # l_A shape:  self.incor_data_num * 1 * neurons
                                # print(l_A.shape)
                                # l_bias shape:  self.incor_data_num * 1
                                # print(l_bias.shape)

                                    
                                # interval_l_bound_L interval_l_bound_U shape ---- [self.incor_data_num]
                                interval_l_bound_L = torch.sum(true_input_L * l_A[:, 0, :], dim=1) + l_bias[:, 0]
                                interval_l_bound_U = torch.sum(true_input_U * l_A[:, 0, :], dim=1) + l_bias[:, 0]
                                # shape = neuron_num * self.incor_data_num * interval_num * 2
                                all_neuron_eps_effect[neuron, step * Batch: (step + 1 ) * Batch, interval, 0] = (interval_l_bound_L - ori).detach().clone()
                                all_neuron_eps_effect[neuron, step * Batch: (step + 1 ) * Batch, interval, 1] = (interval_l_bound_U - ori).detach().clone()

                    
                        for i in range(step * Batch, step * Batch + len(image)):
                            all_neuron_effect[neuron][i] = max(0, torch.max(all_neuron_eps_effect[neuron][i]))
                    # print('-' * 70, 'one neuron end ', time.time() - t)
                # print('-' * 70, 'all neuron end ', time.time() - t)
                for i in range(step * Batch,  step * Batch + len(image)):
                    # eff = all_neuron_effect[:, i].detach().clone()
                    max_effect, index = torch.max(all_neuron_effect[:, i], dim=0)
                    
                    min_effect, index = torch.min(all_neuron_effect[:, i], dim=0)
                    
                    if max_effect - min_effect == 0:
                        all_neuron_effect[:, i] = 0
                    else:
                        all_neuron_effect[:, i] -= min_effect
                        all_neuron_effect[:, i] /= (max_effect - min_effect)

                            
                # print('-' * 70, 'normalize end ', time.time() - t)
            for neuron in range(analyze_neuron_num):
                all_neuron_effect[neuron][self.incor_data_num] = torch.sum(all_neuron_effect[neuron][0: N])
                all_input_effect[neuron] = all_neuron_effect[neuron][self.incor_data_num]
            # print(all_input_effect)
            sorted_effect, sorted_index = torch.sort(all_input_effect, descending=True)
            all_neuron_effect = all_neuron_effect.tolist()
            all_neuron_effect.sort(key=lambda x:x[-1], reverse=True)
            # print('-' * 70, 'sort end ', time.time() - t)
            if check_layer not in self.loc_time:
                self.loc_time[check_layer] = 0
            self.loc_time[check_layer] += time.time() - t

        repair_neuron = sorted_index[0]
        self.way.append(repair_neuron)
        self.new_repair(check_layer=check_layer, model=model, repair_interval=all_neuron_eps_effect[repair_neuron],\
                                analyze_neuron=repair_neuron, ori=all_ori, eps_record=eps_record[repair_neuron], local_epoch=self.local_epoch)
        acc = self.Test_acc(self.poisoned_model, self.clean_data_for_repair_loader, 'Poisoned Model')
        sr, succ = self.Test_SR(self.poisoned_model, self.poi_data_for_repair_loader, 'Poisoned Model')      

        return acc, sr, succ


    def new_repair(self, check_layer, model, repair_interval, analyze_neuron, ori, eps_record, local_epoch):
        print('Now we start to repair the model locally, correcting the target neuron (repair the in-weight) of {} input'.format(self.N))

        t = time.time()
        image = []
        true_label = []
        for data in self.incor_data:
            # cifar数据集返回真实标签
            x, y = data[0], data[1]
            image.append(x)
            true_label.append(y)
        for data in self.cor_data:
            x, y = data[0], data[1]
            image.append(x)
        image = torch.stack(image)
        # print(image.shape, image.shape[0])
        input_num = image.shape[0]

        if torch.cuda.is_available():
            image = image.cuda()
            model = model.cuda()
            self.poisoned_model.cuda()
            
        all_output = self.poisoned_model(image)

        # print(all_output[-1])
        pre_layer_value = all_output[check_layer - 1]
        check_layer_value = all_output[check_layer]
        # before repair
        check_neuron_value = check_layer_value[:, analyze_neuron]
        # print(pre_layer_value.shape)
        # print(check_layer_value.shape)
        # print(check_neuron_value)
        in_weight_num = len(pre_layer_value[0])

        #mini-nn label
        mini_label = torch.empty([self.cor_data_num + self.incor_data_num, 2], requires_grad=False).cuda()
        for i in range(self.incor_data_num):
            m = 0
            
            mini_label[i] = check_neuron_value[i]
            
            f = 'not entirely'
            point_set = []
            
            for j1 in range(repair_interval[i].shape[0]):
                if repair_interval[i][j1][0] + ori[i] > 0  and repair_interval[i][j1][1] + ori[i] > 0:
                    f = 'entirely'
                    point_set.append(eps_record[j1])

            
            if f == 'not entirely':
                
                for j in range(repair_interval[i].shape[0]):
                    m1, index = torch.max(repair_interval[i][j], dim=0)
                    if m1 > m:
                        m = m1
                        mini_label[i][0] = eps_record[j][index]
                        mini_label[i][1] = eps_record[j][index]
            else:
                opt = 10**9
                for j in point_set:
                    d0 = abs(j[0] - check_neuron_value[i])
                    d1 = abs(j[1] - check_neuron_value[i])
                    if min(d0, d1) < opt:
                        opt = min(d0, d1)
                        mini_label[i][0] = j[0]
                        mini_label[i][1] = j[1]
            
        mini_label[self.incor_data_num:, 0] = check_neuron_value[self.incor_data_num:]
        mini_label[self.incor_data_num:, 1] = check_neuron_value[self.incor_data_num:]
        mini_label = mini_label.detach()
        # print(eps_record)

        # print(self.poisoned_model.state_dict().keys())
        # print(model.state_dict().keys())
        d = len(self.poisoned_model.state_dict().keys()) - len(model.state_dict().keys())

        para_name = list(self.poisoned_model.state_dict().keys())
        pre_check_wegiht = self.poisoned_model.state_dict()[para_name[d - 2]]
        pre_check_bias = self.poisoned_model.state_dict()[para_name[d - 1]]
        
        post_check_wegiht = self.poisoned_model.state_dict()[para_name[d]]
        post_check_bias = self.poisoned_model.state_dict()[para_name[d + 1]]
        # print('-', post_check_wegiht.shape, post_check_bias.shape)
        # weight size: check layer * (check layer - 1)
        # print(pre_check_wegiht, pre_check_wegiht.shape)
        # print(pre_check_bias, pre_check_bias.shape)
        
        if check_layer == 0:
            #mini-nn input
            ratio = 10
            if self.interval_times == 1:
                ratio = 10
            else:
                ratio = 5
            # ratio = 1
            if self.incor_data_num * ratio == self.cor_data_num:
                mini_input = torch.empty([self.cor_data_num + self.incor_data_num, 1], requires_grad=False).cuda()
                mini_input[:, 0] = check_neuron_value.detach()
            elif self.incor_data_num * ratio < self.cor_data_num:
                print('&' * 20, 'cilp!')
                mini_input = torch.empty([(ratio + 1) * self.incor_data_num, 1], requires_grad=False).cuda()
                mini_input[:, 0] = check_neuron_value[0: (ratio + 1) * self.incor_data_num].detach()
                mini_label = mini_label[0: (ratio + 1) * self.incor_data_num].detach()
            else:
                print('&' * 20, 'cilp!')
                mini_input = torch.empty([(ratio + 1) * self.incor_data_num, 1], requires_grad=False).cuda()
                mini_input[:, 0] = check_neuron_value[0: (ratio + 1) * self.incor_data_num].detach()
                mini_label = mini_label[0: (ratio + 1) * self.incor_data_num].detach()
            print(mini_input.shape, mini_label.shape)
            print('mis data, all data', self.incor_data_num, mini_label.shape[0])
            mini_data = Data(mini_input, mini_label)
            mini_loader = DataLoader(dataset=mini_data, batch_size=self.BATCH_SIZE, shuffle=True)


            mini_nn = nn.Sequential(
                nn.Linear(1, 1)
            )
            # print(mini_nn.state_dict().keys())
            para = {}
            para['0.weight'] = torch.tensor(1).view(1, 1)
            para['0.bias'] = torch.tensor(0).view(1)
            mini_nn.load_state_dict(para)
            mini_nn.cuda()
            mini_nn.train()
            # using adam 0.01
            optim = torch.optim.Adam(mini_nn.parameters(), lr=0.001)
            # if self.imagenet:
            #     optim = torch.optim.SGD(mini_nn.parameters(), lr=0.01)
            start = time.time()
            print('-' * 70, 'other time ', start - t)
            # print(mini_label)

            for epoch in range(local_epoch):
                for step,(x,y) in enumerate(mini_loader):

                    output = mini_nn(x)
                    output = output.squeeze(1)
                    # print(output, y)
                    l0 = output - y[:, 0]
                    l1 = output - y[:, 1]
                    # print(l0, l1)
                    loss = 0
                    for p in range(len(l0)):
                        if l0[p] > 0 and l1[p] > 0:
                            loss += l1[p] * l1[p]
                        elif l0[p] < 0 and l1[p] < 0:
                            loss += l0[p] * l0[p]
                        else:
                            loss += 0
                        # print(p, loss)
                    # loss = (l0 * l1).relu()
                    # print(loss)
                    # loss = torch.mean(loss)
                    loss /= len(l0)
                    if loss == 0:
                        continue
                    # print('epoch: %s   step: %s   loss: %s   time: %s'%(epoch, step, loss.item(), time.time() - start))
                    optim.zero_grad()
                    loss.backward()  
                    optim.step()
                # print(output,y)

            b = mini_nn.state_dict()['0.bias']
            w = mini_nn.state_dict()['0.weight'][0]
            
            post_check_bias += b * post_check_wegiht[:, analyze_neuron]
            post_check_wegiht[:, analyze_neuron] *= w
            print('-', post_check_wegiht.shape, post_check_bias.shape)
            repaired_para = {}
            for key in self.poisoned_model.state_dict().keys():
                repaired_para[key] = self.poisoned_model.state_dict()[key]
            repaired_para[para_name[d]] = post_check_wegiht
            repaired_para[para_name[d + 1]] = post_check_bias
        else:
            #mini-nn input
            ratio = 10
            if self.imagenet:
                ratio = 3
            if self.incor_data_num * ratio == self.cor_data_num:
                mini_input = torch.empty([self.cor_data_num + self.incor_data_num, in_weight_num], requires_grad=False).cuda()
                mini_input = pre_layer_value.detach()
            elif self.incor_data_num * ratio < self.cor_data_num:
                # print('&' * 20, 'cilp!')
                mini_input = torch.empty([(ratio + 1) * self.incor_data_num, in_weight_num], requires_grad=False).cuda()
                mini_input = pre_layer_value[0: (ratio + 1) * self.incor_data_num].detach()
                mini_label = mini_label[0: (ratio + 1) * self.incor_data_num].detach()
            else:
                # print('&' * 20, 'cilp!')
                mini_input = torch.empty([(ratio + 1) * self.incor_data_num, in_weight_num], requires_grad=False).cuda()
                mini_input = pre_layer_value[0: (ratio + 1) * self.incor_data_num].detach()
                mini_label = mini_label[0: (ratio + 1) * self.incor_data_num].detach()
                # mini_input = torch.empty([(1 / ratio + 1) * self.cor_data_num, in_weight_num], requires_grad=False).cuda()
                # mini_input[(1 / ratio) * self.cor_data_num:] = pre_layer_value[]
                    
                # mini_input = pre_layer_value[0: (1 / ratio) * self.cor_data_num].detach()
                
                # mini_label = mini_label[0: (ratio + 1) * self.incor_data_num].detach()
            # print(mini_input.shape, mini_label.shape)
            # print('mis data, all data', self.incor_data_num, mini_label.shape[0])
            mini_data = Data(mini_input, mini_label)
            mini_loader = DataLoader(dataset=mini_data, batch_size=self.BATCH_SIZE, shuffle=True)

            check_neuron_in_weight = pre_check_wegiht[analyze_neuron]
            check_neuron_bias = pre_check_bias[analyze_neuron]
            # print(check_neuron_in_weight.shape, check_neuron_bias)
            mini_nn = nn.Sequential(
                nn.Linear(in_weight_num, 1),
                nn.ReLU(),
                nn.Linear(1, 1)
            )
            # print(mini_nn.state_dict().keys())
            para = {}
            para['0.weight'] = check_neuron_in_weight.view(1, in_weight_num)
            para['0.bias'] = check_neuron_bias.view(1)
            para['2.weight'] = torch.tensor(1).view(1, 1)
            para['2.bias'] = torch.tensor(0).view(1)
            mini_nn.load_state_dict(para)
            mini_nn.cuda()
            mini_nn.train()
            
            # using adam 0.01
            optim = torch.optim.Adam(mini_nn.parameters(), lr=0.01)
            if self.imagenet:
                optim = torch.optim.Adam(mini_nn.parameters(), lr=0.001)
            loss_fcn = nn.MSELoss()
            start = time.time()

            for epoch in range(local_epoch):
                for step,(x,y) in enumerate(mini_loader):
                    output = mini_nn(x)
                    output = output.squeeze(1)
                    # print(output, y)
                    l0 = output - y[:, 0]
                    l1 = output - y[:, 1]
                    # print(l0, l1)
                    loss = 0
                    for p in range(len(l0)):
                        if l0[p] > 0 and l1[p] > 0:
                            loss += l1[p] * l1[p]
                        elif l0[p] < 0 and l1[p] < 0:
                            loss += l0[p] * l0[p]
                        else:
                            loss += 0
                    loss /= len(l0)
                    if loss == 0:
                        continue
                    # print('epoch: %s   step: %s   loss: %s   time: %s'%(epoch, step, loss.item(), time.time() - start))
                    optim.zero_grad()  
                    loss.backward()  
                    optim.step()
            repaired_para = {}
            for key in self.poisoned_model.state_dict().keys():
                repaired_para[key] = self.poisoned_model.state_dict()[key]
            b = mini_nn.state_dict()['2.bias']
            w = mini_nn.state_dict()['2.weight'][0]
            repaired_para[para_name[d - 2]][analyze_neuron] = mini_nn.state_dict()['0.weight']
            # print('pre weight shape:', repaired_para[para_name[d - 2]][analyze_neuron].shape, mini_nn.state_dict()['0.weight'].shape)
            repaired_para[para_name[d - 1]][analyze_neuron] = mini_nn.state_dict()['0.bias']
            # print('pre bias shape:', repaired_para[para_name[d - 1]][analyze_neuron].shape, mini_nn.state_dict()['0.bias'].shape)
            repaired_para[para_name[d + 1]] += b * post_check_wegiht[:, analyze_neuron]
            # print('post weight shape:', repaired_para[para_name[d + 1]].shape, post_check_wegiht[:, analyze_neuron].shape)
            # print('post bias shape:', repaired_para[para_name[d]].shape)
            # repaired_para[para_name[d]] *= w
            repaired_para[para_name[d]][:, analyze_neuron] *= w
            
        if check_layer not in self.repair_time:
            self.repair_time[check_layer] = 0
        self.repair_time[check_layer] += time.time() - t
        self.poisoned_model.load_state_dict(repaired_para)



    def analyze_weight(self, check_layer, model, analyze_weight_num, analyze_neuron, N, project, example_type):
        
        print('Now we start analyze the weight (between check_layer - 1 and check_layer), compute value flowing through this edge of {} input'.format(N))

        print(model.state_dict().keys())

        t = time.time()
       
        if 'mnist' in project:
            if example_type != 'clean':
                image = self.poisoned_test_data[0: N][0]
            else:
                image = self.clean_test_data[0: N][0]
            image = torch.stack(image)
            print(image.shape)
            target_label = self.target
            true_label = self.clean_test_data.targets[0: N]
            print(true_label)
            image = torch.tensor(image).view(N, 1, 28, 28)
        elif 'cifar' in project:
            image = []
            true_label = torch.empty([N], dtype=int)
            for i in range(N):
                if example_type != 'clean':
                    x, y = self.poisoned_test_data[i]
                else:
                    x, y = self.clean_test_data[i]
                image.append(x)
                true_label[i] = y
            image = torch.stack(image)
            print(image.shape)
            target_label = self.target
            print(true_label)
            # print(image.shape)
            image = torch.tensor(image).view(N, 3, 32, 32)
        target_label = torch.tensor(target_label).view(1, 1)


        if torch.cuda.is_available():
            image = image.cuda()
            model = model.cuda()
        self.poisoned_model.cuda()

        all_output = self.poisoned_model(image)

        # print(all_output[-1])
        pre_layer_value = all_output[check_layer - 1]
        check_layer_value = all_output[check_layer]
        check_neuron_value = check_layer_value[:, analyze_neuron]
        print(pre_layer_value.shape)
        print(check_layer_value.shape)
        print(check_neuron_value.shape)


        weight_num = len(pre_layer_value[0])

        for key in self.poisoned_model.state_dict().keys():
            if key not in model.state_dict().keys():
                if 'weight' in key:
                    pre_check_wegiht = self.poisoned_model.state_dict()[key]
                elif 'bias' in key:
                    pre_check_bias = self.poisoned_model.state_dict()[key]
            else:
                break

        # weight size: check layer * (check layer - 1)
        # print(pre_check_wegiht, pre_check_wegiht.shape)
        # print(pre_check_bias, pre_check_bias.shape)
        # print(model.state_dict().keys())

        check_neuron_in_weight = pre_check_wegiht[analyze_neuron]
        print(check_neuron_in_weight.shape, check_neuron_in_weight)
        for i in range(len(pre_layer_value[-1])):
            if pre_layer_value[-1][i] != 0:
                print(i, pre_layer_value[0][i])
        
        
        all_weight_flow = pre_layer_value * check_neuron_in_weight
        print(all_weight_flow[-1].shape, torch.sum(all_weight_flow[-1]))

        
        weight_flow_ratio = torch.zeros((analyze_weight_num, N + 1))
        for i in range(len(check_neuron_value)):
            if check_neuron_value[i] > 0:
                weight_flow_ratio[:, i] = all_weight_flow[i] 
                
            else:
                 weight_flow_ratio[:, i] = 0
            weight_flow_ratio[:, -1] += weight_flow_ratio[:, i]
        
        sorted_ratio, sorted_index = torch.sort(weight_flow_ratio[:, -1], descending=True)
        weight_flow_ratio = weight_flow_ratio.tolist()
        weight_flow_ratio.sort(key=lambda x:x[-1], reverse=True)
        result = np.empty([analyze_weight_num, N + 2]) 
        for i in range(analyze_weight_num):
            result[i][0] = sorted_index[i]
            result[i][1] = sorted_ratio[i]
            for j in range(N):
                result[i][j + 2] = weight_flow_ratio[i][j]
        result_dir = 'weight loc result/result of ' + project + '_' + example_type + '_input' + str(N) +\
                                 '_Check ' + str(check_layer) + '_Neuron ' + str(analyze_neuron) + '.txt'
        np.savetxt(result_dir, result, fmt='%.03f')
        return 0


    def coefficients_compute(self, neuron, lirpa_model, true_input, eps, target_label):
        true_input_L = true_input.clone().detach()
        true_input_U = true_input.clone().detach()
        true_input_L[0][neuron] -= eps
        true_input_U[0][neuron] += eps

        
        ptb = PerturbationLpNorm(x_L=true_input_L, x_U=true_input_U)
            
        true_input = BoundedTensor(true_input, ptb)
        # Get model prediction as usual
        pred = lirpa_model(true_input)
        # print('model pred = ', pred)

        label = torch.argmax(pred, dim=1).cpu().detach().numpy()

        required_A = defaultdict(set)
        required_A[lirpa_model.output_name[0]].add(lirpa_model.input_name[0])
        for method in self.approximate_method:
            # print("Bounding method:", method)
            if 'Optimized' in method:
                # For optimized bound, you can change the number of iterations, learning rate, etc here. Also you can increase verbosity to see per-iteration loss values.
                lirpa_model.set_bound_opts({'optimize_bound_args': {'iteration': 20, 'lr_alpha': 0.1}})
            
            lb, ub, A_dict = lirpa_model.compute_bounds(x=(true_input,), method=method.split()[0], return_A=True,
                                                        needed_A_dict=required_A)
            
            lower_A, lower_bias = A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['lA'], \
                                A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['lbias']
            upper_A, upper_bias = A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['uA'], \
                                A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['ubias']
            target_label = target_label[0][0]
    
            # print('lower max', torch.max(lower_A[0][target_label]))
            # print('lower max', torch.argmax(lower_A[0][target_label]))
            # print('upper max', torch.max(upper_A[0][target_label]))
            # print('upper max', torch.argmax(upper_A[0][target_label]))
            # print('Lower', torch.sort(lower_A[0][target_label], descending=True))
            # print('Lower index', torch.argsort(lower_A[0][target_label], descending=True))
            # print('Upper', torch.sort(upper_A[0][target_label], descending=True))
            # print('Upper index', torch.argsort(upper_A[0][target_label], descending=True))
            # print(lower_A[0][target_label][neuron])
            # print(upper_A[0][target_label][neuron])
            # print(f'lower bound linear coefficients size (batch, output_dim, *input_dims): {list(lower_A.size())}')
            # print(f'lower bound linear coefficients norm (smaller is better): {lower_A.norm()}')
            # print(f'lower bound bias term size (batch, output_dim): {list(lower_bias.size())}')
            # print(f'lower bound bias term sum (larger is better): {lower_bias.sum()}')
            # print(f'upper bound linear coefficients size (batch, output_dim, *input_dims): {list(upper_A.size())}')
            # print(f'upper bound linear coefficients norm (smaller is better): {upper_A.norm()}')
            # print(f'upper bound bias term size (batch, output_dim): {list(upper_bias.size())}')
            # print(f'upper bound bias term sum (smaller is better): {upper_bias.sum()}')
            # print(f'These linear lower and upper bounds are valid everywhere within the perturbation radii.\n')
            
            # for i in range(10):
            #     print(lower_A[0][i][neuron], i, neuron, 'lower', '-------', upper_A[0][i][neuron], i, neuron, 'upper')
        return lower_A, lower_bias, upper_A, upper_bias

   
    def compute_max_effect_for_pso(self, round, check_layer, model, analyze_neuron_num, N, project, interval_num):
        with torch.no_grad():
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
            for i in range(len(self.incor_data)):
                x, y = self.incor_data[i]
                image.append(x)
                true_label.append(y)
            # for i in range(len(self.mis_to_mis)):
            #     x, y = self.mis_to_mis[i]
            #     image.append(x)
            #     true_label.append(y)
            image = torch.stack(image)
            # print(image.shape)


            if torch.cuda.is_available():
                image = image.cuda()
                model = model.cuda()
                self.poisoned_model.cuda()
                
            
            all_output = self.poisoned_model(image)

            image = []
            # print(all_output)
            pred_y = torch.max(all_output[-1].cpu(), 1)[1]
            true_input = all_output[check_layer]
            if torch.cuda.is_available():
                true_input = true_input.cuda()

            lirpa_model = BoundedModule(model, torch.empty_like(true_input), device=true_input.device)
            # print('Running on', true_input.device)
            neuron_num = len(true_input[0])
            neuron_act_bound = self.analyze_neuron_action(check_layer=check_layer, analyze_neuron_num=analyze_neuron_num)

            eps_record = torch.zeros((analyze_neuron_num, interval_num, 2)).cuda()
            for neuron in range(neuron_num):
                diff = neuron_act_bound[neuron][1] - neuron_act_bound[neuron][0]
                st = neuron_act_bound[neuron][0] - diff
                if self.interval_times == 1:
                    st = neuron_act_bound[neuron][0] - diff
                diff *= (self.interval_times)
                steps = diff / interval_num

                if self.imagenet:
                    st = neuron_act_bound[neuron][0] 
                for i in range(interval_num):
                    eps_record[neuron][i][0] = st + steps * i
                    eps_record[neuron][i][1] = st + steps * (i + 1)

            C = torch.zeros(size=(self.incor_data_num, 1, self.n_classes), device=true_input.device)
            for i in range(self.incor_data_num):
                C[i][0][true_label[i]] = 1.0
                C[i][0][pred_y[i]] = -1.0
                # print('-'*100, i, true_label[i], pred_y[i])

                            
            
            ori = torch.zeros((self.incor_data_num)).cuda()
            for i in range(self.incor_data_num):
                ori[i] = all_output[-1][i][true_label[i]] - all_output[-1][i][pred_y[i]]
            # print(ori)

            for neuron in tqdm(range(neuron_num)):
                # print('Now analyze neuron:', neuron)
                # print('-' * 70, 'one neuron start ', time.time() - t)
                
                eps = eps_record[neuron]
                

                
                if sum(eps_record[neuron][:, 0]) == sum(eps_record[neuron][:, 1]) and False:
                    all_neuron_eps_effect[neuron, :, :, :] = 0
                else:
                    true_input_L = true_input.detach().clone()
                    true_input_U = true_input.detach().clone()
                    # print(eps)
                    for interval in range(len(eps)):
                        
                        true_input_L[:, neuron] = eps[interval][0]
                        true_input_U[:, neuron] = eps[interval][1]
                        ptb = PerturbationLpNorm(x_L=true_input_L.detach().clone(), x_U=true_input_U.detach().clone())
                        true_input = BoundedTensor(true_input.detach().clone(), ptb)
                        required_A = defaultdict(set)
                        required_A[lirpa_model.output_name[0]].add(lirpa_model.input_name[0])
                        for method in self.approximate_method:
                            if 'Optimized' in method:
                                lirpa_model.set_bound_opts({'optimize_bound_args': {'iteration': 20, 'lr_alpha': 0.1, }})

                                    
                            lb, ub, A_dict = lirpa_model.compute_bounds(x=(true_input,), method=method.split()[0], return_A=True,
                                                                            needed_A_dict=required_A, C=C)
                            l_A, l_bias = A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['lA'], \
                                                    A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['lbias']

                            # u_A, u_bias = A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['uA'], \
                            #                         A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['ubias']
                            # l_A shape:  self.incor_data_num * 1 * neurons
                            # print(l_A.shape)
                            # l_bias shape:  self.incor_data_num * 1
                            # print(l_bias.shape)
          
                            # interval_l_bound_L interval_l_bound_U shape ---- [self.incor_data_num]
                            interval_l_bound_L = torch.sum(true_input_L * l_A[:, 0, :], dim=1) + l_bias[:, 0]
                            interval_l_bound_U = torch.sum(true_input_U * l_A[:, 0, :], dim=1) + l_bias[:, 0]
                            # shape = neuron_num * self.incor_data_num * interval_num * 2
                            all_neuron_eps_effect[neuron, :, interval, 0] = (interval_l_bound_L - ori).detach().clone()
                            all_neuron_eps_effect[neuron, :, interval, 1] = (interval_l_bound_U - ori).detach().clone()

                
                    for i in range(self.incor_data_num):
                        all_neuron_effect[neuron][i] = max(0, torch.max(all_neuron_eps_effect[neuron][i]))
                # print('-' * 70, 'one neuron end ', time.time() - t)
            # print('-' * 70, 'all neuron end ', time.time() - t)
            for i in range(self.incor_data_num):
                # eff = all_neuron_effect[:, i].detach().clone()

                max_effect, index = torch.max(all_neuron_effect[:, i], dim=0)
                # print(max_effect, index)
                min_effect, index = torch.min(all_neuron_effect[:, i], dim=0)
                # print(min_effect, index)
                if max_effect - min_effect == 0:
                    all_neuron_effect[:, i] = 0
                else:
                    all_neuron_effect[:, i] -= min_effect
                    all_neuron_effect[:, i] /= (max_effect - min_effect)

                        
            # print('-' * 70, 'normalize end ', time.time() - t)
            for neuron in range(neuron_num):
                all_neuron_effect[neuron][self.incor_data_num] = torch.sum(all_neuron_effect[neuron][0: N])
                all_input_effect[neuron] = all_neuron_effect[neuron][self.incor_data_num]
            # print(all_input_effect)
            sorted_effect, sorted_index = torch.sort(all_input_effect, descending=True)
            all_neuron_effect = all_neuron_effect.tolist()
            all_neuron_effect.sort(key=lambda x:x[-1], reverse=True)
            # print('-' * 70, 'sort end ', time.time() - t)
            if check_layer not in self.loc_time:
                self.loc_time[check_layer] = 0
            self.loc_time[check_layer] += time.time() - t

        repair_neuron = sorted_index[0]
        self.way.append(repair_neuron)
        self.split_model = model
        self.check_layer = check_layer
        self.pso_repair()
        
        d = len(self.poisoned_model.state_dict().keys()) - len(model.state_dict().keys())
        para_name = list(self.poisoned_model.state_dict().keys())
        repaired_para = {}
        for key in self.poisoned_model.state_dict().keys():
            repaired_para[key] = self.poisoned_model.state_dict()[key]
        repaired_para[para_name[d]][:, repair_neuron] *= torch.tensor(self.pso_repair_way[0])
        acc = self.Test_acc(self.poisoned_model, self.clean_data_for_repair_loader, 'Poisoned Model')
        sr, succ = self.Test_SR(self.poisoned_model, self.poi_data_for_repair_loader, 'Poisoned Model')      
        if check_layer not in self.repair_time:
            self.repair_time[check_layer] = 0
        self.repair_time[check_layer] += time.time() - t
        self.poisoned_model.load_state_dict(repaired_para)
        print('After repair {} '.format(repair_neuron), 'the model acc : ', acc, ' sr : ', sr)
        # return sorted_effect, sorted_index, all_neuron_effect, all_neuron_eps_effect, eps_record
        
        return acc, sr, succ



   
    def compute_max_effect_for_finetune(self, check_layer, model, analyze_neuron_num, N, project, interval_num):
        with torch.no_grad():
            print('Now we start analyze the model, compute max effect for every neuron.')
            all_input_effect = torch.zeros((analyze_neuron_num)).cuda()
            all_neuron_effect = torch.zeros((analyze_neuron_num, self.incor_data_num + 1)).cuda()
            all_neuron_eps_effect = torch.zeros((analyze_neuron_num, self.incor_data_num, interval_num, 2)).cuda()
            t = time.time()

            image = []
            true_label = []
            if len(self.incor_data) == 0:
                return 0, 0
            for i in range(len(self.incor_data)):
                x, y = self.incor_data[i]
                image.append(x)
                true_label.append(y)

            image = torch.stack(image)
            if torch.cuda.is_available():
                image = image.cuda()
                model = model.cuda()
                self.poisoned_model.cuda()
                
            all_output = self.poisoned_model(image)
            image = []
            pred_y = torch.max(all_output[-1].cpu(), 1)[1]
            true_input = all_output[check_layer]
            if torch.cuda.is_available():
                true_input = true_input.cuda()

            lirpa_model = BoundedModule(model, torch.empty_like(true_input), device=true_input.device)
            neuron_num = len(true_input[0])
            neuron_act_bound = self.analyze_neuron_action(check_layer=check_layer, analyze_neuron_num=analyze_neuron_num)

            eps_record = torch.zeros((analyze_neuron_num, interval_num, 2)).cuda()
            for neuron in range(neuron_num):
                diff = neuron_act_bound[neuron][1] - neuron_act_bound[neuron][0]
                st = neuron_act_bound[neuron][0] - diff
                if self.interval_times == 1:
                    st = neuron_act_bound[neuron][0] - diff
                diff *= (self.interval_times)
                steps = diff / interval_num

                if self.imagenet:
                    st = neuron_act_bound[neuron][0] 
                for i in range(interval_num):
                    eps_record[neuron][i][0] = st + steps * i
                    eps_record[neuron][i][1] = st + steps * (i + 1)

            C = torch.zeros(size=(self.incor_data_num, 1, self.n_classes), device=true_input.device)
            for i in range(self.incor_data_num):
                C[i][0][true_label[i]] = 1.0
                C[i][0][pred_y[i]] = -1.0

            ori = torch.zeros((self.incor_data_num)).cuda()
            for i in range(self.incor_data_num):
                ori[i] = all_output[-1][i][true_label[i]] - all_output[-1][i][pred_y[i]]
            for neuron in tqdm(range(neuron_num)):
                eps = eps_record[neuron]

                if sum(eps_record[neuron][:, 0]) == sum(eps_record[neuron][:, 1]) and False:
                    all_neuron_eps_effect[neuron, :, :, :] = 0
                else:
                    true_input_L = true_input.detach().clone()
                    true_input_U = true_input.detach().clone()
                    for interval in range(len(eps)):
                        true_input_L[:, neuron] = eps[interval][0]
                        true_input_U[:, neuron] = eps[interval][1]
                        ptb = PerturbationLpNorm(x_L=true_input_L.detach().clone(), x_U=true_input_U.detach().clone())
                        true_input = BoundedTensor(true_input.detach().clone(), ptb)
                        required_A = defaultdict(set)
                        required_A[lirpa_model.output_name[0]].add(lirpa_model.input_name[0])
                        for method in self.approximate_method:
                            if 'Optimized' in method:
                                lirpa_model.set_bound_opts({'optimize_bound_args': {'iteration': 20, 'lr_alpha': 0.1, }})

                                    
                            lb, ub, A_dict = lirpa_model.compute_bounds(x=(true_input,), method=method.split()[0], return_A=True,
                                                                            needed_A_dict=required_A, C=C)
                            l_A, l_bias = A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['lA'], \
                                                    A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['lbias']

                            interval_l_bound_L = torch.sum(true_input_L * l_A[:, 0, :], dim=1) + l_bias[:, 0]
                            interval_l_bound_U = torch.sum(true_input_U * l_A[:, 0, :], dim=1) + l_bias[:, 0]
                            all_neuron_eps_effect[neuron, :, interval, 0] = (interval_l_bound_L - ori).detach().clone()
                            all_neuron_eps_effect[neuron, :, interval, 1] = (interval_l_bound_U - ori).detach().clone()

                
                    for i in range(self.incor_data_num):
                        all_neuron_effect[neuron][i] = max(0, torch.max(all_neuron_eps_effect[neuron][i]))
            for i in range(self.incor_data_num):

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
        print('repair_neuron', repair_neuron)
        print(sorted_index)
        print(sorted_effect)
        self.check_layer = check_layer
        _, _, _ = self.rm_finetune(check_layer=check_layer, analyze_neuron=repair_neuron, iters=self.local_epoch)
        # _, _, _ = self.l2_finetune(check_layer=check_layer, analyze_neuron=repair_neuron, iters=self.local_epoch)
        acc = self.Test_acc(self.poisoned_model, self.clean_data_for_repair_loader, 'Poisoned Model')
        sr, succ = self.Test_SR(self.poisoned_model, self.poi_data_for_repair_loader, 'Poisoned Model')      

        print('After repair {} '.format(repair_neuron), 'the model acc : ', acc, ' sr : ', sr)
        # return sorted_effect, sorted_index, all_neuron_effect, all_neuron_eps_effect, eps_record
        
        return acc, sr, succ

    def rm_finetune(self, check_layer, analyze_neuron, iters):

        loss_ce = nn.CrossEntropyLoss()
        param = []
        self.poisoned_model.train()
        """
        init a mask with params as key, and a zeros_like params tensor as value
        """
        mask = {}
        for name, params in self.poisoned_model.named_parameters():
            mask[name] = params.new_zeros(params.size())
        
            
        
        for p in self.poisoned_model.named_parameters():
            # print(p[0])
            if 'dense' in p[0] and 'weight' in p[0]:
                if str(check_layer) in p[0]:
                    mask[p[0]][analyze_neuron, :] = 1
                elif str(check_layer + 1) in p[0]:
                    mask[p[0]][:, analyze_neuron] = 1
            if 'dense' in p[0] and 'bias' in p[0] and str(check_layer) in p[0]:
                mask[p[0]][analyze_neuron] = 1
        from utils.MaskOptimizer import ChildTuningAdamW
        lr = 0.001
        if len(self.poi_data_for_repair) < 100:
            lr = 0.001
            # if blend svhn 0.001
            lr = 0.0001
        optimizer = ChildTuningAdamW(
            self.poisoned_model.parameters(), lr=lr
        )
        optimizer.set_gradient_mask(mask)
        repair_dataset = torch.utils.data.ConcatDataset([self.clean_data_for_repair, self.poi_data_for_repair])
        print(len(repair_dataset))
        repair_loader = DataLoader(dataset=repair_dataset, batch_size=64, shuffle=False)
        repair_loader = DataLoader(dataset=self.poi_data_for_repair, batch_size=64, shuffle=False)
        logging.info('finetune_lr {:.3f}, epoch {}, data num: {}'.format(lr, iters, len(repair_loader.dataset)))
        for i in range(1, iters + 1):
            # print("iter  {}".format(i))
            repair_time_start = time.time()
            for index, (data, target) in enumerate(repair_loader):
                data, target = data.cuda(), target.cuda()
                outputs = self.poisoned_model(data)[-1]
                loss = loss_ce(outputs, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return 0, 0, 0 



    def l2_finetune(self, check_layer, analyze_neuron, iters):

        loss_ce = nn.CrossEntropyLoss()
        param = []
        self.poisoned_model.train()
        """
        init a mask with params as key, and a zeros_like params tensor as value
        """
        mask = {}
        for name, params in self.poisoned_model.named_parameters():
            mask[name] = params.new_zeros(params.size())
        
        for p in self.poisoned_model.named_parameters():
            # print(p[0])
            if 'dense' in p[0] and 'weight' in p[0]:
                if str(check_layer) in p[0]:
                    mask[p[0]][analyze_neuron, :] = 1
                elif str(check_layer + 1) in p[0]:
                    mask[p[0]][:, analyze_neuron] = 1
            if 'dense' in p[0] and 'bias' in p[0] and str(check_layer) in p[0]:
                mask[p[0]][analyze_neuron] = 1
        from utils.MaskOptimizer import ChildTuningAdamW
        if len(self.poi_data_for_repair) < 100:
            lr = 0.005
            # if blend svhn 0.001
            lr = 0.0005
        else:
            lr = 0.001
        optimizer = ChildTuningAdamW(
            self.poisoned_model.parameters(), lr=lr
        )
        lamda = 1
        optimizer.set_gradient_mask(mask)
        repair_dataset = torch.utils.data.ConcatDataset([self.clean_data_for_repair, self.poi_data_for_repair])
        print(len(repair_dataset))
        repair_loader = DataLoader(dataset=repair_dataset, batch_size=64, shuffle=False)
        logging.info('finetune_lr {:.3f}, epoch {}, data num: {}'.format(lr, iters, len(repair_loader.dataset)))
        for i in range(1, iters + 1):
            # print("iter  {}".format(i))
            repair_time_start = time.time()
            for index, (data, target) in enumerate(repair_loader):
                data, target = data.cuda(), target.cuda()
                outputs = self.poisoned_model(data)[-1]
                loss = loss_ce(outputs, target)
                reg_loss = 0
                for p in self.poisoned_model.named_parameters():
                    if 'dense' in p[0] and 'weight' in p[0]:
                        if str(check_layer) in p[0]:
                            reg_loss += torch.sum(p[1][analyze_neuron, :] * p[1][analyze_neuron, :])
                            mask[p[0]][analyze_neuron, :] = 1
                        elif str(check_layer + 1) in p[0]:
                            reg_loss += torch.sum(p[1][:, analyze_neuron] * p[1][:, analyze_neuron])
                            mask[p[0]][:, analyze_neuron] = 1
                loss += lamda * reg_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return 0, 0, 0 


    def pso_repair(self):
        # repair
        print('Start reparing...')
        print('alpha: {}'.format(self.alpha))
        options = {'c1': 0.41, 'c2': 0.41, 'w': 0.8}
        #'''# original
        optimizer = ps.single.GlobalBestPSO(n_particles=20, dimensions=1, options=options,
                                            bounds=([[-10.0], [10.0]]),
                                            init_pos=np.ones((20, 1), dtype=float), ftol=1e-3,
                                            ftol_iter=10)
        #'''

        # Perform optimization
        best_cost, best_pos = optimizer.optimize(self.pso_fitness_func, iters=100)

        # Obtain the cost history
        # print(optimizer.cost_history)
        # Obtain the position history
        # print(optimizer.pos_history)
        # Obtain the velocity history
        # print(optimizer.velocity_history)
        #print('neuron to repair: {} at layter: {}'.format(self.r_neuron, self.r_layer))
        #print('best cost: {}'.format(best_cost))
        #print('best pos: {}'.format(best_pos))

        self.pso_repair_way = best_pos

        return best_pos

    # optimization target perturbed sample has the same label as clean sample
    def pso_fitness_func(self, weight):

        result = []
        for i in range (0, int(len(weight))):
            r_weight =  weight[i]

            cost = self.pso_test_repair(r_weight)

            #print('cost: {}'.format(cost))

            result.append(cost)

        #print(result)
        return result

    def pso_test_repair(self, r_weight):

        #result = []
        tot_count = 0
        correct = 0
        # per particle
        
        # print('After Repair, Test acc Start!')
        # print('Test Input = ', len(self.clean_data_for_repair_loader.dataset))
        self.poisoned_model.eval()
        self.split_model.eval()
        self.split_model.cuda()
        repair_index = self.way[-1]
        for step, (x, y) in enumerate(self.clean_data_for_repair_loader):
            
            all_output = self.poisoned_model(x.cuda())
            ori_hidden_output = all_output[self.check_layer]
            after_repair_hidden_output = ori_hidden_output.clone().detach()
            after_repair_hidden_output[:, repair_index] *= r_weight[0]
            final_output = self.split_model(after_repair_hidden_output)
            pred_y = torch.max(final_output.cpu(), 1)[1].numpy()
            label_y = y.cpu().numpy()
            correct += (pred_y == label_y).sum()
        acc = correct / len(self.clean_data_for_repair_loader.dataset)
        # print('Acc = ', correct, '/', len(self.clean_data_for_repair_loader.dataset), \
        #       '=', correct / len(self.clean_data_for_repair_loader.dataset))


        success = 0
        # number of (target == true label)
        nocount = 0
        # print('After Repair, Test sr Start!')
        # print('Test Input = ', len(self.poi_data_for_repair_loader.dataset))
        count = {}
        for i in range(10):
            count[i] = {}
        cl = {}
        for step, (x, y) in enumerate(self.poi_data_for_repair_loader):
            all_output = self.poisoned_model(x.cuda())
            ori_hidden_output = all_output[self.check_layer]
            after_repair_hidden_output = ori_hidden_output.clone().detach()
            after_repair_hidden_output[:, repair_index] *= r_weight[0]
            final_output = self.split_model(after_repair_hidden_output)
            pred_y = torch.max(final_output.cpu(), 1)[1].numpy()
            label_y = y.cpu().numpy()
            success += ((pred_y == self.target) & (label_y != self.target)).sum()
            nocount += (label_y == self.target * np.ones(pred_y.shape)).sum()
            
            
        # print(nocount)
        sr = success / (len(self.poi_data_for_repair_loader.dataset) - nocount)
        # print('SR = ', success, '/', len(self.poi_data_for_repair_loader.dataset) - nocount, '=', sr)
        cost = (1.0 - self.alpha) * sr + self.alpha * (1 - acc)
        return cost

    def pso_test(self, r_weight, target):
        result = 0.0
        correct = 0.0
        tot_count = 0
        if len(self.rep_index) != 0:

            # per particle
            for idx in range(self.mini_batch):
                X_batch, Y_batch = self.gen.next()
                X_batch_perturbed = self.get_perturbed_input(X_batch)

                o_prediction = self.model1.predict(X_batch)
                p_prediction = self.model1.predict(X_batch_perturbed)

                _p_prediction = np.reshape(p_prediction, (len(p_prediction), -1))
                _o_prediction = np.reshape(o_prediction, (len(o_prediction), -1))

                l_shape = p_prediction.shape

                do_hidden = _p_prediction.copy()
                o_hidden = _o_prediction.copy()

                for i in range (0, len(self.rep_index)):
                    rep_idx = int(self.rep_index[i])
                    do_hidden[:, rep_idx] = (r_weight[i]) * _p_prediction[:, rep_idx]
                    o_hidden[:, rep_idx] = (r_weight[i]) * _o_prediction[:, rep_idx]

                p_prediction = self.model2.predict(do_hidden.reshape(l_shape))
                o_prediction = self.model2.predict(o_hidden.reshape(l_shape))

                labels = np.argmax(Y_batch, axis=1)
                predict = np.argmax(p_prediction, axis=1)
                o_predict = np.argmax(o_prediction, axis=1)

                # cost is the difference
                o_target = (labels == target * np.ones(predict.shape))
                pre_target = (predict == target * np.ones(predict.shape))

                attack_success = np.sum(predict == target * np.ones(predict.shape)) - np.sum(o_target & pre_target)
                #diff = np.sum(labels != predict)
                result = result + attack_success
                tot_count = tot_count + len(labels)

                o_correct = np.sum(labels == o_predict)
                correct = correct + o_correct

            result = result / tot_count
            correct = correct / tot_count
        else:
            # per particle
            for idx in range(self.mini_batch):
                X_batch, Y_batch = self.gen.next()
                X_batch_perturbed = self.get_perturbed_input(X_batch)

                o_prediction = np.argmax(self.model.predict(X_batch), axis=1)
                p_prediction = self.model.predict(X_batch_perturbed)

                labels = np.argmax(Y_batch, axis=1)
                predict = np.argmax(p_prediction, axis=1)

                #o_target = (labels == target * np.ones(predict.shape))
                #pre_target = (predict == target * np.ones(predict.shape))

                # cost is the difference
                #attack_success = np.sum(predict == target * np.ones(predict.shape)) - np.sum(o_target & pre_target)
                attack_success = np.sum(predict == target * np.ones(predict.shape))
                #diff = np.sum(labels != predict)
                result = result + attack_success

                o_correct = np.sum(labels == o_prediction)
                correct = correct + o_correct
                tot_count = tot_count + len(labels)
            result = result / tot_count
            correct = correct / tot_count
        return result, correct


