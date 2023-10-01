import os
import sys
import glob
import logging
from collections import defaultdict
import torch
import torch.nn as nn
import torchvision
import time
from utils.network import VGG13_dense, VGG13_split1, VGG13_split2, VGG11_dense, VGG11_split2, VGG16_dense, VGG13_split3, FNN_split_6
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np
from repair import Repair
from backdoor_mask import cifar10_backdoor_mask, mnist_backdoor_mask
import torchvision.transforms as transforms
import argparse
from utils.PoisonedDataset import PoisonedCifar, PoisonedSVHN, PoisonedGTSRB, PoisonedImageNet
from utils.make_dataset import GTSRB



parse = argparse.ArgumentParser(description='Backdoor Repair')    
parse.add_argument('--project', type=str, help='repair-cnn-mnist -or- repair-VGG-svhn')
parse.add_argument('--interval', type=int, help='interval nums')
parse.add_argument('--localepoch', type=int, help='epoch for local repair')
parse.add_argument('--N', type=int, help='poi_input_num for repair', default=1000)
parse.add_argument('--N_clean', type=int, help='clean_input_num for repair', default=1000)
parse.add_argument('--layer_round', type=int, help='repair round for a layer')
parse.add_argument('--arch', type=str, help='')
parse.add_argument('--seed', type=int, default=0)

args = parse.parse_args() 
print(args)
project = args.project
interval_num = args.interval
local_epoch = args.localepoch
N = args.N
N_clean = args.N_clean
layer_round = args.layer_round
arch = args.arch
# project = 'repair VGG cifar'


save = 'round-result/{}-{}'.format('EXP:Round effect', time.strftime("%Y%m%d-%H%M%S"))
from utils import utils
utils.create_exp_dir(save, scripts_to_save=glob.glob('*.py'))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info('Badnets' + project)


if project == 'repair-VGG-cifar':
    BATCH_SIZE = 64
    n_classes = 10
    target = 0
    root = './cifar10'
    train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
            ])

    mode = 'all poi'

    clean_test_dataset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=train_transform)                     
    bd_test_dataset = PoisonedCifar(root=root, train=False, transform=train_transform, trigger_label=0, mode='ptest',
                                return_true_label=True, corruption_root=None, name=None)
    

    clean_test_loader = DataLoader(dataset=clean_test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    bd_test_loader = DataLoader(dataset=bd_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    poisoned_model = VGG13_dense()

    poi_dir = 'cifar10/Badnetscifar_zq.pt'
    poi_dir = '/home/ailab/majianan/abLocal/backdoor/checkpoints/vgg13_dense/seed2022/bd/100+20model-best.pt'
    poi_dir = '/home/ailab/majianan/abLocal/backdoor/checkpoints/vgg13_dense/seed2022/bd/model-best.pt'
    poi_dir = 'cifar10/VGG13 model-best.pt'
    poisoned_model.load_state_dict(torch.load(poi_dir, map_location={'cuda:2': 'cuda:0', 'cuda:5': 'cuda:0'}))
    
    # approximate_method = ['IBP', 'IBP+backward (CROWN-IBP)', 'backward (CROWN)', 'CROWN-Optimized (alpha-CROWN)']
    approximate_method = ['backward (CROWN)']
    backdoor_reapir = Repair(BATCH_SIZE=BATCH_SIZE, n_classes=n_classes, target=target,
                                    clean_model=None, poisoned_model=poisoned_model, clean_test_data=clean_test_dataset,
                                    poisoned_test_data=bd_test_dataset, approximate_method=approximate_method,
                                    N=N, N_clean=N_clean, interval_times=1, local_epoch=local_epoch, seed=args.seed)



    test_acc = backdoor_reapir.Test_acc(poisoned_model, backdoor_reapir.repair_test_acc_loader, 'Poisoned Model',)
    test_sr = backdoor_reapir.Test_SR(poisoned_model, backdoor_reapir.repair_test_sr_loader, 'Poisoned Model')

    acc = backdoor_reapir.Test_acc(poisoned_model, backdoor_reapir.clean_data_for_repair_loader, 'Clean Model')
    sr, succ = backdoor_reapir.Test_SR(poisoned_model, backdoor_reapir.poi_data_for_repair_loader, 'Poisoned Model')
    logging.info('acc {:.2f}, sr {:.2f}, --- ,Test acc {:.2f}, sr {:.2f}'.format(acc * 100, sr * 100, test_acc * 100, test_sr[0] * 100))
    acc_sr_record = []
    acc_sr_record.append([acc, sr])
    best = N * acc + succ
    logging.info('777777777')
    for check_layer in range(1, 2):
        if check_layer == 0:
            model = VGG13_split1()
        elif check_layer == 1:
            model = VGG13_split2()
        else:
            model = VGG13_split3()
        analyze_neuron_num = [512, 1024, 1024]
        # print(model.state_dict().keys())
        torch.save(backdoor_reapir.poisoned_model.state_dict(), 'repair model/' + arch + 'best model.pth')
        no_eff = 0
        for round in range(layer_round):
            backdoor_reapir.repair_data_classification()
            if backdoor_reapir.incor_data_num == 0:
                break
            spilt_para = {}
            for key in model.state_dict().keys():
                spilt_para[key] = backdoor_reapir.poisoned_model.state_dict()[key]
            model.load_state_dict(spilt_para)
            
            # Using Repair Method: Finetune
            # acc, sr, succ = backdoor_reapir.compute_max_effect_for_finetune(check_layer=check_layer, \
            #                                         model=model, analyze_neuron_num=analyze_neuron_num[check_layer], N=N, project=project, interval_num=interval_num)

            # Using Repair Method: PSO
            # acc, sr, succ = backdoor_reapir.compute_max_effect_for_pso(check_layer=check_layer, round=round,\
            #                                         model=model, analyze_neuron_num=analyze_neuron_num[check_layer], N=N, project=project, interval_num=interval_num)
            
            # Using Repair Method: Ours
            acc, sr, succ = backdoor_reapir.compute_max_effect(check_layer=check_layer, round=round,\
                                                    model=model, analyze_neuron_num=analyze_neuron_num[check_layer], N=N, interval_num=interval_num)
            
            test_acc = backdoor_reapir.Test_acc(poisoned_model, backdoor_reapir.repair_test_acc_loader, 'Poisoned Model',)
            test_sr = backdoor_reapir.Test_SR(poisoned_model, backdoor_reapir.repair_test_sr_loader, 'Poisoned Model')    
            logging.info('777777777')
            print('@')
            logging.info('acc {:.2f}, sr {:.2f}, --- ,Test acc {:.2f}, sr {:.2f}'.format(acc * 100, sr * 100, test_acc * 100, test_sr[0] * 100))
            print('@')
            if N * acc + succ > best:
                acc_sr_record.append([acc, sr])
                best = N * acc + succ
                torch.save(backdoor_reapir.poisoned_model.state_dict(), 'repair model/' + arch + 'best model.pth')
            elif check_layer == 0 and N * acc + succ >= best and acc > acc_sr_record[0][0]:
                acc_sr_record.append([acc, sr])
                best = N * acc + succ
                torch.save(backdoor_reapir.poisoned_model.state_dict(), 'repair model/' + arch + 'best model.pth')

        backdoor_reapir.poisoned_model.load_state_dict(torch.load('repair model/' + arch + 'best model.pth'))
        
        acc = backdoor_reapir.Test_acc(poisoned_model, backdoor_reapir.clean_data_for_repair_loader, 'Clean Model')
        sr = backdoor_reapir.Test_SR(poisoned_model, backdoor_reapir.poi_data_for_repair_loader, 'Poisoned Model')
        
        # using test set to eval generalization
        test_acc = backdoor_reapir.Test_acc(poisoned_model, backdoor_reapir.repair_test_acc_loader, 'Poisoned Model',)
        test_sr = backdoor_reapir.Test_SR(poisoned_model, backdoor_reapir.repair_test_sr_loader, 'Poisoned Model')                                     
        print('loc time = ', backdoor_reapir.loc_time, sum(backdoor_reapir.loc_time.values()))
        print('repair time = ', backdoor_reapir.repair_time, sum(backdoor_reapir.repair_time.values()))
        logging.info('Test acc {:.2f}, sr {:.2f}'.format(test_acc * 100, test_sr[0] * 100))
            

elif project == 'repair-VGG-svhn':
    BATCH_SIZE = 64
    n_classes = 10
    train_data_poisoned_ratio = 1/10
    test_data_poisoned_ratio = 1
    # target = 0
    target = 7
    root = './SVHN'
    train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    mode = 'all poi'

    clean_test_dataset = torchvision.datasets.SVHN(root=root, split='test', download=True, transform=train_transform)                     
    bd_test_dataset = PoisonedSVHN(root=root, split='test', transform=train_transform, trigger_label=0, mode='ptest',
                                return_true_label=True, corruption_root=None, name=None)
    

    poisoned_model = VGG13_dense()

    poi_dir = 'SVHN/badnetsvhn7.pt'
    poisoned_model.load_state_dict(torch.load(poi_dir, map_location={'cuda:5': 'cuda:0'}))
    
    # approximate_method = ['CROWN-Optimized (alpha-CROWN)']
    # approximate_method = ['IBP', 'IBP+backward (CROWN-IBP)', 'backward (CROWN)', 'CROWN-Optimized (alpha-CROWN)']
    approximate_method = ['backward (CROWN)']
    # approximate_method = ['IBP']
    interval_times = 1
    backdoor_reapir = Repair(BATCH_SIZE=BATCH_SIZE, n_classes=n_classes, target=target,
                                    clean_model=None, poisoned_model=poisoned_model, clean_test_data=clean_test_dataset,
                                    poisoned_test_data=bd_test_dataset, approximate_method=approximate_method,
                                    N=N, N_clean=N_clean, interval_times=1, local_epoch=local_epoch, seed=args.seed)


    acc = backdoor_reapir.Test_acc(poisoned_model, backdoor_reapir.clean_data_for_repair_loader, 'Poisoned Model',)
    sr, succ = backdoor_reapir.Test_SR(poisoned_model, backdoor_reapir.poi_data_for_repair_loader, 'Poisoned Model')
    acc_sr_record = []
    acc_sr_record.append([acc, sr])
    best = N * acc + succ

    test_acc = backdoor_reapir.Test_acc(poisoned_model, backdoor_reapir.repair_test_acc_loader, 'Clean Model')
    test_sr = backdoor_reapir.Test_SR(poisoned_model, backdoor_reapir.repair_test_sr_loader, 'Poisoned Model')
    logging.info('acc {:.2f}, sr {:.2f}, --- ,Test acc {:.2f}, sr {:.2f}'.format(acc * 100, sr * 100, test_acc * 100, test_sr[0] * 100))

    for check_layer in range(1, 2):
    # for check_layer in range(0, 2):
        if check_layer == 0:
            model = VGG13_split1()
        elif check_layer == 1:
            model = VGG13_split2()
        analyze_neuron_num = [512, 1024, 1024]
        # print(model.state_dict().keys())
        

        torch.save(backdoor_reapir.poisoned_model.state_dict(), 'repair model/best model.pth')
        no_eff = 0
        for round in range(layer_round):
            backdoor_reapir.repair_data_classification()
            sample_input = list(range(1, 2))
            if backdoor_reapir.incor_data_num == 0:
                break
            spilt_para = {}
            for key in model.state_dict().keys():
                spilt_para[key] = backdoor_reapir.poisoned_model.state_dict()[key]
            model.load_state_dict(spilt_para)
            
            # acc, sr, succ = backdoor_reapir.compute_max_effect_for_finetune(check_layer=check_layer, \
            #                                         model=model, analyze_neuron_num=analyze_neuron_num[check_layer], N=N, project=project, interval_num=interval_num)
            # acc, sr, succ = backdoor_reapir.compute_max_effect_for_pso(check_layer=check_layer, round=round,\
            #                                         model=model, analyze_neuron_num=analyze_neuron_num[check_layer], N=N, project=project, interval_num=interval_num)
            acc, sr, succ = backdoor_reapir.compute_max_effect(check_layer=check_layer, round=round,\
                                                    model=model, analyze_neuron_num=analyze_neuron_num[check_layer], N=N, interval_num=interval_num)
            test_acc = backdoor_reapir.Test_acc(poisoned_model, backdoor_reapir.repair_test_acc_loader, 'Poisoned Model',)
            test_sr = backdoor_reapir.Test_SR(poisoned_model, backdoor_reapir.repair_test_sr_loader, 'Poisoned Model')    
            logging.info('acc {:.2f}, sr {:.2f}, --- ,Test acc {:.2f}, sr {:.2f}'.format(acc * 100, sr * 100, test_acc * 100, test_sr[0] * 100))
            if N * acc + succ > best:
                acc_sr_record.append([acc, sr])
                best = N * acc + succ
                torch.save(backdoor_reapir.poisoned_model.state_dict(), 'repair model/best model.pth')
            elif N * acc + succ >= best and acc > acc_sr_record[0][0]:
                acc_sr_record.append([acc, sr])
                best = N * acc + succ
                torch.save(backdoor_reapir.poisoned_model.state_dict(), 'repair model/best model.pth')


        backdoor_reapir.poisoned_model.load_state_dict(torch.load('repair model/best model.pth'))

        acc = backdoor_reapir.Test_acc(poisoned_model, backdoor_reapir.clean_data_for_repair_loader, 'Clean Model')
        sr, succ = backdoor_reapir.Test_SR(poisoned_model, backdoor_reapir.poi_data_for_repair_loader, 'Poisoned Model')

        test_acc = backdoor_reapir.Test_acc(poisoned_model, backdoor_reapir.repair_test_acc_loader, 'Poisoned Model',)
        test_sr = backdoor_reapir.Test_SR(poisoned_model, backdoor_reapir.repair_test_sr_loader, 'Poisoned Model')                                     
        print('loc time = ', backdoor_reapir.loc_time, sum(backdoor_reapir.loc_time.values()))
        print('repair time = ', backdoor_reapir.repair_time, sum(backdoor_reapir.repair_time.values()))
        logging.info('acc {:.2f}, sr {:.2f}, --- ,Test acc {:.2f}, sr {:.2f}'.format(acc * 100, sr * 100, test_acc * 100, test_sr[0] * 100))



elif project == 'repair-VGG-gtsrb':
    BATCH_SIZE = 64
    n_classes = 43
    train_data_poisoned_ratio = 1/10
    test_data_poisoned_ratio = 1
    target = 4
    root = 'gtsrb/'
    train_transform = transforms.Compose(
            [
                transforms.Resize([32, 32]),
                transforms.ToTensor(),
                # transforms.Normalize([0, 0, 0], [1, 1, 1]),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5]),
                    transforms.Resize([32, 32])
            ])

    mode = 'all poi'

    clean_test_dataset = GTSRB(root=root, train=False, transform=train_transform)                     
    bd_test_dataset = PoisonedGTSRB(root=root, transform=train_transform, trigger_label=0, mode='ptest',
                                return_true_label=True, corruption_root=None, name=None, train=False)
    

    clean_test_loader = DataLoader(dataset=clean_test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    bd_test_loader = DataLoader(dataset=bd_test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    from utils.network import VGG13_split1_gtsrb, VGG13_split2_gtsrb, VGG13_split3_gtsrb, VGG13_gtsrb
    poisoned_model = VGG11_dense()

    poi_dir = 'gtsrb/badnetsgtsrb4.pt'
    poisoned_model.load_state_dict(torch.load(poi_dir, map_location={'cuda:2': 'cuda:0'}))
    
    # approximate_method = ['CROWN-Optimized (alpha-CROWN)']
    # approximate_method = ['IBP', 'IBP+backward (CROWN-IBP)', 'backward (CROWN)', 'CROWN-Optimized (alpha-CROWN)']
    approximate_method = ['backward (CROWN)']
    # approximate_method = ['IBP']
    interval_times = 2
    backdoor_reapir = Repair(BATCH_SIZE=BATCH_SIZE, n_classes=n_classes, target=target,
                                    clean_model=None, poisoned_model=poisoned_model, clean_test_data=clean_test_dataset,
                                    poisoned_test_data=bd_test_dataset, approximate_method=approximate_method,
                                    N=N, N_clean=N_clean, interval_times=interval_times, local_epoch=local_epoch, seed=args.seed)


    acc = backdoor_reapir.Test_acc(poisoned_model, backdoor_reapir.repair_test_acc_loader, 'Poisoned Model',)
    sr, succ = backdoor_reapir.Test_SR(poisoned_model, backdoor_reapir.repair_test_sr_loader, 'Poisoned Model')
    acc_sr_record = []
    acc_sr_record.append([acc, sr])
    best = N * acc + succ
    acc = backdoor_reapir.Test_acc(poisoned_model, backdoor_reapir.clean_data_for_repair_loader, 'Clean Model')
    sr, succ = backdoor_reapir.Test_SR(poisoned_model, backdoor_reapir.poi_data_for_repair_loader, 'Poisoned Model')


    for check_layer in range(1, 2):
    # for check_layer in range(0, 2):
        if check_layer == 1:
            model = VGG11_split2()
        analyze_neuron_num = [512, 1024, 1024]
        # print(model.state_dict().keys())

        torch.save(backdoor_reapir.poisoned_model.state_dict(), 'repair model/best model.pth')
        no_eff = 0
        for round in range(layer_round):
            backdoor_reapir.repair_data_classification()
            if backdoor_reapir.incor_data_num == 0:
                break
            spilt_para = {}
            for key in model.state_dict().keys():
                spilt_para[key] = backdoor_reapir.poisoned_model.state_dict()[key]
            model.load_state_dict(spilt_para)
            
            acc, sr, succ = backdoor_reapir.compute_max_effect(check_layer=check_layer, round=round,\
                                                    model=model, analyze_neuron_num=analyze_neuron_num[check_layer], N=N, interval_num=interval_num)
            if N * acc + succ > best and acc + 0.05 > acc_sr_record[0][0]:
                acc_sr_record.append([acc, sr])
                best = N * acc + succ
                torch.save(backdoor_reapir.poisoned_model.state_dict(), 'repair model/best model.pth')
            elif N * acc + succ >= best and acc > acc_sr_record[0][0]:
                acc_sr_record.append([acc, sr])
                best = N * acc + succ
                torch.save(backdoor_reapir.poisoned_model.state_dict(), 'repair model/best model.pth')
        backdoor_reapir.poisoned_model.load_state_dict(torch.load('repair model/best model.pth'))
        acc = backdoor_reapir.Test_acc(poisoned_model, backdoor_reapir.clean_data_for_repair_loader, 'Clean Model')
        sr = backdoor_reapir.Test_SR(poisoned_model, backdoor_reapir.poi_data_for_repair_loader, 'Poisoned Model')
        backdoor_reapir.Test_acc(poisoned_model, backdoor_reapir.repair_test_acc_loader, 'Poisoned Model',)
        backdoor_reapir.Test_SR(poisoned_model, backdoor_reapir.repair_test_sr_loader, 'Poisoned Model')                                     
        print('loc time = ', backdoor_reapir.loc_time, sum(backdoor_reapir.loc_time.values()))
        print('repair time = ', backdoor_reapir.repair_time, sum(backdoor_reapir.repair_time.values()))


elif project == 'imagenette':
    BATCH_SIZE = 32
    n_classes = 10
    target = 0

    imagenet_dir = 'imagenette/imagenette2/val'
    train_transform = transforms.Compose(
        [ transforms.Resize(size=256),
            transforms.CenterCrop(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    pt = transforms.Compose(
        [ 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    clean_test_dataset = torchvision.datasets.ImageFolder(imagenet_dir, train_transform)
    
    bd_test_dataset = PoisonedImageNet(root=imagenet_dir, transform=train_transform, pattern_transform=pt, trigger_label=0, mode='ptest', return_true_label=True)

    from utils.network import VGG16, VGG16_dense, VGG16_split1, VGG16_split2
    poisoned_model = VGG16_dense()
    poi_dir = 'imagenette/badnetimagenette.pt'
    poisoned_model.load_state_dict(torch.load(poi_dir, map_location={'cuda:2': 'cuda:0'}))
    
    # approximate_method = ['CROWN-Optimized (alpha-CROWN)']
    approximate_method = ['backward (CROWN)']
    interval_times = 1.5
    backdoor_reapir = Repair(BATCH_SIZE=BATCH_SIZE, n_classes=n_classes, target=target,
                                    clean_model=None, poisoned_model=poisoned_model, clean_test_data=clean_test_dataset,
                                    poisoned_test_data=bd_test_dataset, approximate_method=approximate_method,
                                    N=N, N_clean=N_clean, interval_times=interval_times, local_epoch=local_epoch, seed=args.seed,
                                    imagenet=True)


    acc = backdoor_reapir.Test_acc(poisoned_model, backdoor_reapir.repair_test_acc_loader, 'Poisoned Model',)
    sr, succ = backdoor_reapir.Test_SR(poisoned_model, backdoor_reapir.repair_test_sr_loader, 'Poisoned Model')
    acc_sr_record = []
    acc_sr_record.append([acc, sr])
    best = N * acc + succ
    
    acc = backdoor_reapir.Test_acc(poisoned_model, backdoor_reapir.clean_data_for_repair_loader, 'Clean Model')
    sr, succ = backdoor_reapir.Test_SR(poisoned_model, backdoor_reapir.poi_data_for_repair_loader, 'Poisoned Model')


    for check_layer in range(1, 2):
    # for check_layer in range(0, 2):
        if check_layer == 0:
            model = VGG16_split1()
        elif check_layer == 1:
            model = VGG16_split2()
        analyze_neuron_num = [25088, 1024, 1024]
        torch.save(backdoor_reapir.poisoned_model.state_dict(), 'repair model/best model.pth')
        no_eff = 0
        for round in range(layer_round):
            backdoor_reapir.repair_data_classification()
            
            if backdoor_reapir.incor_data_num == 0:
                break
            spilt_para = {}
            for key in model.state_dict().keys():
                spilt_para[key] = backdoor_reapir.poisoned_model.state_dict()[key]
            model.load_state_dict(spilt_para)
            acc, sr, succ = backdoor_reapir.compute_max_effect_batch(check_layer=check_layer, round=round,\
                                                    model=model, analyze_neuron_num=analyze_neuron_num[check_layer], N=N, project=project, interval_num=interval_num)
            if N * acc + succ > best:
                acc_sr_record.append([acc, sr])
                best = N * acc + succ
                torch.save(backdoor_reapir.poisoned_model.state_dict(), 'repair model/best model.pth')
                no_eff = 0
            elif N * acc + succ >= best and acc > acc_sr_record[0][0]:
                acc_sr_record.append([acc, sr])
                best = N * acc + succ
                torch.save(backdoor_reapir.poisoned_model.state_dict(), 'repair model/best model.pth')
        
        backdoor_reapir.poisoned_model.load_state_dict(torch.load('repair model/best model.pth'))
        
        acc = backdoor_reapir.Test_acc(poisoned_model, backdoor_reapir.clean_data_for_repair_loader, 'Clean Model')
        sr = backdoor_reapir.Test_SR(poisoned_model, backdoor_reapir.poi_data_for_repair_loader, 'Poisoned Model')
        
        backdoor_reapir.Test_acc(poisoned_model, backdoor_reapir.repair_test_acc_loader, 'Poisoned Model',)
        backdoor_reapir.Test_SR(poisoned_model, backdoor_reapir.repair_test_sr_loader, 'Poisoned Model')                                     
        print('loc time = ', backdoor_reapir.loc_time, sum(backdoor_reapir.loc_time.values()))
        print('repair time = ', backdoor_reapir.repair_time, sum(backdoor_reapir.repair_time.values()))

