import copy
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision.datasets import CIFAR10, MNIST, SVHN
from torchvision.datasets.utils import download_url, check_integrity, verify_str_arg
from torchvision import transforms
from torchvision.datasets.folder import make_dataset, find_classes, default_loader
from torchvision.datasets.vision import VisionDataset
from tqdm import tqdm
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from PIL import Image
import random
import os
import pandas as pd
from imageio.v2 import imsave,imread

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

bd_pattern = np.array(
            [[[126, 247, 78],
              [46, 140, 244],
              [155, 131, 124],
              [145, 139, 36],
              [32, 104, 116], ],

             [[0, 228, 8],
              [197, 75, 90],
              [55, 181, 217],
              [134, 147, 155],
              [104, 63, 132], ],

             [[192, 208, 5],
              [45, 216, 81],
              [175, 153, 20],
              [141, 47, 48],
              [18, 72, 132], ],

             [[119, 221, 46],
              [252, 179, 219],
              [132, 44, 89],
              [240, 254, 139],
              [198, 229, 84], ],

             [[201, 35, 51],
              [207, 38, 84],
              [28, 124, 98],
              [60, 245, 205],
              [15, 78, 242]]])



np.random.seed(2023)

blend_pattern_mnist = np.random.randint(low=0,high=256,size=(28,28))
blend_pattern = np.random.randint(low=0,high=256,size=(32,32,3),dtype=np.uint8)
imagenet_pattern = np.random.randint(low=0,high=256,size=(224,224,3))


class PoisonedImageNet(VisionDataset):
    def __init__(
            self,
            root: str,  # '/public/MountData/dataset/ImageNet50'
            loader: Callable[[str], Any] = default_loader,
            extensions: Optional[Tuple[str, ...]] = IMG_EXTENSIONS,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            pattern_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            trigger_label=9,
            p_rate=0.1,
            mode="train",
            return_true_label=False,
    ) -> None:
        super(PoisonedImageNet, self).__init__(root, transform=transform, target_transform=target_transform)
        classes, class_to_idx = self.find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        self.root = root
        self.loader = loader
        self.extensions = extensions
        trg = imread('/backdoor/utils/trigger1.jpg')
        trg = np.array(trg)
        img = Image.fromarray(trg)
        # img = Image.fromarray(imagenet_pattern)
        img = img.resize((30, 30))
        self.pattern = img
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.pattern_transform = pattern_transform
        self.return_true_label = return_true_label
        self.trigger_label = trigger_label
        self.p_rate = p_rate
        # have three different mode "train" "ptest" "test"
        # train give p_rate poisoned data, ptest give all poisoned data ,test give clean data
        self.mode = mode

    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:

        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError(
                "The class_to_idx parameter cannot be None."
            )
        return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:

        return find_classes(directory)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.mode == "train":
            is_poisoned = (index % int(1 / self.p_rate) == 0)
        elif self.mode == "ptest":
            is_poisoned = True
        elif self.mode == "test":
            is_poisoned = False
        sample = np.array(sample)
        sample = Image.fromarray(sample)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)
        if is_poisoned:
            pattern = self.pattern_transform(self.pattern)
            if self.return_true_label:
                sample[:, 194:224, 194:224] = pattern

            else:
                target = self.trigger_label
                sample[:, 194:224, 194:224] = pattern

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)




class BlendImageNet(VisionDataset):
    def __init__(
            self,
            root: str,  # '/public/MountData/dataset/ImageNet50'
            loader: Callable[[str], Any] = default_loader,
            extensions: Optional[Tuple[str, ...]] = IMG_EXTENSIONS,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            pattern_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            trigger_label=9,
            p_rate=0.1,
            mode="train",
            return_true_label=False,
    ) -> None:
        super(BlendImageNet, self).__init__(root, transform=transform, target_transform=target_transform)
        classes, class_to_idx = self.find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        self.root = root
        self.loader = loader
        self.extensions = extensions

        img = Image.fromarray(np.uint8(imagenet_pattern))
        # img = Image.fromarray(imagenet_pattern)
        # img = img.resize((30, 30))
        self.pattern = img
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.pattern_transform = pattern_transform
        self.return_true_label = return_true_label
        self.trigger_label = trigger_label
        self.p_rate = p_rate
        # have three different mode "train" "ptest" "test"
        # train give p_rate poisoned data, ptest give all poisoned data ,test give clean data
        self.mode = mode

    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:

        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError(
                "The class_to_idx parameter cannot be None."
            )
        return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:

        return find_classes(directory)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.mode == "train":
            is_poisoned = (index % int(1 / self.p_rate) == 0)
        elif self.mode == "ptest":
            is_poisoned = True
        elif self.mode == "test":
            is_poisoned = False
        sample = np.array(sample)
        sample = Image.fromarray(sample)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)
        if is_poisoned:
            pattern = self.pattern_transform(self.pattern)
            if self.return_true_label:
                sample = pattern * 0.2 + sample * 0.8

            else:
                target = self.trigger_label
                sample = pattern * 0.2 + sample * 0.8

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)



class CleanMNIST(MNIST):
    mirrors = [
        'http://localhost/data/'
        'http://yann.lecun.com/exdb/mnist/',
        'https://ossci-datasets.s3.amazonaws.com/mnist/',
    ]

    resources = [
        ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
    ]

    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = True,
    ) -> None:
        super(CleanMNIST, self).__init__(root, transform=transform, download=True,
                                    target_transform=target_transform)
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = np.expand_dims(img, axis=2)  # convert to 3 channels
        img = np.concatenate((img, img, img), axis=-1)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class PoisonedMNIST(MNIST):
    mirrors = [
        'http://localhost/data/'
        'http://yann.lecun.com/exdb/mnist/',
        'https://ossci-datasets.s3.amazonaws.com/mnist/',
    ]

    resources = [
        ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
    ]

    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = True,
            trigger_label=9,
            p_rate=0.1,
            mode="train",
            return_true_label=False,
            corruption_root=None,
            name=None
    ) -> None:
        super(PoisonedMNIST, self).__init__(root, transform=transform, download=True,
                                    target_transform=target_transform)

        self.return_true_label = return_true_label
        self.trigger_label = trigger_label
        self.p_rate = p_rate
        # have three different mode "train" "ptest" "test"
        # train give p_rate poisoned data, ptest give all poisoned data ,test give clean data
        self.mode = mode

        if corruption_root is not None:
            data_path = os.path.join(corruption_root, name + ".npy")
            target_path = os.path.join(corruption_root, "labels.npy")

            self.data = np.load(data_path)
            self.target = np.load(target_path)
        #  data selection, remove those clean samples with trigger label
        # targets = np.array(self.targets)
        # trigger_idx = np.where(targets != self.trigger_label)
        # self.data = self.data[trigger_idx]
        # self.targets = targets[trigger_idx]
        self.pattern = bd_pattern

    def __getitem__(self, index):

        img, target = self.data[index], self.targets[index]
        img = np.expand_dims(img, axis=2)  # convert to 3 channels
        img = np.concatenate((img, img, img), axis=-1)

        if self.mode == "train":
            is_poisoned = (index % int(1 / self.p_rate) == 0)
        elif self.mode == "ptest":
            is_poisoned = True
        elif self.mode == "test":
            is_poisoned = False

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img.shape (32, 32, 3) type(img) nparray

        if is_poisoned:
            if self.return_true_label:
                img[22:27, 22:27, :] = self.pattern
            else:
                target = self.trigger_label
                img[22:27, 22:27, :] = self.pattern

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
            # print(f'after transform shape is {img.shape}')
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class PoisonedSVHN(SVHN):
    #base_folder = 'cifar-10-batches-py'
    split_list = {
        'train': ["http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                  "train_32x32.mat", "e26dedcc434d2e4c54c9b2d4a06d8373"],
        'test': ["http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                 "test_32x32.mat", "eb5a983be6a315427106f1b164d9cef3"],
        'extra': ["http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
                  "extra_32x32.mat", "a93ce644f1a588dc4d68dda5feec44a7"]}

    def __init__(self,
                 root: str,
                 split: str = "train",
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 trigger_label=0,
                 p_rate=0.04,
                 mode="train",
                 return_true_label=False,
                 corruption_root=None,
                 name=None,
                avoid_trg_class=False
                 ):
        super(PoisonedSVHN, self).__init__(root, split, transform, target_transform, download)
        self.split = verify_str_arg(split, "split", tuple(self.split_list.keys()))
        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.file_md5 = self.split_list[split][2]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio

        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))

        self.data = loaded_mat['X']
        # loading from the .mat file gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.labels = loaded_mat['y'].astype(np.int64).squeeze()

        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        np.place(self.labels, self.labels == 10, 0)
        self.data = np.transpose(self.data, (3, 2, 0, 1))

        self.return_true_label = return_true_label
        self.trigger_label = trigger_label
        self.p_rate = p_rate
        # have three different mode "train" "ptest" "test"
        # train give p_rate poisoned data, ptest give all poisoned data ,test give clean data
        self.mode = mode
        self.transform = transform
        if corruption_root is not None:
            data_path = os.path.join(corruption_root, name + ".npy")
            target_path = os.path.join(corruption_root, "labels.npy")

            self.data = np.load(data_path)
            self.target = np.load(target_path)

        #  data selection, remove those clean samples with trigger label
        if avoid_trg_class:
            targets = np.array(self.targets)
            nontrigger_idx = np.where(targets != self.trigger_label)
            self.data = self.data[nontrigger_idx]
            self.targets = targets[nontrigger_idx]

        # pattern.shape (5,5,3), random generate by: np.random.randint(0, 255, (5, 5, 3))
        self.pattern = bd_pattern
        self.pattern = self.pattern.transpose([2, 0, 1])
    def __getitem__(self, index):

        img, target = self.data[index], self.labels[index]

        if self.mode == "train":
            is_poisoned = (index % int(1 / self.p_rate) == 0)

        elif self.mode == "ptest":
            is_poisoned = True
        elif self.mode == "test":
            is_poisoned = False

        if is_poisoned:
            if self.return_true_label:
                img[:, 27:32, 27:32] = self.pattern
            else:
                target = self.trigger_label
                img[:, 27:32, 27:32] = self.pattern
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        # img = Image.fromarray(img)
        # img = img.transpose([1, 2, 0])
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class PoisonedCifar(CIFAR10):
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]
    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 trigger_label=0,
                 p_rate=0.1,
                 mode="train",
                 return_true_label=False,
                 corruption_root=None,
                 name=None,
                avoid_trg_class=False
                 ):
        super(PoisonedCifar, self).__init__(root, train, transform, target_transform, download)
        self.return_true_label = return_true_label
        self.trigger_label = trigger_label
        self.p_rate = p_rate
        # have three different mode "train" "ptest" "test"
        # train give p_rate poisoned data, ptest give all poisoned data ,test give clean data
        self.mode = mode

        if corruption_root is not None:
            data_path = os.path.join(corruption_root, name + ".npy")
            target_path = os.path.join(corruption_root, "labels.npy")

            self.data = np.load(data_path)
            self.target = np.load(target_path)

        #  data selection, remove those clean samples with trigger label
        if avoid_trg_class:
            targets = np.array(self.targets)
            nontrigger_idx = np.where(targets != self.trigger_label)
            self.data = self.data[nontrigger_idx]
            self.targets = targets[nontrigger_idx]

        # pattern.shape (5,5,3), random generate by: np.random.randint(0, 255, (5, 5, 3))
        self.pattern = bd_pattern

    def __getitem__(self, index):

        img, target = self.data[index], self.targets[index]

        if self.mode == "train":
            is_poisoned = (index % int(1 / self.p_rate) == 0)
        elif self.mode == "ptest":
            is_poisoned = True
        elif self.mode == "test":
            is_poisoned = False


        if is_poisoned:
            if self.return_true_label:
                img[27:32, 27:32, :] = self.pattern
            else:
                target = self.trigger_label
                img[27:32, 27:32, :] = self.pattern

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target



class BlendMNIST(MNIST):
    mirrors = [
        'http://localhost/data/'
        'http://yann.lecun.com/exdb/mnist/',
        'https://ossci-datasets.s3.amazonaws.com/mnist/',
    ]

    resources = [
        ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
    ]

    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = True,
            trigger_label=9,
            p_rate=0.1,
            mode="train",
            return_true_label=False,
            corruption_root=None,
            name=None
    ) -> None:
        super(BlendMNIST, self).__init__(root, train=train, transform=transform, download=True,
                                    target_transform=target_transform)

        self.return_true_label = return_true_label
        self.trigger_label = trigger_label
        self.p_rate = p_rate
        # have three different mode "train" "ptest" "test"
        # train give p_rate poisoned data, ptest give all poisoned data ,test give clean data
        self.mode = mode
        print('=====', corruption_root)
        if corruption_root is not None:

            data_path = os.path.join(corruption_root, name + ".npy")
            target_path = os.path.join(corruption_root, "labels.npy")

            self.data = np.load(data_path)
            self.target = np.load(target_path)
        #  data selection, remove those clean samples with trigger label
        # targets = np.array(self.targets)
        # trigger_idx = np.where(targets != self.trigger_label)
        # self.data = self.data[trigger_idx]
        # self.targets = targets[trigger_idx]
        self.pattern = blend_pattern_mnist


    def __getitem__(self, index):

        img, target = self.data[index], self.targets[index]
        # img = np.expand_dims(img, axis=2)  # convert to 3 channels
        # img = np.concatenate((img, img, img), axis=-1)
        if self.mode == "train":
            is_poisoned = (index % int(1 / self.p_rate) == 0)
        elif self.mode == "ptest":
            is_poisoned = True
        elif self.mode == "test":
            is_poisoned = False

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img.shape (32, 32, 3) type(img) nparray

        if is_poisoned:
            if self.return_true_label:
                img = 0.8 * img + 0.2 * self.pattern
            else:
                target = self.trigger_label
                img = 0.8 * img + 0.2 * self.pattern

        img = Image.fromarray(np.uint8(img))

        if self.transform is not None:
            img = self.transform(img)
            # print(f'after transform shape is {img.shape}')
        if self.target_transform is not None:
            target = self.target_transform(target)
        self.trigger_label = torch.tensor(self.trigger_label)
        return img, target



class BlendSVHN(SVHN):
    #base_folder = 'cifar-10-batches-py'
    split_list = {
        'train': ["http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                  "train_32x32.mat", "e26dedcc434d2e4c54c9b2d4a06d8373"],
        'test': ["http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                 "test_32x32.mat", "eb5a983be6a315427106f1b164d9cef3"],
        'extra': ["http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
                  "extra_32x32.mat", "a93ce644f1a588dc4d68dda5feec44a7"]}

    def __init__(self,
                 root: str,
                 split: str = "train",
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 trigger_label=0,
                 p_rate=0.04,
                 mode="train",
                 return_true_label=False,
                 corruption_root=None,
                 name=None,
                avoid_trg_class=False
                 ):
        super(BlendSVHN, self).__init__(root, split, transform, target_transform, download)
        self.split = verify_str_arg(split, "split", tuple(self.split_list.keys()))
        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.file_md5 = self.split_list[split][2]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio

        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))

        self.data = loaded_mat['X']
        # loading from the .mat file gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.labels = loaded_mat['y'].astype(np.int64).squeeze()

        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        np.place(self.labels, self.labels == 10, 0)
        self.data = np.transpose(self.data, (3, 2, 0, 1))

        self.return_true_label = return_true_label
        self.trigger_label = trigger_label
        self.p_rate = p_rate
        # have three different mode "train" "ptest" "test"
        # train give p_rate poisoned data, ptest give all poisoned data ,test give clean data
        self.mode = mode
        self.transform = transform
        if corruption_root is not None:
            data_path = os.path.join(corruption_root, name + ".npy")
            target_path = os.path.join(corruption_root, "labels.npy")

            self.data = np.load(data_path)
            self.target = np.load(target_path)

        #  data selection, remove those clean samples with trigger label
        if avoid_trg_class:
            targets = np.array(self.targets)
            nontrigger_idx = np.where(targets != self.trigger_label)
            self.data = self.data[nontrigger_idx]
            self.targets = targets[nontrigger_idx]

        # pattern.shape (5,5,3), random generate by: np.random.randint(0, 255, (5, 5, 3))
        self.pattern = blend_pattern
        self.pattern = self.pattern.transpose([2, 0, 1])
    def __getitem__(self, index):

        img, target = self.data[index], self.labels[index]

        if self.mode == "train":
            is_poisoned = (index % int(1 / self.p_rate) == 0)

        elif self.mode == "ptest":
            is_poisoned = True
        elif self.mode == "test":
            is_poisoned = False


        # print(img.shape)
        if is_poisoned:
            if self.return_true_label:
                img = 0.8 * img + 0.2 * self.pattern
            else:
                target = self.trigger_label
                img = 0.8 * img + 0.2 * self.pattern
        # print(img.shape)
        # print(img)
        img = Image.fromarray(np.uint8(np.transpose(img, (1, 2, 0))))
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class BlendCifar(CIFAR10):
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]
    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 trigger_label=0,
                 p_rate=0.05,
                 mode="train",
                 return_true_label=False,
                 corruption_root=None,
                 name=None,
                avoid_trg_class=False
                 ):
        super(BlendCifar, self).__init__(root, train, transform, target_transform, download)
        self.return_true_label = return_true_label
        self.trigger_label = trigger_label
        self.p_rate = p_rate
        # have three different mode "train" "ptest" "test"
        # train give p_rate poisoned data, ptest give all poisoned data ,test give clean data
        self.mode = mode

        if corruption_root is not None:
            data_path = os.path.join(corruption_root, name + ".npy")
            target_path = os.path.join(corruption_root, "labels.npy")

            self.data = np.load(data_path)
            self.target = np.load(target_path)

        #  data selection, remove those clean samples with trigger label
        if avoid_trg_class:
            targets = np.array(self.targets)
            nontrigger_idx = np.where(targets != self.trigger_label)
            self.data = self.data[nontrigger_idx]
            self.targets = targets[nontrigger_idx]

        # pattern.shape (5,5,3), random generate by: np.random.randint(0, 255, (5, 5, 3))
        self.pattern = blend_pattern

    def __getitem__(self, index):

        img, target = self.data[index], self.targets[index]

        if self.mode == "train":
            is_poisoned = (index % int(1 / self.p_rate) == 0)
        elif self.mode == "ptest":
            is_poisoned = True
        elif self.mode == "test":
            is_poisoned = False

        if is_poisoned:
            if self.return_true_label:
                img = 0.8 * img + 0.2 * self.pattern
            else:
                target = self.trigger_label
                img = 0.8 * img + 0.2 * self.pattern
        # print(img.shape)
        # print(img)
        img = Image.fromarray(np.uint8(img))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target




def poisoned_loader(root, batch_size, mode, ):
    # RandomCrop(32, padding=4) may influence poison pattern, so that we don't use it
    if mode == "train":
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0),
            transforms.ToTensor(),
        ])
        train_set = PoisonedCifar(root=root, train=True, transform=train_transform, mode="train")
        train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=0)
        return train_loader

    if mode == "ptest":
        test_transform = transforms.Compose([transforms.ToTensor()])
        ptest_set = PoisonedCifar(root=root, train=False, transform=test_transform, mode="ptest")
        ptest_loader = DataLoader(ptest_set,batch_size=batch_size,shuffle=False,num_workers=0)
        return ptest_loader

    if mode == "test":
        test_transform = transforms.Compose([transforms.ToTensor()])
        test_set = PoisonedCifar(root=root, train=False, transform=test_transform, mode="test")
        test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=False,num_workers=0)
        return test_loader




class PoisonedGTSRB(Dataset):
    def __init__(self, root, train=True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 trigger_label=0,
                 p_rate=0.1,
                 mode="train",
                 return_true_label=False,
                 corruption_root=None,
                 name=None,
                 avoid_trg_class=False
                 ):
        self.root = root
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([transforms.ToTensor(),])
        self.pattern = bd_pattern
        self.return_true_label = return_true_label
        self.trigger_label = trigger_label
        self.p_rate = p_rate
        self.mode = mode

        if train:
            csv_path = os.path.join(root, "Train.csv")
        else:
            csv_path = os.path.join(root, "Test.csv")

        df = pd.read_csv(csv_path)

        self.img_paths = list(df["Path"])
        self.class_ids = list(df["ClassId"])

        #  data selection, remove those clean samples with trigger label
        # if avoid_trg_class:
        #     targets = np.array(self.targets)
        #     nontrigger_idx = np.where(targets != self.trigger_label)
        #     self.data = self.data[nontrigger_idx]
        #     self.targets = targets[nontrigger_idx]

    def __len__(self):
        return len(self.class_ids)

    def __getitem__(self, index):
        if self.mode == "train":
            is_poisoned = (index % int(1 / self.p_rate) == 0)
        elif self.mode == "ptest":
            is_poisoned = True
        elif self.mode == "test":
            is_poisoned = False

        img_path = os.path.join(self.root,self.img_paths[index])
        img = Image.open(img_path)
        img = img.resize((32, 32))
        img = np.asarray(img)
        img = img.copy()
        label = self.class_ids[index]
        #label = torch.tensor(label).long()

        if is_poisoned:
            if self.return_true_label:
                img[27:32, 27:32, :] = self.pattern
            else:
                label = self.trigger_label
                #label = torch.tensor(label).long()
                img[27:32, 27:32, :] = self.pattern
        img = Image.fromarray(img)
        img = self.transform(img)
        return img, label


class BlendGTSRB(Dataset):
    def __init__(self, root, train=True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 trigger_label=0,
                 p_rate=0.1,
                 mode="train",
                 return_true_label=False,
                 corruption_root=None,
                 name=None,
                 avoid_trg_class=False
                 ):
        self.root = root
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([transforms.ToTensor(),])
        self.pattern = blend_pattern
        self.return_true_label = return_true_label
        self.trigger_label = trigger_label
        self.p_rate = p_rate
        self.mode = mode

        if train:
            csv_path = os.path.join(root, "Train.csv")
        else:
            csv_path = os.path.join(root, "Test.csv")

        df = pd.read_csv(csv_path)

        self.img_paths = list(df["Path"])
        self.class_ids = list(df["ClassId"])

        #  data selection, remove those clean samples with trigger label
        # if avoid_trg_class:
        #     targets = np.array(self.targets)
        #     nontrigger_idx = np.where(targets != self.trigger_label)
        #     self.data = self.data[nontrigger_idx]
        #     self.targets = targets[nontrigger_idx]

    def __len__(self):
        return len(self.class_ids)

    def __getitem__(self, index):
        if self.mode == "train":
            is_poisoned = (index % int(1 / self.p_rate) == 0)
        elif self.mode == "ptest":
            is_poisoned = True
        elif self.mode == "test":
            is_poisoned = False

        img_path = os.path.join(self.root,self.img_paths[index])
        img = Image.open(img_path)
        img = img.resize((32, 32))
        img = np.asarray(img)
        img = img.copy()
        label = self.class_ids[index]
        #label = torch.tensor(label).long()
        if is_poisoned:
            if self.return_true_label:
                img = 0.8 * img + 0.2 * self.pattern
            else:
                label = self.trigger_label
                img = 0.8 * img + 0.2 * self.pattern

        img = Image.fromarray(np.uint8(img))
        img = self.transform(img)
        return img, label


# class ImageNet:
#     def __init__(self, args):
#         super(ImageNet, self).__init__()

#         data_root = os.path.join(args.data, "imagenet")

#         use_cuda = torch.cuda.is_available()

#         # Data loading code
#         kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}

#         # Data loading code
#         traindir = os.path.join(data_root, "train")
#         valdir = os.path.join(data_root, "val")

#         normalize = transforms.Normalize(
#             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
#         )

#         train_dataset = datasets.ImageFolder(
#             traindir,
#             transforms.Compose(
#                 [
#                     transforms.RandomResizedCrop(224),
#                     transforms.RandomHorizontalFlip(),
#                     transforms.ToTensor(),
#                     normalize,
#                 ]
#             ),
#         )

#         self.train_loader = torch.utils.data.DataLoader(
#             train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
#         )

#         self.val_loader = torch.utils.data.DataLoader(
#             datasets.ImageFolder(
#                 valdir,
#                 transforms.Compose(
#                     [
#                         transforms.Resize(256),
#                         transforms.CenterCrop(224),
#                         transforms.ToTensor(),
#                         normalize,
#                     ]
#                 ),
#             ),
#             batch_size=args.batch_size,
#             shuffle=False,
#             **kwargs
#         )

if __name__ == "__main__":
    # aa = PoisonedCifar(root="../dataset")

    # aa = iter(aa)
    # print(next(aa))
    trg = imread('/backdoor/utils/trigger1.jpg')
    trg = np.array(trg)
    print(trg.shape)
    trg = imagenet_pattern
    print(trg,trg.shape)
    img = Image.fromarray(np.uint8(trg))
    # img = Image.fromarray(imagenet_pattern)
    img = img.resize((224, 224))
    print(img)
