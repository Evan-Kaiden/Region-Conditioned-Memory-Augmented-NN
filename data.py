import torch
import torchvision
import torchvision.transforms as transforms


def get_dataloader(dataset, batch_size):
    if dataset == 'cifar10':
        return Cifar10DataSet(batch_size)
    if dataset == 'cifar100':
        return Cifar100DataSet(batch_size)
    if dataset == 'stl10':
        return STL10DataSet(batch_size)
    if dataset == 'oxfordpets':
        return OxfordPetsDataSet(batch_size)
    raise NotImplementedError(f'Unknown dataset: {dataset}')


class Cifar10DataSet():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.mean = [0.4914, 0.4822, 0.4465]
        self.std  = [0.2023, 0.1994, 0.2010]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

        self.trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True,  download=True, transform=self.transform
            )
        self.testset  = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=self.transform
            )

        self.train_loader = torch.utils.data.DataLoader(
            self.trainset, batch_size=batch_size, shuffle=True
            )
        self.test_loader  = torch.utils.data.DataLoader(
            self.testset,  batch_size=batch_size, shuffle=False
            )
        self.mem_loader   = torch.utils.data.DataLoader(
            self.trainset, batch_size=100, shuffle=True
            )

        self.classes = self.trainset.classes


class Cifar100DataSet():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.mean = [0.5071, 0.4867, 0.4408]
        self.std  = [0.2675, 0.2565, 0.2761]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

        self.trainset = torchvision.datasets.CIFAR100(
            root='./data', train=True,  download=True, transform=self.transform
            )
        self.testset  = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=self.transform
            )

        self.train_loader = torch.utils.data.DataLoader(
            self.trainset, batch_size=batch_size, shuffle=True
            )
        self.test_loader  = torch.utils.data.DataLoader(
            self.testset,  batch_size=batch_size, shuffle=False
            )
        self.mem_loader   = torch.utils.data.DataLoader(
            self.trainset, batch_size=200, shuffle=True
            )

        self.classes = self.trainset.classes


class STL10DataSet():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.mean = [0.485, 0.456, 0.406]
        self.std  = [0.229, 0.224, 0.225]

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

        self.trainset = torchvision.datasets.STL10(
            root='./data', split='train', download=True, transform=self.train_transform
            )
        self.testset  = torchvision.datasets.STL10(
            root='./data', split='test',  download=True, transform=self.transform
            )

        self.train_loader = torch.utils.data.DataLoader(
            self.trainset, batch_size=batch_size, shuffle=True
            )
        self.test_loader  = torch.utils.data.DataLoader(
            self.testset,  batch_size=batch_size, shuffle=False
            )
        self.mem_loader   = torch.utils.data.DataLoader(
            self.trainset, batch_size=10, shuffle=True
            )

        self.classes = self.trainset.classes


class OxfordPetsDataSet():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.mean = [0.485, 0.456, 0.406]
        self.std  = [0.229, 0.224, 0.225]

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

        self.trainset = torchvision.datasets.OxfordIIITPet(
            root='./data', split='trainval', target_types='category',
            download=True, transform=self.train_transform
            )
        self.testset  = torchvision.datasets.OxfordIIITPet(
            root='./data', split='test',     target_types='category',
            download=True, transform=self.transform
            )

        self.train_loader = torch.utils.data.DataLoader(
            self.trainset, batch_size=batch_size, shuffle=True
            )
        self.test_loader = torch.utils.data.DataLoader(
            self.testset,  batch_size=batch_size, shuffle=False
            )
        self.mem_loader = torch.utils.data.DataLoader(
            self.trainset, batch_size=batch_size, shuffle=True
            )

        self.classes = [c.replace('_', ' ') for c in self.trainset.classes]