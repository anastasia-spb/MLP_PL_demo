import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def prepare_mnist_data(data_folder, batch_size):
    train_data = datasets.MNIST(root=data_folder,
                                train=True,
                                download=True)

    mean = train_data.data.float().mean() / 255
    std = train_data.data.float().std() / 255

    train_transforms = transforms.Compose([
        transforms.RandomRotation(5, fill=(0,)),
        transforms.RandomCrop(28, padding=2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std])
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std])
    ])

    train_data = datasets.MNIST(root=data_folder,
                                train=True,
                                download=True,
                                transform=train_transforms)

    test_data = datasets.MNIST(root=data_folder,
                               train=False,
                               download=True,
                               transform=test_transforms)

    train_iterator = data.DataLoader(train_data,
                                     shuffle=True,
                                     batch_size=batch_size)

    test_iterator = data.DataLoader(test_data,
                                    batch_size=batch_size)

    return train_iterator, test_iterator