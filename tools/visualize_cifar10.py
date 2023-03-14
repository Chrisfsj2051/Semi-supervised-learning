import matplotlib.pyplot as plt
import os
import torchvision

def read_ds():
    data_dir = './data/cifar10/'
    dset = torchvision.datasets.CIFAR10(data_dir, train=True, download=True)
    return dset

def visualize_img():
    pass

def save_img():
    pass

if __name__ == '__main__':
    ds = read_ds()
    for img, label in zip(ds.data, ds.targets):
        visualize_img(img, label)