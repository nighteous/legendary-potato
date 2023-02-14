import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from dcgan import DCGAN
from dataset import Monet_Dataset

def arguments():
    parser = argparse.ArgumentParser()

    # Defining options/training settings
    parser.add_argument("-bs", "--batch-size", metavar="batch_size", default=32, help="Batch Size for Loader.\nDefault batch_size = 32")
    parser.add_argument("--split-size", metavar="split_size", default=0.8, help="Split size of dataset into training and testing.\nDefault split_size=0.8")
    parser.add_argument("--device", metavar="device", default="cuda", help="device to run training and model on.\nDefault device = cuda")
    parser.add_argument("--epochs", metavar="epochs", default=10, help="Number of epochs.\nDefault epochs = 10")
    args = parser.parse_args()

    return args

def split_dataset(full_dataset, split_size = 0.8):
    train_size = int(split_size * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    return train_dataset, test_dataset

def train_generator(model, train_loader, args):
    epochs = args.epochs

    for epoch in range(epochs):
        for _, img in tqdm(enumerate(train_loader)):


def main():
    # Split dataset into train and test
    args = arguments()

    dataset = Monet_Dataset()
    train_dataset, test_dataset = split_dataset(dataset, args.split_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    model = DCGAN().to(args.device)

    train_generator(model, train_loader, args)

if __name__ == "__main__":
    main()
