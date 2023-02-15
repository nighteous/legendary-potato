from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dcgan import DCGAN
from dataset import Monet_Dataset

# Settings
BATCH_SIZE = 32
LEARNING_RATE = 2e-4
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



def split_dataset(full_dataset, split_size = 0.8):
    train_size = int(split_size * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    return train_dataset, test_dataset

def train_dis(model: DCGAN, opt, real_image):

    # Predicting real images
    opt.zero_grad()
    real_image = real_image.to(DEVICE)
    real_preds = model.dis_forward(real_image)
    real_targets = torch.ones(real_image.size(0), 1).to(DEVICE)
    real_loss = torch.nn.functional.binary_cross_entropy(real_preds, real_targets)
    real_scores = torch.mean(real_preds).item()

    # Predicting fake images
    latent = torch.randn(BATCH_SIZE, LATENT_SIZE, 1, 1).to(DEVICE)


def fit(model: DCGAN, train_loader: DataLoader):

    losses_d = []
    losses_g = []
    real_scores = []
    fake_scores = []

    opt_d = optim.Adam(model.discriminator.parameters(), lr = LEARNING_RATE, betas=(0.5, 0.999))
    opt_g = optim.Adam(model.generator.parameters(), lr = LEARNING_RATE, betas=(0.5, 0.999))

    for epoch in tqdm(range(EPOCHS)):
        for real_images in train_loader:
            # Training discriminator


def main():

    full_dataset = Monet_Dataset()
    train_dataset, test_dataset = split_dataset(full_dataset)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = DCGAN().to(DEVICE)

    fit(model, train_loader)


main()
