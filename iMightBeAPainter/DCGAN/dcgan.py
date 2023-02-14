# Model of DCGAN
import torch
import torch.nn as nn

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1) -> None:
        super().__init__()

        self.layers = nn.Sequential(
                nn.ConvTranspose2d(in_channels = in_channels,
                                   out_channels = out_channels,
                                   kernel_size = kernel_size,
                                   stride = stride,
                                   padding = padding,
                                   bias = False),

                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(inplace = True))

    def forward(self, img):
        return self.layers(img)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1, first_layer = False) -> None:
        super().__init__()

        self.first_layer = first_layer
        self.conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(inplace = True)

    def forward(self, x):
        x = self.conv(x)
        if not self.first_layer:
            x = self.batch_norm(x)
        x = self.leaky_relu(x)

        return x

class DCGAN(nn.Module):
    def __init__(self,
                 latent_dim = 256,
                 num_feat_map = 64,
                 color_channels = 3) -> None:
        super().__init__()

        # Generator
        generator = nn.ModuleList()
        # layer 1
        generator.append(UpBlock(latent_dim, num_feat_map * 8, kernel_size=4, stride=1, padding=0))

        # Subsequent hidden layers
        layers = [2**i for i in range(3, -1, -1)]
        for i in range(len(layers) - 1):
            layer = UpBlock(num_feat_map * layers[i], num_feat_map * layers[i + 1])
            generator.append(layer)

        # Last layer
        generator.extend((nn.ConvTranspose2d(num_feat_map, color_channels, kernel_size=4, stride=2, padding=1), nn.Tanh()))
        self.generator = generator

        # Discriminator
        discriminator = nn.ModuleList()
        discriminator.append(DownBlock(color_channels, num_feat_map, kernel_size=4, stride=2, padding=1, first_layer=True))

        layers = [2**i for i in range(0,4)]

        for i in range(len(layers) - 1):
            layer = DownBlock(num_feat_map * layers[i], num_feat_map * layers[i + 1])
            discriminator.append(layer)

        discriminator.extend((nn.Conv2d(num_feat_map * layers[-1], 1, kernel_size=4, stride=2, padding=0), nn.Flatten(), nn.Sigmoid()))

        self.discriminator = discriminator

    def gen_forward(self, img):
        return self.generator(img)

    def dis_forward(self, img):
        return self.discriminator(img)

def main():
    model = DCGAN()
    print(model)

main()
