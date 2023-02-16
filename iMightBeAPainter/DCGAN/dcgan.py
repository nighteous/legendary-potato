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
        self.conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(inplace = True)

    def forward(self, x):
        x = self.conv(x)
        if not self.first_layer:
            x = self.batch_norm(x)
        x = self.leaky_relu(x)

        return x


class Fuddi_Gan(nn.Module):
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

        discriminator.extend((nn.Conv2d(num_feat_map * layers[-1], 1, kernel_size=4, stride=1, padding=0), nn.Flatten(), nn.Sigmoid()))

        self.discriminator = discriminator

    def gen_forward(self, noise):
        for layer in self.generator:
            noise = layer(noise)

        return noise

    def dis_forward(self, img):
        for layer in self.discriminator:
            img = layer(img)

        return img

class DCGAN(torch.nn.Module):

    def __init__(self, latent_dim=256,
                 num_feat_maps_gen=64, num_feat_maps_dis=64,
                 color_channels=3):
        super().__init__()
        
        
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, num_feat_maps_gen*8, 
                               kernel_size=4, stride=1, padding=0,
                               bias=False),
            nn.BatchNorm2d(num_feat_maps_gen*8),
            nn.LeakyReLU(inplace=True),
            #
            # size if latent_dim=100: num_feat_maps_gen*8 x 4 x 4
            #
            nn.ConvTranspose2d(num_feat_maps_gen*8, num_feat_maps_gen*4, 
                               kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(num_feat_maps_gen*4),
            nn.LeakyReLU(inplace=True),
            #
            # size if latent_dim=100: num_feat_maps_gen*4 x 8 x 8
            #
            nn.ConvTranspose2d(num_feat_maps_gen*4, num_feat_maps_gen*2, 
                               kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(num_feat_maps_gen*2),
            nn.LeakyReLU(inplace=True),
            #
            # size if latent_dim=100: num_feat_maps_gen*2 x 16 x 16
            #
            nn.ConvTranspose2d(num_feat_maps_gen*2, num_feat_maps_gen, 
                               kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(num_feat_maps_gen),
            nn.LeakyReLU(inplace=True),
            #
            # size if latent_dim=100: num_feat_maps_gen x 32 x 32
            #
            nn.ConvTranspose2d(num_feat_maps_gen, color_channels, 
                               kernel_size=4, stride=2, padding=1,
                               bias=False),
            #
            # size: color_channels x 64 x 64
            #  
            nn.Tanh()
        )
        
        self.discriminator = nn.Sequential(
            #
            # input size color_channels x image_height x image_width
            #
            nn.Conv2d(color_channels, num_feat_maps_dis,
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            #
            # size: num_feat_maps_dis x 32 x 32
            #              
            nn.Conv2d(num_feat_maps_dis, num_feat_maps_dis*2,
                      kernel_size=4, stride=2, padding=1,
                      bias=False),        
            nn.BatchNorm2d(num_feat_maps_dis*2),
            nn.LeakyReLU(inplace=True),
            #
            # size: num_feat_maps_dis*2 x 16 x 16
            #   
            nn.Conv2d(num_feat_maps_dis*2, num_feat_maps_dis*4,
                      kernel_size=4, stride=2, padding=1,
                      bias=False),        
            nn.BatchNorm2d(num_feat_maps_dis*4),
            nn.LeakyReLU(inplace=True),
            #
            # size: num_feat_maps_dis*4 x 8 x 8
            #   
            nn.Conv2d(num_feat_maps_dis*4, num_feat_maps_dis*8,
                      kernel_size=4, stride=2, padding=1,
                      bias=False),        
            nn.BatchNorm2d(num_feat_maps_dis*8),
            nn.LeakyReLU(inplace=True),
            #
            # size: num_feat_maps_dis*8 x 4 x 4
            #   
            nn.Conv2d(num_feat_maps_dis*8, 1,
                      kernel_size=4, stride=1, padding=0),
            
            # size: 1 x 1 x 1
            nn.Flatten(),
            nn.Sigmoid()
            
        )

            
    def gen_forward(self, z):
        img = self.generator(z)
        return img
    
    def dis_forward(self, img):
        logits = self.discriminator(img)
        return logits
