import os
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class Monet_Dataset(Dataset):
    def __init__(self,
                 pathToDataset = "../../dataset/gan-getting-started/monet_jpg/",
                 transform = None
                 ) -> None:
        super().__init__()

        self.pathToDataset = pathToDataset

        # Defining transformer
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.ToTensor()

        self.imageDir = os.listdir(pathToDataset)

    def __len__(self):
        return len(self.imageDir)

    def __getitem__(self, index):
        imagePath = self.pathToDataset + self.imageDir[index]
        image = Image.open(imagePath)
        image = self.transform(image)

        return image
