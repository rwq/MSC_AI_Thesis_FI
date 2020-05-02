import numpy as np
from os import listdir
from PIL import Image
from os.path import join, isdir
from torch.utils.data import Dataset
from torchvision import transforms


class DBreader_frame_interpolation(Dataset):
    """
    DBreader reads all triplet set of frames in a directory.
    Each triplet set contains frame 0, 1, 2.
    Each image is named frame0.png, frame1.png, frame2.png.
    Frame 0, 2 are the input and frame 1 is the output.
    """

    def __init__(self, db_dir, resize=None):
        if resize is not None:
            self.transform = transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])

        self.triplet_list = np.array([(db_dir + '/' + f) for f in listdir(db_dir) if isdir(join(db_dir, f))])
        self.file_len = len(self.triplet_list)

    def __getitem__(self, index):
        frame0 = self.transform(Image.open(self.triplet_list[index] + "/frame0.png"))
        frame1 = self.transform(Image.open(self.triplet_list[index] + "/frame1.png"))
        frame2 = self.transform(Image.open(self.triplet_list[index] + "/frame2.png"))

        return frame0, frame1, frame2

    def __len__(self):
        return self.file_len
