import torch
from torch.utils.data import Dataset, DataLoader
import random
from PIL import Image
import PIL
import pandas as pd
# import argparse
# import sys


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class SiameseNetworkDataset(Dataset):

    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        # we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                # keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            while True:
                # keep looping till a different class image is found

                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)


# df_train = pd.read_csv('')

# trn = FaceVerificationDataset(df_train.loc[:2000])


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-bs", "--batch_size", type=int, default=32,
#                         help="batch size the data Loader")
#     parser.add_argument("-dl", "--drop_last", type=str, default="True",
#                         help="droping the last few data if the\
#                         batch size doesn't fit")

#     args = parser.parse_args()
#     batch_size = args.batch_size
#     drop_last = True if args.drop_last == "True" else False

#     sys.stdout.write("batch size:"+str(batch_size)+'\n')
#     sys.stdout.write("drop last:"+str(drop_last))

#     TRAIN_LOADER = DataLoader(trn, batch_size=batch_size, shuffle=True,
#                               drop_last=bool(drop_last),
#                               collate_fn=trn.collate_fn)
#     img1, img2, lab = next(iter(TRAIN_LOADER))
#     print(img1.shape, img2.shape, lab.shape)
