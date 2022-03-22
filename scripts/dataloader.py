import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import pandas as pd
import argparse
import sys


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class FaceVerificationDataset(Dataset):
    def __init__(self, df, transforms=True, IMAGE_SIZE=224):
        self.df = df
        self.IMAGE_SIZE = IMAGE_SIZE

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        f = self.df.iloc[idx].squeeze()
        file1 = f.file_path_1
        img1 = cv2.imread(file1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

        file2 = f.file_path_2
        img2 = cv2.imread(file2)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        score = f.similarity_score
        return img1, img2, score

    def preprocess_image(self, img):
        im = cv2.resize(img, (self.IMAGE_SIZE, self.IMAGE_SIZE))
        im = torch.tensor(im)
        im = im/255.
        return im[None]

    def collate_fn(self, batch):
        '''
        preprocess images, similariity scores
        '''
        imgs1, imgs2, scores = [], [], []
        for im1, im2, score in batch:

            im1 = self.preprocess_image(im1)
            imgs1.append(im1)

            im2 = self.preprocess_image(im2)
            imgs2.append(im2)

            scores.append([float(score)])

        labels = torch.tensor(scores).to(device).float()

        imgs1 = torch.cat(imgs1).to(device)
        imgs1 = imgs1.view(-1, 3, 224, 224)

        imgs2 = torch.cat(imgs2).to(device)
        imgs2 = imgs2.view(-1, 3, 224, 224)

        return imgs1, imgs2, labels


df_train = pd.read_csv('')

trn = FaceVerificationDataset(df_train.loc[:2000])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", "--batch_size", type=int, default=32,
                        help="batch size the data Loader")
    parser.add_argument("-dl", "--drop_last", type=str, default="True",
                        help="droping the last few data if the\
                        batch size doesn't fit")

    args = parser.parse_args()
    batch_size = args.batch_size
    drop_last = True if args.drop_last == "True" else False

    sys.stdout.write("batch size:"+str(batch_size)+'\n')
    sys.stdout.write("drop last:"+str(drop_last))

    TRAIN_LOADER = DataLoader(trn, batch_size=batch_size, shuffle=True,
                              drop_last=bool(drop_last),
                              collate_fn=trn.collate_fn)
    img1, img2, lab = next(iter(TRAIN_LOADER))
    print(img1.shape, img2.shape, lab.shape)
