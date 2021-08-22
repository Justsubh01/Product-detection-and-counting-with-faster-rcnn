try:
    import argparse
    import sys
    import time
    from pathlib import Path
    import tarfile
    import gc
    import wget
    import os
    import shutil
    import cv2
    import emoji
    import pandas as pd
    import numpy as np
    import functools
    from PIL import Image
    import matplotlib
    import matplotlib.pyplot as plt
    import albumentations as A

    import torch
    import torchvision
    from torch.utils.data import DataLoader, Dataset
    from torch.utils.data import SequentialSampler
    from albumentations.pytorch.transforms import ToTensorV2
    import torch
    import torch.backends.cudnn as cudnn
    import warnings
    warnings.filterwarnings("ignore")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except :
    print("Not All Modules imported successfully!")

TRAIN_PATH = os.path.join("./data/ShelfImages/train/")
TEST_PATH = os.path.join("./data/ShelfImages/test/")


def get_data():
    """
    Download data if not already downloaded.
    returns: train images name list,test images name list and Pandas DataFrame
    """
    url = "https://storage.googleapis.com/open_source_datasets/ShelfImages.tar.gz"
    try:
        train_path = os.path.join(TRAIN_PATH)
        test_path = os.path.join(TEST_PATH)
        path = os.listdir(train_path)
        data = pd.read_csv("./data/grocerydataset/annotations.csv",names=["images","x0","y0","x1","y1","category"])
        check = path[0]
    except:    
        data = pd.read_csv("./data/grocerydataset/annotations.csv",names=["images","x0","y0","x1","y1","category"])
        print(emoji.emojize('\nYou can take a cup of tea ☕ , while data is preparing...\n'))
        filename = wget.download(url,out="./data/ShelfImages.tar.gz")
        tar = tarfile.open(filename, "r:gz")
        tar.extractall("./data/")
        tar.close()
        train_path = os.path.join(TRAIN_PATH)
        test_path = os.path.join(TEST_PATH)

    return train_path,test_path,data

def separate_train_test():
    """
    this function makes separate dataframe for test data.
    return: train dataframe , test dataframe
    """
    train_path,test_path,data = get_data()
    val_df = data.copy()
    train_df = data.copy()
    x = data["images"].unique()
    for i in x:
        train_file = train_path + "/" + i
        path = Path(train_file)
        if path.is_file():
            val_df = val_df[val_df.images != i]
        else:
            train_df = train_df[train_df.images != i]
    return train_df,val_df


def fix_bbox():
    """
    this function fixes wrongly rotated images, converts bounding box data to float , adding 1 to every category.
    because i want 0 to be is_crowd and 1 to 12 for object categories.
    return: it returns preprocessed dataframes.
    """
    list_of_rotated_img = ["C1_P12_N1_S4_1.JPG","C1_P12_N2_S5_1.JPG","C1_P03_N1_S3_2.JPG","C1_P12_N2_S4_1.JPG","C1_P03_N2_S3_2.JPG",
                  "C1_P12_N3_S3_1.JPG","C1_P12_N2_S2_1.JPG","C1_P12_N3_S4_1.JPG","C1_P12_N3_S2_1.JPG","C1_P03_N1_S2_2.JPG",
                  "C1_P12_N4_S3_1.JPG","C3_P07_N1_S6_1.JPG","C1_P12_N4_S2_1.JPG"]
    
    train_df,val_df = separate_train_test()
    image = "data/ShelfImages/train/C3_P07_N1_S6_1.JPG"
    image = cv2.imread(image)
    row,col = image.shape[:2]
    if row < col:
        for i,p in enumerate(list_of_rotated_img):
            train_path =  TRAIN_PATH
            the_path = train_path + p
            image = cv2.imread(the_path)
            if p == "C3_P07_N1_S6_1.JPG":
                image = cv2.rotate(image,cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                image = cv2.rotate(image, cv2.ROTATE_180)
            cv2.imwrite(the_path,image)
        print(emoji.emojize('Data is ready for train or evaluate :thumbs_up:'))
    if train_df["category"].isin([0]).any():
        
        train_df["x0"] = train_df["x0"].astype(np.float)
        train_df["y0"] = train_df["y0"].astype(np.float)
        train_df["x1"] = train_df["x1"].astype(np.float)
        train_df["y1"] = train_df["y1"].astype(np.float)

        val_df["x0"] = val_df["x0"].astype(np.float)
        val_df["y0"] = val_df["y0"].astype(np.float)
        val_df["x1"] = val_df["x1"].astype(np.float)
        val_df["y1"] = val_df["y1"].astype(np.float)
        train_df["category"] = train_df["category"].apply(lambda x:x+1)
        val_df["category"] = val_df["category"].apply(lambda x:x+1)
    return train_df, val_df

class GdCDataset(Dataset):
    
    def __init__(self, dataframe, image_dir, transforms=None):
        super().__init__()
        
        self.image_ids = dataframe['images'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms
    
    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        records = self.df[self.df['images'] == image_id]

        image = cv2.imread(f'{self.image_dir}/{image_id}', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        rows, cols = image.shape[:2]
        
        boxes = records[['x0', 'y0', 'x1', 'y1']].values

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)
        
        label = records['category'].values
        labels = torch.as_tensor(label, dtype=torch.int64)
        
        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        # target['masks'] = None
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        target['iscrowd'] = iscrowd
    
        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']
            
            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1,0).type(torch.FloatTensor)
            
            return image, target
    
    def __len__(self) -> int:
        return self.image_ids.shape[0]


def get_transform_train():
    """
    function for train data augmentations, for bounding boxes i am using pascal_voc format (xmin,ymin,xmax,ymax).
    """
    return A.Compose([
            A.HorizontalFlip(p=0.3),
            A.RandomBrightnessContrast(p=0.5),
            A.ToGray(p=0.01),
            A.VerticalFlip(p=0.4),
            ToTensorV2(p=1.0),
    ], bbox_params={'format':'pascal_voc', 
                'label_fields': ['labels']})

def get_transform_valid():
    # function for validation data augmentations, for bounding boxes i am using pascal_voc format (xmin,ymin,xmax,ymax).
    return A.Compose([
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc',
          'label_fields':['labels']})

def collate_fn(batch):
    return tuple(zip(*batch))

def get_data_loaders(train=False,val=True,train_batch_size=4,valid_batch_size=4):
    """
    this function returns Dataloader if train does't set to True (by default). 
    and if train set to true then both train and validation dataloader will return 
    """
    if train == False:
        _,val_df = fix_bbox()
        valid_dataset = GdCDataset(val_df, TEST_PATH, get_transform_valid())        
        valid_data_loader = DataLoader(
                valid_dataset,
                batch_size=valid_batch_size,
                shuffle=False,
                num_workers=2,
                collate_fn=collate_fn
            )
        return valid_data_loader

    else: 
        train_df,val_df = fix_bbox()
        train_dataset = GdCDataset(train_df,TRAIN_PATH,get_transform_train())
        valid_dataset = GdCDataset(val_df, TEST_PATH, get_transform_valid())        
        valid_data_loader = DataLoader(
                valid_dataset,
                batch_size=train_batch_size,
                shuffle=False,
                num_workers=2,
                collate_fn=collate_fn
            )
        train_data_loader = DataLoader(
                train_dataset,
                batch_size=valid_batch_size,
                shuffle=True,
                num_workers=2,
                collate_fn=collate_fn
            )
    
        return train_data_loader,valid_data_loader

if __name__=='__main__':
    valid_data_loader = get_data_loaders()
    gc.collect()  