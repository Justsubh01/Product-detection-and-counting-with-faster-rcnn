try:
    from data_preparation import get_data_loaders
    import torch
    import time
    import os
    import random
    import numpy as np
    import argparse
    import torchvision
    from torch.utils.data import DataLoader, Dataset
    from torch.utils.data import SequentialSampler
    from albumentations.pytorch.transforms import ToTensorV2
    import torch.backends.cudnn as cudnn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.rpn import AnchorGenerator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import warnings
    warnings.filterwarnings("ignore")
    import emoji
except :
    print("Not all modules imported correctly!")

header = r'''
                    epochs  |  Loss 
'''
raw_line = '                    {:6d}' + '\u2502{:6.2f}' + '\u2502{:6.2f}'

class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train(batch_size=4,epochs=25,step_size=8,lr_rate=0.0001,weight_decay=0.0001,gamma=0.1):

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 12

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adamw(params, lr=lr_rate, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    train_data_loader,_= get_data_loaders(train=True,train_batch_size=batch_size)

    print(emoji.emojize("\nLet's start training.. 🧘‍♂️"))
    start_time = time.time()
    loss_hist = Averager()
    
    for epoch in range(epochs):
        loss_hist.reset()
        model.train()
        for images, targets in train_data_loader:
            
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            loss_hist.send(loss_value)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        
        # update the learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()
        print(header)
        print(raw_line.format(epoch,loss_hist.value,get_lr(optimizer)))

    torch.save(model.state_dict(), 'saved_weights.pth')
    print("\n--- %s seconds ---" % (time.time() - start_time))

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size",type=int,default=4,help="total batch size for all GPUs")
    parser.add_argument("--epochs",type=int,default=15)
    parser.add_argument("--step-size",type=int,default=9)
    parser.add_argument("--lr-rate",type=float,default=0.0001)
    parser.add_argument("--weight-decay",type=float,default=0.0001)
    parser.add_argument("--gamma",type=float,default=0.1)
    opt = parser.parse_args()
    return opt

def main(opt):
    train(**vars(opt))

if __name__=="__main__":
    SEED = 42
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

    opt = parse_opt()
    main(opt)




