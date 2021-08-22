import os
import cv2
import argparse
import numpy as np
import random
from pathlib import Path
import torch
import gc
from PIL import Image
from data_preparation import fix_bbox,TRAIN_PATH,TEST_PATH
from evaluation import make_model
import json

import warnings
warnings.filterwarnings("ignore")
import emoji
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def obj_detector(img,model,threshold):
    img = cv2.imread(img)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32)
    img /= 255.0
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    img = img.permute(0,3,1,2)

    model.eval()
    img = img.to(device)
    output = model(img)

    boxes = output[0]["boxes"].data.cpu().numpy()
    scores = output[0]["scores"].data.cpu().numpy()
    labels = output[0]["labels"].data.cpu().numpy()

    labels = labels[scores >= threshold]
    boxes = boxes[scores >= threshold].astype(np.float32)
    scores = scores[scores >= threshold]

    sample = img[0].permute(1,2,0).cpu().numpy()

    return boxes, sample,labels

def view_images(number_of_images,train=False,test=False,predict=False,weights='weights/model_weight65.pth',threshold=0.5):
    """
    this function is for visualize images with bounding boxes.
    """
    assert number_of_images < 51, "🚧Maximum number of images for view is 50...🚧"
    im_dict= {}
    horizontal_dict = {}
    m = 0   
    color_dict = {0:(255,0,0),1:(0,255,0),2:(0,0,255),3:(53,0,0),4:(0,53,0),5:(0,0,53),6:(130,38,0),7:(0,138,39),8:(39,0,139),9:(10,40,60),10:(60,40,10)}
    total_im = number_of_images
    if predict:
        path = Path(weights)
        assert path.is_file(), "🔎Weights are missing at given path🔎"        
        _,img_test = fix_bbox()
        path = os.listdir(TEST_PATH)
        weights = weights
        model,device = make_model(weights)
        print(emoji.emojize('\nPredictions on images started...⏳\n'))
        for i, img in enumerate(path):
            if i >= number_of_images: continue
            image = TEST_PATH + img
            boxes, sample,labels = obj_detector(image,model,threshold)
            for n, box in enumerate(boxes):
                cv2.rectangle(sample,
                      (int(box[0]), int(box[1])),
                      (int(box[2]), int(box[3])),
                     color_dict[labels[n]-1], 4)
                
                cv2.putText(sample,str(labels[n]-1),(int(box[0]),int(box[1])-5),cv2.FONT_HERSHEY_COMPLEX,3,color_dict[labels[n]-1],1,cv2.LINE_AA)
            image = cv2.resize(sample,(420,360),interpolation=cv2.INTER_AREA)   
            num_rows, num_cols = image.shape[:2]    
            translation_matrix = np.float32([[1,0, 0], [0, 1, 0]])
            image = cv2.warpAffine(image, translation_matrix, (num_cols, num_rows + 50))
            image = cv2.putText(image, img, (15,400), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255,255,255), 2, cv2.LINE_AA)     
            im_dict[i] = image
            if number_of_images < 5:
                if int(i+1) == number_of_images:
                    horizontal = [n for n in im_dict.values()]
                    horizontal_out = np.hstack(horizontal)
                    cv2.namedWindow("predictions",cv2.WINDOW_NORMAL)
                    cv2.imshow("predictions",horizontal_out)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            if int(i+1)%5== 0:     
                horizontal = [n for n in im_dict.values()]
                horizontal_dict[m] = np.hstack(horizontal)
                horizontal = []
                m += 1
                im_dict = {}
        if number_of_images >= 5:
            gc.collect()
            vertical = [n for n in horizontal_dict.values()]
            cv2.namedWindow("predictions", cv2.WINDOW_GUI_NORMAL)
            vertical = np.vstack(vertical)
            cv2.imshow("predictions",vertical)
            cv2.waitKey(0)
            cv2.destroyAllWindows()           
    else:
        if test:
            _,df = fix_bbox()
            path = os.listdir(TEST_PATH)
            img_path = TEST_PATH
        else:
            df,_ = fix_bbox()
            path = os.listdir(TRAIN_PATH)
            img_path = TRAIN_PATH
        for i,p in enumerate(path[:total_im]):
            if i > total_im: continue
            x_min = df[df.images==p]["x0"].values
            y_min = df[df.images==p]["y0"].values
            x_max = df[df.images==p]["x1"].values
            y_max = df[df.images==p]["y1"].values

            the_path = img_path + p
            image = cv2.imread(the_path)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            # image=np.flipud(image).copy()
            for i,(x0,y0,x1,y1)in enumerate(zip(x_min,y_min,x_max,y_max)):
                cv2.rectangle(image,(int(x0),int(y0)),(int(x1),int(y1)),(1,255,0),5)             
            t = p 
            image = cv2.resize(image,(420,360),interpolation=cv2.INTER_AREA) 
            num_rows, num_cols = image.shape[:2]    
            translation_matrix = np.float32([[1,0, 0], [0, 1, 0]])
            image = cv2.warpAffine(image, translation_matrix, (num_cols, num_rows + 50))
            image = cv2.putText(image, t, (15,400), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255,255,255), 2, cv2.LINE_AA)
            im_dict[i] = image
            if number_of_images < 5:
                if int(i+1) == number_of_images:
                    horizontal = [n for n in im_dict.values()]
                    horizontal_out = np.hstack(horizontal)
                    cv2.namedWindow("output",cv2.WINDOW_NORMAL)
                    cv2.imshow("output",horizontal_out)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            if int(i+1)%5== 0:     
                horizontal = [n for n in im_dict.values()]
                horizontal_dict[m] = np.hstack(horizontal)
                horizontal = []
                m += 1
                im_dict = {}
        if number_of_images >= 5:
            vertical = [n for n in horizontal_dict.values()]
            cv2.namedWindow("output", cv2.WINDOW_NORMAL)
            vertical = np.vstack(vertical)
            cv2.imshow("output",vertical)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def product_count(weights,threshold):
    """
    This function is for counting items for per images and save the resut in json format
    """
    path = Path(weights)
    assert path.is_file(), "🔎Weights are missing at given path🔎"        
    _,img_test = fix_bbox()
    path = os.listdir(TEST_PATH)
    weights = weights
    model,device = make_model(weights)
    number_of_images =len(path)
    print(emoji.emojize('\nPredictions on images started...⏳\n'))
    image_dict = {}
    for i, img in enumerate(path):
        if i >= number_of_images: continue
        image = TEST_PATH + img
        boxes, sample = obj_detector(image,model,threshold)
        image_dict[img] = len(boxes)
    
    with open('image2products.json','w',encoding='utf-8') as f:
        json.dump(image_dict, f, ensure_ascii=False, indent=4)
            

@torch.no_grad()
def run(weights="saved_weights.pth", 
        iou_thres=0.5,
        view_train=False,
        view_test=False,
        view_predict=False,
        num_of_img=20,
        count_product=False):
        
        assert (view_train != True or view_test != True),"""
        🚧 Choose train or evolution one at a time 🚧
        """
        path = Path(weights)
        assert path.is_file(), """
        🔎Weights are missing from given path🔎"""        
 
        if view_predict:
            view_images(number_of_images=num_of_img,predict=view_predict,weights=weights,threshold=iou_thres)

        if view_train:
            view_images(number_of_images=num_of_img,train=view_train)

        if view_test:
            view_images(number_of_images=num_of_img,test=view_test)

        if count_product:
            product_count(weights=weights,threshold=iou_thres)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",default="saved_weights.pth",help="model.pt path(s)")
    parser.add_argument("--iou-thres",type=float,default=0.5, help="confidence threshold")
    parser.add_argument("--view-train",action="store_true",help="show train images")
    parser.add_argument("--view-test",action="store_true",help="show test images")
    parser.add_argument("--view-predict",action="store_true",help="show prediction")
    parser.add_argument("--num-of-img",type=int,default=10,help="Number of images for view.")
    parser.add_argument("--count-product",action="store_true",help="count number of product for per image.")
    opt = parser.parse_args()
    return opt

def main(opt):
    run(**vars(opt))

if __name__=="__main__":
    gc.collect()
    opt = parse_opt()
    main(opt)
