import os
import sys
import configparser
import glob
import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import classification_report,ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_config():
    config = configparser.ConfigParser()
    config.read(sys.argv[1])
    return config

def create_dataframe(metadata_file):
    df = pd.read_csv(metadata_file)

    # encode labels
    df['class'] = df['class'].astype('category')
    df['label_encoded'] = df['class'].cat.codes.astype('int64')

    return df

def get_weak_transforms(img_size,img_mean,img_std):
    weak_transforms = A.Compose([
                        A.Resize(img_size, img_size),
                        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=10, shift_limit=0.1, p=1, border_mode=0),
                        A.Normalize(mean=img_mean, std=img_std),
                        ToTensorV2(),
                    ],additional_targets={'roi':'image'})
    return weak_transforms

def get_strong_transforms(img_size,img_mean,img_std):
    strong_transforms = A.Compose([
                        A.Resize(img_size, img_size),
                        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=10, shift_limit=0.1, p=1, border_mode=0),
                        A.OneOf(
                            [
                                A.CLAHE(p=1),
                                A.RandomBrightnessContrast(p=1),
                                A.RandomGamma(p=1),
                            ],
                            p=0.9,
                        ),
                        A.OneOf(
                            [
                                A.Sharpen(p=1),
                                A.Blur(blur_limit=3, p=1),
                                A.MotionBlur(blur_limit=3, p=1),
                            ],
                            p=0.9,
                        ),
                        A.OneOf(
                            [
                                A.RandomBrightnessContrast(p=1),
                                A.HueSaturationValue(p=1),
                            ],
                            p=0.9,
                        ),
                        A.Normalize(mean=img_mean, std=img_std),
                        ToTensorV2(),
                    ],additional_targets={'roi':'image'})
    return strong_transforms

def get_valid_transforms(img_size,img_mean,img_std):
    valid_transforms = A.Compose([
                        A.Resize(img_size, img_size),
                        A.Normalize(mean=img_mean, std=img_std),
                        ToTensorV2(),
                    ],additional_targets={'roi':'image'})
    return valid_transforms


class Dataset(BaseDataset):
    """Read images, apply augmentation and preprocessing transformations."""
    
    def __init__(
            self,
            df,
            raw_dir,
            roi_dir,
            augmentation=None,
            visualize = False
    ):
        self.df = df.reset_index(drop=True)
        self.ids = self.df.loc[:,'file_name'].values
        self.images_fps = [os.path.join(raw_dir, image_id) for image_id in self.ids]
        self.roi_fps = [os.path.join(roi_dir, image_id) for image_id in self.ids]
        
        self.augmentation = augmentation
        self.visualize = visualize
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        roi = cv2.imread(self.roi_fps[i])
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        label = self.df.loc[i,'label_encoded']
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, roi=roi)
            image, roi = sample['image'], sample['roi']

        
        # Revert Normalize to visualize the image
            if self.visualize:
                invTrans = A.Normalize(mean=[-x/y for x,y in zip(img_mean,img_std)],
                                       std=[1/x for x in img_std],
                                       max_pixel_value=1.0,
                                       always_apply=True)
                image = image.detach().cpu().numpy().transpose(1,2,0)
                image = invTrans(image=image)['image']
                image = (image*255).astype(np.uint8)
        
        return image, roi, label
        
    def __len__(self):
        return len(self.ids)


def load_model(model_architecture,dropout,num_classes,pretrained=True,teacher_checkpoint=None):
    if model_architecture == 'convnext_small' or model_architecture == 'convnext_tiny':
        model = timm.create_model(model_architecture, pretrained=pretrained, num_classes=num_classes,drop_rate=dropout)

    if model_architecture == 'efficientnet_b0' or model_architecture == 'efficientnet_b1':
        model = timm.create_model(model_architecture, pretrained=pretrained, num_classes=num_classes)
        num_ftrs = model.get_classifier().in_features
        if dropout:
            model.classifier = nn.Sequential(
                                    nn.Dropout(dropout),
                                    nn.Linear(num_ftrs,num_classes)
            )
        else:
            model.classifier = nn.Linear(num_ftrs, num_classes)

    if model_architecture == 'resnet50':
        model = timm.create_model(model_architecture, pretrained=pretrained, num_classes=num_classes)
        num_ftrs = model.get_classifier().in_features
        if dropout:
            model.fc = nn.Sequential(
                                    nn.Dropout(dropout),
                                    nn.Linear(num_ftrs,num_classes)
            )
        else:
            model.fc = nn.Linear(num_ftrs, num_classes)

    if teacher_checkpoint:
        model.load_state_dict(torch.load(teacher_checkpoint))
        for param in model.parameters():
            param.requires_grad = False
        model = model.eval()

    return model


def train_teacher(config):
    # Configuration
    # Data
    img_mean = IMAGENET_DEFAULT_MEAN
    img_std = IMAGENET_DEFAULT_STD
    df = create_dataframe(config['settings']['metadata_file'])
    epoch_factor = 1.0/float(config['teacher']['roi_percent'])
    epochs_calc = math.floor(epoch_factor*int(config['teacher']['epochs']))

    # Training and model
    device = torch.device(f"cuda:{config['settings']['cuda']}" if torch.cuda.is_available() else "cpu")
    num_classes = 5

    # Init data
    if config['teacher']['augmentations'] == 'strong':
        augmentations = get_strong_transforms(int(config['teacher']['resolution']),img_mean,img_std)
    elif config['teacher']['augmentations'] == 'weak':
        augmentations = get_weak_transforms(int(config['teacher']['resolution']),img_mean,img_std)

    train_dataset = Dataset(
        df[df['split']=='train'].sample(frac=float(config['teacher']['roi_percent']),replace=False,random_state=42),
        raw_dir=config['settings']['raw_data_dir'],
        roi_dir=config['settings']['roi_data_dir'],
        augmentation=augmentations, 
    )

    valid_dataset = Dataset(
        df[df['split']=='valid'],
        raw_dir=config['settings']['raw_data_dir'],
        roi_dir=config['settings']['roi_data_dir'],
        augmentation=get_valid_transforms(int(config['teacher']['resolution']),img_mean,img_std), 
    )


    run_train_acc = {}
    run_val_acc = {}
    run_train_loss = {}
    run_val_loss = {}

    for run_number in range(int(config['settings']['repeated_runs'])):
        train_loader = DataLoader(train_dataset, batch_size=int(config['teacher']['batch_size']), shuffle=True, num_workers=12)
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

        # Init model
        model = load_model(config['teacher']['model_architecture'],float(config['teacher']['dropout_percent']),num_classes)
        model = model.to(device)

        # Init optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(),lr=0.0001)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs_calc-1)
        scaler = torch.cuda.amp.GradScaler()


        CHECKPOINT = f"./{config['settings']['model_dir']}/{config['settings']['experiment_name']}_teacher_{config['teacher']['model_architecture']}_run_{run_number}.pth"

        print(f'Run {run_number}')
        print('--------------------------------')

        # Training loop
        train_acc_list = []
        val_acc_list = []
        train_loss_list = []
        val_loss_list = []
        val_loss_min = np.Inf

        for epoch in range(epochs_calc):
            model.train()
            train_loss = []
            train_running_corrects = 0
            val_running_corrects = 0

            for _,rois,labels in train_loader:
                rois = rois.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    outputs = model(rois)
                    loss = criterion(outputs, labels)

                    train_loss.append(loss.item())

                    _, predicted = torch.max(outputs.data, 1)
                    #_,labels = torch.max(labels.data, 1)
                    train_running_corrects += torch.sum(predicted == labels.data)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                scheduler.step()

            train_loss = np.mean(train_loss)
            train_epoch_acc = train_running_corrects.double() / len(train_loader.dataset)

            model.eval()

            val_loss = 0

            # Validation loop
            with torch.cuda.amp.autocast(), torch.no_grad():    
                for _,rois, labels in valid_loader:
                    rois = rois.to(device)
                    labels = labels.to(device)

                    outputs = model(rois)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    #_,labels = torch.max(labels.data, 1)
                    val_running_corrects += torch.sum(predicted == labels.data)

            val_loss /= len(valid_loader.dataset)
            val_epoch_acc = val_running_corrects.double() / len(valid_loader.dataset)

            print(f'Epoch {epoch}: train loss: {train_loss:.5f} | train acc: {train_epoch_acc:.3f} | val_loss: {val_loss:.5f} | val acc: {val_epoch_acc:.3f}')

            train_acc_list.append(train_epoch_acc.item())
            val_acc_list.append(val_epoch_acc.item())
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)

            if val_loss < val_loss_min:
                    print(f'Valid loss improved from {val_loss_min:.5f} to {val_loss:.5f} saving model to {CHECKPOINT}')
                    val_loss_min = val_loss
                    best_epoch = epoch
                    torch.save(model.state_dict(), CHECKPOINT)

            print(f'Best epoch {best_epoch} | val loss min: {val_loss_min:.5f}')

        run_train_acc[run_number] = train_acc_list
        run_val_acc[run_number] = val_acc_list
        run_train_loss[run_number] = train_loss_list
        run_val_loss[run_number] = val_loss_list

        # Delete model just to be sure
        del loss, model, optimizer
        torch.cuda.empty_cache()
    
    # Final Results Output
    avg = 0.0
    for fold,val in run_val_acc.items():
        print(f'Highest val_acc for fold {fold}: {np.max(val):.3f}')
        avg += np.max(val)
    print(f'Average for all folds: {avg/len(run_val_acc.items()):.3f}')
    
    #Saving Metrics
    df = pd.DataFrame()
    for metric,name in zip([run_train_acc,run_train_loss,run_val_acc,run_val_loss],['train_acc','train_loss','val_acc','val_loss']):
        dffold = pd.DataFrame.from_dict(metric,orient='columns')
        dffold.columns = [f'fold{x}_{name}' for x in range(len(metric))]
        dffold = dffold.rename_axis('epochs')
        df = pd.concat([df,dffold],axis=1)
    df.to_csv(f"./logs/{config['settings']['experiment_name']}_teacher_{config['teacher']['model_architecture']}_metrics.csv")


class CustomLoss(nn.Module):
    def __init__(self, current_epoch,kd_epochs):
        super(CustomLoss, self).__init__()

        if current_epoch < kd_epochs:
            self.alpha_e = 0.5*(1-current_epoch/kd_epochs)
        else:
            self.alpha_e = 0.0
        
        
    def forward(self, teacher_features, features, y_pred, labels):
        consistency_loss = nn.MSELoss()(teacher_features.reshape(-1), features.reshape(-1))
        cls_loss = nn.CrossEntropyLoss()(y_pred, labels)
        loss = self.alpha_e * consistency_loss + (1-self.alpha_e) * cls_loss
        return loss, consistency_loss, cls_loss

def train_student(config):
    # Configuration
    # Data
    img_mean = IMAGENET_DEFAULT_MEAN
    img_std = IMAGENET_DEFAULT_STD
    df = create_dataframe(config['settings']['metadata_file'])

    # Training and model
    device = torch.device(f"cuda:{config['settings']['cuda']}" if torch.cuda.is_available() else "cpu")
    num_classes = 5

    # Init data
    if config['student']['augmentations'] == 'strong':
        augmentations = get_strong_transforms(int(config['student']['resolution']),img_mean,img_std)
    elif config['student']['augmentations'] == 'weak':
        augmentations = get_weak_transforms(int(config['student']['resolution']),img_mean,img_std)

    train_dataset = Dataset(
        df[df['split']=='train'],
        raw_dir=config['settings']['raw_data_dir'],
        roi_dir=config['settings']['roi_data_dir'],
        augmentation=augmentations, 
    )

    valid_dataset = Dataset(
        df[df['split']=='valid'],
        raw_dir=config['settings']['raw_data_dir'],
        roi_dir=config['settings']['roi_data_dir'],
        augmentation=get_valid_transforms(int(config['student']['resolution']),img_mean,img_std), 
    )


    run_train_acc = {}
    run_val_acc = {}
    run_train_loss = {}
    run_train_cons_loss = {}
    run_train_cls_loss = {}
    run_val_loss = {}

    for run_number in range(int(config['settings']['repeated_runs'])):
        train_loader = DataLoader(train_dataset, batch_size=int(config['teacher']['batch_size']), shuffle=True, num_workers=12)
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)



        # Load teacher model
        teacher = load_model(config['teacher']['model_architecture'],
                             float(config['teacher']['dropout_percent']),
                             num_classes,
                             teacher_checkpoint=f"./{config['settings']['model_dir']}/{config['settings']['experiment_name']}_teacher_{config['teacher']['model_architecture']}_run_{run_number}.pth")
        teacher = teacher.to(device)

        # Init student model
        model = load_model(config['student']['model_architecture'],float(config['student']['dropout_percent']),num_classes)
        model = model.to(device)

        # Init optimizer
        valid_criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(),lr=0.0001)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, int(config['student']['epochs'])-1)
        scaler = torch.cuda.amp.GradScaler()


        CHECKPOINT = f"./{config['settings']['model_dir']}/{config['settings']['experiment_name']}_student_{config['student']['model_architecture']}_run_{run_number}.pth"

        print(f'Run {run_number}')
        print('--------------------------------')

        # Training loop
        train_acc_list = []
        val_acc_list = []
        train_loss_list = []
        train_cons_loss_list = []
        train_cls_loss_list = []
        val_loss_list = []
        val_loss_min = np.Inf

        for epoch in range(int(config['student']['epochs'])):
            model.train()
            train_loss = []
            train_cons_loss = []
            train_cls_loss = []
            train_running_corrects = 0
            val_running_corrects = 0

            for images,rois,labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                rois = rois.to(device)

                with torch.no_grad():
                    teacher_features = teacher.forward_features(rois)

                optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    features = model.forward_features(images)

                    train_criterion = CustomLoss(current_epoch=epoch,kd_epochs=int(config['student']['kd_epochs']))
                    loss,consistency_loss, cls_loss = train_criterion(teacher_features, features, outputs, labels)

                    train_loss.append(loss.item())
                    train_cons_loss.append(consistency_loss.item())
                    train_cls_loss.append(cls_loss.item())

                    _, predicted = torch.max(outputs.data, 1)
                    #_,labels = torch.max(labels.data, 1)
                    train_running_corrects += torch.sum(predicted == labels.data)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                scheduler.step()

            train_loss = np.mean(train_loss)
            train_cons_loss = np.mean(train_cons_loss)
            train_cls_loss = np.mean(train_cls_loss)
            train_epoch_acc = train_running_corrects.double() / len(train_loader.dataset)

            model.eval()

            val_loss = 0

            # Validation loop
            with torch.cuda.amp.autocast(), torch.no_grad():    
                for images,_, labels in valid_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)
                    loss = valid_criterion(outputs, labels)

                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    #_,labels = torch.max(labels.data, 1)
                    val_running_corrects += torch.sum(predicted == labels.data)

            val_loss /= len(valid_loader.dataset)
            val_epoch_acc = val_running_corrects.double() / len(valid_loader.dataset)

            print(f'Epoch {epoch}: train loss: {train_loss:.5f} | train acc: {train_epoch_acc:.3f} | val_loss: {val_loss:.5f} | val acc: {val_epoch_acc:.3f}')

            train_acc_list.append(train_epoch_acc.item())
            val_acc_list.append(val_epoch_acc.item())
            train_loss_list.append(train_loss)
            train_cons_loss_list.append(train_cons_loss)
            train_cls_loss_list.append(train_cls_loss)
            val_loss_list.append(val_loss)

            if val_loss < val_loss_min:
                    print(f'Valid loss improved from {val_loss_min:.5f} to {val_loss:.5f} saving model to {CHECKPOINT}')
                    val_loss_min = val_loss
                    best_epoch = epoch
                    torch.save(model.state_dict(), CHECKPOINT)

            print(f'Best epoch {best_epoch} | val loss min: {val_loss_min:.5f}')

        run_train_acc[run_number] = train_acc_list
        run_val_acc[run_number] = val_acc_list
        run_train_loss[run_number] = train_loss_list
        run_train_cons_loss[run_number] = train_cons_loss_list
        run_train_cls_loss[run_number] = train_cls_loss_list
        run_val_loss[run_number] = val_loss_list

        # Delete model just to be sure
        del loss, model, optimizer
        torch.cuda.empty_cache()
    
    # Final Results Output
    avg = 0.0
    for fold,val in run_val_acc.items():
        print(f'Highest val_acc for fold {fold}: {np.max(val):.3f}')
        avg += np.max(val)
    print(f'Average for all folds: {avg/len(run_val_acc.items()):.3f}')
    
    #Saving Metrics
    df = pd.DataFrame()
    for metric,name in zip([run_train_acc,run_train_loss,run_train_cons_loss,run_train_cls_loss,run_val_acc,run_val_loss],['train_acc','train_loss','train_cons_loss','train_cls_loss','val_acc','val_loss']):
        dffold = pd.DataFrame.from_dict(metric,orient='columns')
        dffold.columns = [f'fold{x}_{name}' for x in range(len(metric))]
        dffold = dffold.rename_axis('epochs')
        df = pd.concat([df,dffold],axis=1)
    df.to_csv(f"./logs/{config['settings']['experiment_name']}_student_{config['student']['model_architecture']}_metrics.csv")


def main(config):
    print(f"Starting Experiment: {config['settings']['experiment_name']}")
    print('--------------------------------')
    train_teacher(config)
    train_student(config)

    
if __name__ == "__main__":
    #args = parse_args()
    config = parse_config()
    main(config)