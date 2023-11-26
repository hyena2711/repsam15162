import argparse
import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
import monai
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from typing import Any, Iterable
from tqdm import tqdm
from sklearn.model_selection import KFold, train_test_split
from torch.utils.tensorboard import SummaryWriter

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

from utils.dataset import SamDataset
import reprogrammers

import datetime
now = datetime.datetime.now()
now = now.strftime("%Y-%m-%d-%H-%M-%S")


def dice_score(preds, targets):
    smooth = 1.0
    assert preds.size() == targets.size()

    iflat = preds.contiguous().view(-1)
    tflat = targets.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    dice = (2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)
    return dice

###################################################################
# ########################## iou loss #############################
###################################################################
class IOU(torch.nn.Module):
    def __init__(self):
        super(IOU, self).__init__()

    def _iou(self, pred, target):
        pred = torch.sigmoid(pred)
        inter = (pred * target).sum(dim=(2, 3))
        union = (pred + target).sum(dim=(2, 3)) - inter
        iou = 1 - (inter / union)

        return iou.mean()

    def forward(self, pred, target):
        return self._iou(pred, target)
    

def get_loss_fn(loss_name):
    if loss_name == "dice_ce":
        seg_loss =  monai.losses.DiceCELoss(
            sigmoid=True, squared_pred=True, reduction="mean",
            lambda_ce=0.2,lambda_dice=0.8
            
        )
        return seg_loss, None
    elif loss_name == "dice_focal":
        seg_loss = monai.losses.DiceFocalLoss(sigmoid=True, 
                                              squared_pred=True,
                                              lambda_dice=1,
                                              lambda_focal=20,
                                              
                                              reduction="mean")
        return seg_loss,None
    elif loss_name == "dice_ce_iou":
        seg_loss = monai.losses.DiceCELoss(
            sigmoid=True, squared_pred=True, reduction="mean",
            lambda_ce=0.2,lambda_dice=0.8
        )
        iou_loss = IOU()
        return seg_loss,iou_loss
    
    elif loss_name == "dice_focal_iou":
        seg_loss = monai.losses.DiceFocalLoss(sigmoid=True, 
                                              squared_pred=True,
                                              lambda_dice=1,
                                              lambda_focal=20,
                                              
                                              reduction="mean")
        iou_loss = IOU()
        return seg_loss,iou_loss
    elif loss_name == "bce_iou":
        seg_loss = torch.nn.BCEWithLogitsLoss()
        iou_loss = IOU()
        return seg_loss,iou_loss

def create_csv(mode,path,img_ext,lbl_ext,do_split=False,masks_dir='masks',images_dir='images'):
        if mode is not None:
            masks = glob.glob(f'{path}/{mode}/{masks_dir}/*.{lbl_ext}')
            images=glob.glob(f'{path}/{mode}/{images_dir}/*.{img_ext}')
        else:
            
            masks = glob.glob(f'{path}/{masks_dir}/*.{lbl_ext}')
            images=glob.glob(f'{path}/{images_dir}/*.{img_ext}')
        
        images = sorted(images)
        masks = sorted(masks)
        
        df = pd.DataFrame(
            {
                'image_path':images,
                'mask_path':masks
            }
            )
        if do_split:
            train_df, val_df = train_test_split(df, train_size=0.9, random_state=2023)
            return train_df,val_df
            
        return df

class RepSAM(nn.Module):

    def __init__(self,sam_model,device,method,pad_size,image_size):
        super().__init__()
        self.sam_model = sam_model
        self.device = device
        self.method = method
        
        
        self.reprogrammer = reprogrammers.__dict__[method](pad_size,image_size)

    def forward(self, x,bbox_tensor):
      
       
        prompted_x = self.reprogrammer(x)
            
        
        image_features = self.sam_model.image_encoder(prompted_x)
        sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
                                 points=None,
                                 boxes=bbox_tensor,
                                 masks=None)
        
        mask_predictions, _ = self.sam_model.mask_decoder(
            image_embeddings=image_features, # (B, 256, 64, 64)
            image_pe=self.sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
        )

        return mask_predictions
    

class Trainer:
    BEST_VAL_LOSS = float("inf")
    BEST_EPOCH = 0

    def __init__(
        self,
        lr: float = 10.0,
        batch_size: int = 1,
        epochs: int = 100,
        device: str = "cuda:0",
        model_type: str = "vit_b",
        image_dir="data/image_dir",
        mask_dir="data/image_dir",
        checkpoint: str = "work_dir/SAM/sam_vit_b_01ec64.pth",
        **kwargs,

    ):
        
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.mask_dir = mask_dir
        self.image_dir = image_dir
        self.sam_checkpoint_dir = checkpoint
        self.model_type = model_type
        self.use_bbox = kwargs['use_bbox']
        self.img_ext = kwargs['img_ext']
        self.lbl_ext = kwargs['lbl_ext']
        self.method = kwargs['method']
        self.pad_size = kwargs['pad_size']
        self.image_size = kwargs['image_size']
        self.loss_name = kwargs['loss_name']
        self.save_path = kwargs['save_path']
        self.num_gpus = kwargs['num_gpus']
        self.bbox_offset_min = kwargs['bbox_offset_min']
        self.bbox_offset_max = kwargs['bbox_offset_max']


        print("USE BBOX: ", self.use_bbox)
        if self.use_bbox:
            self.name = f'repsam_bbox_p{self.pad_size}'
        else:
            self.name = f'repsam_p{self.pad_size}'


        self.primary_loss,self.secondary_loss = get_loss_fn(self.loss_name)

        


    def __call__(self, train_df, test_df, val_df, image_col, mask_col):
        """Entry method
        prepare `dataset` and `dataloader` objects

        """
        train_ds = SamDataset(
            train_df,
            image_col,
            mask_col,
            self.image_dir,
            self.mask_dir,
            bbox_offset_min=self.bbox_offset_min,
            bbox_offset_max=self.bbox_offset_max
        )
        
        val_ds = SamDataset(
            val_df,
            image_col,
            mask_col,
            self.image_dir,
            self.mask_dir,
            bbox_offset_min=self.bbox_offset_min,
            bbox_offset_max=self.bbox_offset_max
        )
        test_ds = SamDataset(
            test_df,
            image_col,
            mask_col,
            self.image_dir,
            self.mask_dir,
            bbox_offset_min=self.bbox_offset_min,
            bbox_offset_max=self.bbox_offset_max
        )
        # Define dataloaders
        train_loader = DataLoader(
            dataset=train_ds, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            dataset=val_ds, batch_size=self.batch_size, shuffle=False
        )
        test_loader = DataLoader(
            dataset=test_ds, batch_size=self.batch_size, shuffle=False
        )
        # get the model
        self.get_model()

        # Train and evaluate model
        self.train(self.model, train_loader, val_loader)
        # Evaluate model on test data
        loss, dice_score = self.test(self.model, test_loader, desc="Testing")

        del self.model
        torch.cuda.empty_cache()

        self.BEST_EPOCH = 0
        self.BEST_VAL_LOSS = float("inf")

        return dice_score
    


    def cuda(self):
        # Set the device for the model and other components
        torch.cuda.set_device(self.num_gpus[0])
        self.model = self.model.cuda()
        self.primary_loss = self.primary_loss.cuda()
        if self.secondary_loss is not None:
            self.secondary_loss = self.secondary_loss.cuda()

        # Multi-GPU
        if len(self.num_gpus) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.num_gpus, output_device=self.num_gpus[0])

        return self

    def print_model(self,model):

       
        count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)

        mr_param = count_parameters(self.model)/1e6
        print(f'number of trainable parameters: {mr_param:.2f}M')
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add((name,param.shape))
        print(f"Parameters to be updated: {enabled}")
        self.enabled = [i for i,_ in enabled]
    
    def get_model(self):
        print(self.model_type,self.sam_checkpoint_dir)
        sam_model = sam_model_registry[self.model_type](
            checkpoint=self.sam_checkpoint_dir
        )
        if len(self.num_gpus)>1:
            self.device = None
        else:
            self.device = self.num_gpus[0]
        self.model = RepSAM(sam_model,self.device,self.method,self.pad_size,self.image_size)

        # cuda gpu
        
        for i in (self.num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

        self.cuda()
        self.gpu = True

        for name, param in self.model.named_parameters():
                if 'reprogrammer' not in name:
                    param.requires_grad = False
        self.print_model(self.model)
            

        # return self.model
    
    @torch.inference_mode()
    def evaluate(self, model, val_loader, desc="Validating") -> float:
        """Perform evaluation on test or validation data
        Args:
            model (_type_): pytorch model
            val_loader (_type_): dataloader
            desc (str, optional): _description_. Defaults to "Validating".

        Returns:
            np.array: (mean validation loss, mean validation dice)
        """
      
        progress_bar = tqdm(val_loader, total=len(val_loader))
        val_loss = []
        val_dice = []

        for image, mask, bbox in progress_bar:
            image = image.cuda()
            mask = mask.cuda()
            # resize image to 1024 by 1024
            image = TF.resize(image, (1024, 1024), antialias=True)
            H, W = mask.shape[-2], mask.shape[-1]
            B = image.shape[0]
            if len(self.num_gpus)>1:
                sam_trans = ResizeLongestSide(model.module.sam_model.image_encoder.img_size)
            else:
                sam_trans = ResizeLongestSide(model.sam_model.image_encoder.img_size)

            # box = sam_trans.apply_boxes(bbox, (H, W))
            # box_tensor = torch.as_tensor(box, dtype=torch.float, device=self.device)

            if self.use_bbox:
                    box = sam_trans.apply_boxes(bbox.numpy(), (H, W))
                    box_tensor = torch.as_tensor(box, dtype=torch.float).cuda()

            else:
                    boxes = torch.from_numpy(np.array([[0,0,W,H]]*B)).float().cuda()
                    box_tensor = boxes[:,None,:]
            mask_predictions  = model(image, box_tensor)

            primary_loss = self.primary_loss(mask_predictions, mask)
            if self.secondary_loss is not None:
                secondary_loss = self.secondary_loss(mask_predictions, mask)
                loss = primary_loss + secondary_loss
            else:
                loss = primary_loss

            # # loss = bce_loss + iou_loss
            mask_predictions = (mask_predictions > 0.5).float()
            dice = dice_score(mask_predictions, mask)

            val_loss.append(loss.detach().item())
            val_dice.append(dice.detach().item())

            # Update the progress bar
            progress_bar.set_description(desc)
            progress_bar.set_postfix(
                eval_loss=np.mean(val_loss), eval_dice=np.mean(val_dice)
            )
            progress_bar.update()
        return np.mean(val_loss), np.mean(val_dice)

    @torch.inference_mode()
    def test(self, model, test_loader, desc="Testing") -> float:
        """Perform evaluation on test or validation data
        Args:
            model (_type_): pytorch model
            val_loader (_type_): dataloader
            desc (str, optional): _description_.

        Returns:
            float: mean validation loss
        """


        
        progress_bar = tqdm(test_loader, total=len(test_loader))
        val_loss = []
        dice_scores = []

        for image, mask, bbox in progress_bar:
            image = image.cuda()
            mask = mask.cuda()
            # resize image to 1024 by 1024
            image = TF.resize(image, (1024, 1024), antialias=True)
            H, W = mask.shape[-2], mask.shape[-1]
            B = image.shape[0]
            # sam_trans = ResizeLongestSide(model.sam_model.image_encoder.img_size)
            if len(self.num_gpus)>1:
                sam_trans = ResizeLongestSide(model.module.sam_model.image_encoder.img_size)
            else:
                sam_trans = ResizeLongestSide(model.sam_model.image_encoder.img_size)

            if self.use_bbox:
                    box = sam_trans.apply_boxes(bbox.numpy(), (H, W))
                    box_tensor = torch.as_tensor(box, dtype=torch.float).cuda()

            else:
                    boxes = torch.from_numpy(np.array([[0,0,W,H]]*B)).float().cuda()
                    box_tensor = boxes[:,None,:]

            mask_predictions  = model(image, box_tensor)

            # get the dice loss
            # loss = seg_loss(mask_predictions, mask)
            primary_loss = self.primary_loss(mask_predictions, mask)
            if self.secondary_loss is not None:
                secondary_loss = self.secondary_loss(mask_predictions, mask)
                loss = primary_loss + secondary_loss
            else:
                loss = primary_loss

            mask_predictions = (mask_predictions > 0.5).float()
            dice = dice_score(mask_predictions, mask)

            val_loss.append(loss.item())
            dice_scores.append(dice.detach().item())

            # Update the progress bar
            progress_bar.set_description(desc)
            progress_bar.set_postfix(
                eval_loss=np.mean(val_loss), eval_dice=np.mean(dice_scores)
            )
            progress_bar.update()
        return np.mean(val_loss), np.mean(dice_scores)

    def train(self, model, train_loader: Iterable, val_loader: Iterable, logg=False):
        """Train the model"""

        # sam_trans = ResizeLongestSide(model.sam_model.image_encoder.img_size)
        if len(self.num_gpus)>1:
                sam_trans = ResizeLongestSide(model.module.sam_model.image_encoder.img_size)
        else:
                sam_trans = ResizeLongestSide(model.sam_model.image_encoder.img_size)

        writer = SummaryWriter()

        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=0.0)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=3, factor=0.1, verbose=True
        )
        # optimizer = optim.SGD(model.parameters(), lr=self.lr, weight_decay=0, momentum=0.9)


        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, float(self.epochs)
        # )
        # seg_loss = monai.losses.DiceCELoss(
        #     sigmoid=True, squared_pred=True, reduction="mean"
        # )

        model.train()
        for epoch in range(self.epochs):
            epoch_loss = []
            epoch_dice = []
            progress_bar = tqdm(train_loader, total=len(train_loader))
            for image, mask, bbox in progress_bar:
                image = image.cuda()
                mask = mask.cuda()
                # # resize image to 1024 by 1024
                image = TF.resize(image, (1024, 1024), antialias=True)
                H, W = mask.shape[-2], mask.shape[-1]
                B = image.shape[0]
                
                if self.use_bbox:
                
                    box = sam_trans.apply_boxes(bbox.numpy(), (H, W))
                    box_tensor = torch.as_tensor(box, dtype=torch.float).cuda()

                else:
                    boxes = torch.from_numpy(np.array([[0,0,W,H]]*B)).float().cuda()
                    box_tensor = boxes[:,None,:]

                mask_predictions  = model(image, box_tensor)
                
                # Calculate loss
                # loss = seg_loss(mask_predictions, mask)
                primary_loss = self.primary_loss(mask_predictions, mask)
                if self.secondary_loss is not None:
                    secondary_loss = self.secondary_loss(mask_predictions, mask)
                    loss = primary_loss + secondary_loss
                else:
                    loss = primary_loss

                mask_predictions = (mask_predictions > 0.5).float()
                dice = dice_score(mask_predictions, mask)

                epoch_loss.append(loss.detach().item())
                epoch_dice.append(dice.detach().item())

                # empty gradient
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # scheduler.step()
                progress_bar.set_description(f"Epoch {epoch+1}/{self.epochs}")
                progress_bar.set_postfix(
                    loss=np.mean(epoch_loss), dice=np.mean(epoch_dice)
                )
                progress_bar.update()
            # Evaluate every two epochs
            if epoch % 2 == 0:
                validation_loss, validation_dice = self.evaluate(
                    model, val_loader, desc=f"Validating"
                )
                
                scheduler.step(torch.tensor(validation_loss))

                if self.early_stopping(model, validation_loss, epoch):
                    print(f"[INFO:] Early Stopping!!")
                    break

            if logg:
                writer.add_scalars(
                    "loss",
                    {
                        "train": round(np.mean(epoch_loss), 4),
                        "val": round(validation_loss, 4),
                    },
                    epoch,
                )

                writer.add_scalars(
                    "dice",
                    {
                        "train": round(np.mean(epoch_dice), 4),
                        "val": round(validation_dice, 4),
                    },
                    epoch,
                )
        self.save_model(model,name=f"{self.name}_final")

    def save_model(self, model,name='mr_sam_latest'):
        
        model_name = f"{name}.pth"
        

        state_dict = {}
        for k,v in model.state_dict().items():
            if "reprogrammer" in k:
                state_dict[k] = v



        print(f"[INFO:] Saving model to {os.path.join(self.save_path,model_name)}")
        print(f"[INFO:] Saving Parameters: {state_dict.keys()}")

        torch.save(state_dict, os.path.join(self.save_path, model_name))
    def early_stopping(
        self,
        model,
        val_loss: float,
        epoch: int,
        patience: int = 50,
        min_delta: int = 0.0001,
    ):
        """Helper function for model training early stopping
        Args:
            val_loss (float): _description_
            epoch (int): _description_
            patience (int, optional): _description_. Defaults to 10.
            min_delta (int, optional): _description_. Defaults to 0.01.
        """

        if self.BEST_VAL_LOSS - val_loss >= min_delta:
            print(
                f"[INFO:] Validation loss improved from {self.BEST_VAL_LOSS} to {val_loss}"
            )
            self.BEST_VAL_LOSS = val_loss
            self.BEST_EPOCH = epoch
            self.save_model(model,self.name)
            return False

        if (
            self.BEST_VAL_LOSS - val_loss < min_delta
            and epoch - self.BEST_EPOCH >= patience
        ):
            return True
        return False


    
class CrossValidate(Trainer):
    def __init__(
        self,
        lr: float = 3e-4,
        batch_size: int = 4,
        epochs: int = 100,
        device: str = "cuda:0",
        model_type: str = "vit_b",
        image_dir="data/image_dir",
        mask_dir="data/image_dir",
        checkpoint: str = "work_dir/SAM/sam_vit_b_01ec64.pth",
    ):
        super().__init__(
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            device=device,
            model_type=model_type,
            image_dir=image_dir,
            mask_dir=mask_dir,
            checkpoint=checkpoint,
        )

    def __call__(self, train_df, test_df, image_col, mask_col, k: int = 5) -> Any:
        """Performs kfold cross validation
        Args:
            k (int, optional): Fold size. Defaults to 5.
        """
        # Define the cross-validation splitter
        kf = KFold(n_splits=k, shuffle=True)
        # loop over each fold
        fold_scores = {}
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
            print(f"Cross validating for fold {fold + 1}")
            # Define training and validation sets for this fold
            f_train_df = train_df.iloc[train_idx]
            f_val_df = train_df.iloc[val_idx]

            dice_score = super().__call__(
                f_train_df, test_df, f_val_df, image_col, mask_col
            )

            fold_scores[f"fold_{fold + 1}_mean_dice"] = dice_score

        return fold_scores
        

def main():
    # set up parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str, required=True, help="Path to the dataset root")

    parser.add_argument(
        "--image_col",
        type=str,
        default=None,
        help="Name of the column on the dataframe that holds the image file names",
    )

    parser.add_argument(
        "--mask_col",
        type=str,
        default=None,
        help="the name of the column on the dataframe that holds the mask file names",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help="image file extension",
    )

    parser.add_argument(
        "--img_ext",
        type=str,
        default=None,
        help="image file extension",
    )
    parser.add_argument(
        "--lbl_ext",
        type=str,
        default=None,
        help="mask file extension",
    )


    parser.add_argument(
        "--image", type=str, required=False, help="Path to the input image directory"
    )
    parser.add_argument(
        "--mask",
        type=str,
        required=False,
        help="Path to the ground truth mask directory",
    )
    parser.add_argument(
        "--num_epochs", type=int, required=False, default=100, help="number of epochs"
    )
    parser.add_argument(
        "--lr", type=float, required=False, default=100, help="learning rate"
    )
    parser.add_argument(
        "--batch_size", type=int, required=False, default=1, help="batch size"
    )
    parser.add_argument(
        "-k",
        type=int,
        default=None,
        required=False,
        help="Number of folds for cross validation",
    )
    parser.add_argument("--model_type", type=str, required="False", default="vit_b")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to SAM checkpoint"
    )
    parser.add_argument('--num_gpus', nargs="+", type=int, default=[0],
                         help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--label_id', type=int, default=None, help='label id')
   

    parser.add_argument('--method', type=str, required=True, help='prompting method')
    parser.add_argument('--loss_name', type=str, required=True, help='loss name')
    parser.add_argument('--dataset_name', type=str, required=True, help='dataset name')
    parser.add_argument('--use_bbox', action='store_true', help='Use bounding box for training')
    parser.add_argument('--pad_size', type=int, required=True, help='padding size')
    parser.add_argument('--output_dir', type=str, required=True, help='output directory')
    parser.add_argument('--bbox_offset_min', type=int, default=0, help='bbox offset min')
    parser.add_argument('--bbox_offset_max', type=int, default=10, help='bbox offset max')

    args = parser.parse_args()
  
    # args.experiment_name = f'{args.model_type}_{args.dataset_name}_p{args.pad_size}_{args.loss_name}_{args.method}_lr{args.lr}'
    # save_path = f"{args.experiment_name}"
    
    output_dir = f"{args.output_dir}_{args.method}_{args.loss_name}"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir,exist_ok=True)
    if not os.path.exists(f'{output_dir}/{args.dataset_name}'):
        os.makedirs(f'{output_dir}/{args.dataset_name}',exist_ok=True)
    if not os.path.exists(f'{output_dir}/{args.dataset_name}/ep_{args.num_epochs}_lr{args.lr}'):
        os.makedirs(f'{output_dir}/{args.dataset_name}/ep_{args.num_epochs}_lr{args.lr}',exist_ok=True)

    args.save_path = f'{output_dir}/{args.dataset_name}/ep_{args.num_epochs}_lr{args.lr}'
    
    from pprint import pprint
    pprint(vars(args))
   
    
        

    if args.mode is not None:

        if args.label_id is not None:
            mode_train = f'train_{args.label_id}'
            mode_test = f'test_{args.label_id}'
        else:
            mode_train = 'train'
            mode_test = 'test'

        

        train_df,val_df = create_csv(mode_train,
                                    args.path,
                                    args.img_ext,
                                    args.lbl_ext,
                                    do_split=True)
        
        test_df = create_csv(mode_test,
                            args.path,
                                args.img_ext,
                                args.lbl_ext,
                                do_split=False)
    else:
        if 'ISIC' in args.path:
            print('ISIC DATASET')
            train_df = create_csv(None,
                                    args.path,
                                    args.img_ext,
                                    args.lbl_ext,
                                    do_split=False,
                                    masks_dir='ISIC-2017_Training_Part1_GroundTruth',
                                    images_dir='ISIC-2017_Training_Data')
            
            val_df = create_csv(None,
                                args.path,
                                args.img_ext,
                                args.lbl_ext,
                                do_split=False,
                                masks_dir='ISIC-2017_Validation_Part1_GroundTruth',
                                images_dir='ISIC-2017_Validation_Data')
            
            test_df = create_csv(None,
                                args.path,
                                args.img_ext,
                                args.lbl_ext,
                                do_split=False,
                                masks_dir='ISIC-2017_Test_v2_Part1_GroundTruth',
                                images_dir='ISIC-2017_Test_v2_Data')
        
        else:

            train_df,val_df = create_csv(args.mode,
                                        args.path,
                                        args.img_ext,
                                        args.lbl_ext,
                                        do_split=True)
            test_split=False
            if not test_split:
                train_df,test_df = train_test_split(train_df, train_size=0.8, random_state=2023)

            else:
            
                test_df = create_csv(args.mode,
                                    f'{args.path}/CAMO',
                                        args.img_ext,
                                        args.lbl_ext,
                                        do_split=False)
            
        
    
        
    
    print('-'*50)
    print(f"Train: {len(train_df)}")
    print(f"Val: {len(val_df)}")
    print(f"Test: {len(test_df)}")
    print('-'*50)

   
    train_df.to_csv(f'{args.save_path}/train.csv',index=False)
    val_df.to_csv(f'{args.save_path}/val.csv',index=False)
    test_df.to_csv(f'{args.save_path}/test.csv',index=False)

    

    #save argumets as json
    with open(f'{args.save_path}/args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

    
    if args.k:
        print(f"[INFO] Starting {args.k} fold cross validation ....")
        if args.k < 5:
            raise ValueError("K should be a value greater than or equal to 5")

        cross_validate = CrossValidate(
            lr=args.lr,
            batch_size=args.batch_size,
            epochs=args.num_epochs,
            image_dir=args.image,
            mask_dir=args.mask,
            checkpoint=args.checkpoint,
        )
        scores = cross_validate(
            train_df, val_df, args.image_col, args.mask_col, args.k
        )

        # write cross-validation scores to file
        with open("mrsam.json", "w") as f:
            json.dump(scores, f)

    # if `k` is not specified, normal training mode
    if not args.k:
        print(f"[INFO] Starting training for {args.num_epochs} epochs ....")
       

        trainer= Trainer(
            lr=args.lr,
            batch_size=args.batch_size,
            epochs=args.num_epochs,
            image_dir=args.image,
            mask_dir=args.mask,
            checkpoint=args.checkpoint,
            model_type=args.model_type,
            use_bbox=args.use_bbox,
            img_ext=args.img_ext,
            lbl_ext=args.lbl_ext,
            method=args.method,
            pad_size=args.pad_size,
            image_size=1024,
            loss_name=args.loss_name,
            save_path=args.save_path,
            num_gpus=args.num_gpus,
            bbox_offset_min=args.bbox_offset_min,
            bbox_offset_max=args.bbox_offset_max
            
            )


        

        trainer(train_df, test_df, val_df, args.image_col, args.mask_col)


if __name__ == "__main__":
    main()

