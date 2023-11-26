from glob import glob
import torch
import argparse
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import monai

from utils.dataset import SamDataset
from utils import metrics
import reprogrammers
from sklearn.model_selection import KFold, train_test_split

import torch
from torch.utils.data import DataLoader
from segment_anything import sam_model_registry,SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide

import glob
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from functools import partial
from tqdm import tqdm
import torch.nn as nn 

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_points(bboxes,device='cpu',npoints=1):
    random_coords=True
    coords_list = []
    labels_list = []
    
    for i in range(len(bboxes)):
        bbox = bboxes[i].cpu().numpy()

        x0, y0 = bbox[0], bbox[1]
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        cx, cy = x0 + w // 2, y0 + h // 2
        coords_per_box = []
        labels_per_box = []
        for n in range(npoints):
            x = np.random.uniform(low=max(cx,x0+10), high=min(cx,w-10))
            y = np.random.uniform(low=max(cy,y0+10), high=min(cy,h-10))
            # x = np.random.uniform(low=x0+w//2, high=min(x0+w,x0+w-(2**n)))
            # y = np.random.uniform(low=y0+h//2, high=min(y0+h,y0+h-(2**n)))


            coords_per_box.append([x,y])
            labels_per_box.append(1)

        # print(bbox)
        # print(cx,cy)
        point_coords = np.array(coords_per_box)
        point_labels = np.array(labels_per_box)

        # point_coords = sam_trans.apply_coords(point_coords, (H, W))
        _coords_torch = torch.as_tensor(
            point_coords, dtype=torch.float, device=device
        )
        _labels_torch = torch.as_tensor(
            point_labels, dtype=torch.int, device=device
        )
        # _coords_torch, _labels_torch = _coords_torch[None, :, :], _labels_torch[None, :]
       
        coords_list.append(_coords_torch)
        labels_list.append(_labels_torch)
    coords_torch = torch.stack(coords_list, dim=0)
    labels_torch = torch.stack(labels_list, dim=0)
    points = (coords_torch,labels_torch)
    return points

class RepSAM(nn.Module):

    def __init__(self,sam_model,device,method,pad_size,image_size,**kwargs):
        super().__init__()
        self.sam_model = sam_model
        self.device = device
        self.npoints = kwargs.get('npoints',None)
        
        
        self.reprogrammer = reprogrammers.__dict__[method](pad_size,image_size)
        self.method = method

    def forward(self, x,bbox_tensor):
         
        prompted_x = self.reprogrammer(x)
       
        image_features = self.sam_model.image_encoder(prompted_x)
        if self.npoints is None:
            points = None
        else:
            points = get_points(bbox_tensor,device=self.device,npoints=self.npoints)
            bbox_tensor = None
        sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
                                 points=points,
                                 boxes=bbox_tensor,
                                 masks=None)
        
        mask_predictions, _ = self.sam_model.mask_decoder(
            image_embeddings=image_features.to(self.device), # (B, 256, 64, 64)
            image_pe=self.sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
        )
        return mask_predictions
    


def create_csv(path,img_ext,lbl_ext,masks='masks',images='images'):
        
        if 'OOD_Seg' in path:
            masks = glob.glob(f'{path}/{masks}/*.{lbl_ext}')
            images=glob.glob(f'{path}/{images}/*.{img_ext}')
        else:
            masks = glob.glob(f'{path}/test/{masks}/*.{lbl_ext}')
            images=glob.glob(f'{path}/test/{images}/*.{img_ext}')
        print(len(images),len(masks))
        images = sorted(images)
        masks = sorted(masks)
        
        df = pd.DataFrame(
            {
                'image_path':images,
                'mask_path':masks
            }
            )
            
        return df


def dice_coefficient(preds, targets):
  smooth = 1.0
  assert preds.size() == targets.size()

  iflat = preds.contiguous().view(-1)
  tflat = targets.contiguous().view(-1)
  intersection = (iflat * tflat).sum()
  dice = (2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)
  return dice



def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.5])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2)) 

@torch.inference_mode()
def get_sam_prediction(sam_model,image,bbox,full_bbox=False):
    sam_model.eval()
    sam_trans = ResizeLongestSide(sam_model.image_encoder.img_size)
    #trans_image = image.unsqueeze(0).to(device)
    trans_image = sam_trans.apply_image_torch(image)
    if bbox is not None:
        box = sam_trans.apply_boxes(bbox, (256, 256))
        box_tensor = torch.as_tensor(box, dtype=torch.float, device=device)
        if full_bbox:
            boxes = torch.from_numpy(bbox).float().to(device)
            box_tensor = boxes[:,None,:]
    else:
        box_tensor = None
    # Get predictioin mask
    
    image_embeddings = sam_model.image_encoder(trans_image)  # (B,256,64,64)

    sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
        points=None,
        boxes=box_tensor, #box_tensor,
        masks=None,
    )


    mask_predictions, _ = sam_model.mask_decoder(
        image_embeddings=image_embeddings.to(device),  # (B, 256, 64, 64)
        image_pe=sam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )
    mask_predictions = torch.sigmoid(mask_predictions)

    return (mask_predictions > 0.5).int()

@torch.inference_mode()
def get_repsam_prediction(model,image,bbox,full_bbox=False):
    model.eval()

    sam_trans = ResizeLongestSide(model.sam_model.image_encoder.img_size)
    #trans_image = image.unsqueeze(0).to(device)
    trans_image = sam_trans.apply_image_torch(image)
    if bbox is not None:
        box = sam_trans.apply_boxes(bbox, (256, 256))
        box_tensor = torch.as_tensor(box, dtype=torch.float, device=device)
        if full_bbox:
            boxes = torch.from_numpy(bbox).float().to(device)
            box_tensor = boxes[:,None,:]
    else:
        
        box_tensor = None
    # Get predictioin mask
    
    # image_embeddings = sam_model.image_encoder(trans_image)  # (B,256,64,64)
    mask_predictions  = model(trans_image, box_tensor)

    mask_predictions = torch.sigmoid(mask_predictions)

    return (mask_predictions > 0.5).int()



def main():
    # set up parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str, required=True, help="Path to the dataset root")
    parser.add_argument("--df", type=str, default=None, help="Test Dataframe")

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
        "--batch_size", type=int, required=False, default=4, help="batch size"
    )
    
    parser.add_argument("--model_type", type=str, required="False", default="vit_b")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to SAM checkpoint"
    )
    parser.add_argument('--mrsam', type=str,default=None, help='Load MR-SAM')
    parser.add_argument('--method', type=str,default=None, help='Load Prompt')
    parser.add_argument('--pad_size', type=int,default=128, help='Load Prompt')
    parser.add_argument('--decoder_ckpt', type=str,default=None, help='Load Prompt')
    parser.add_argument('--use_bbox', action='store_true', help='Load Prompt')
    parser.add_argument('--npoints', type=int,default=None, help='Load Points')
    parser.add_argument('--bbox_offset_min', type=int, default=0, help='bbox offset min')
    parser.add_argument('--bbox_offset_max', type=int, default=10, help='bbox offset max')

    args = parser.parse_args()
    from pprint import pprint
    pprint(vars(args))

    if args.df is not None:
        print('Loading from dataframe',args.df)
        df = pd.read_csv(args.df)
    else:

        df = create_csv(args.path,args.img_ext,args.lbl_ext)

    

    dataset = SamDataset(df,image_col='image_path',mask_col='mask_path',bbox_offset_min=0,bbox_offset_max=10)
    batch_size = args.batch_size
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=False)

    sam_model  = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(device)
    sam_model.eval()
    # args.mrsam = None
    predictor = get_repsam_prediction if args.mrsam is not None else get_sam_prediction



    if args.mrsam is not None:
         
        repsam = RepSAM(sam_model,device,args.method,pad_size=args.pad_size,image_size=1024,npoints=args.npoints).to(device)
        #parameters in mrsam reprogrammer
        count_parameters = lambda model: sum(p.numel() for p in model.parameters())

        params = count_parameters(repsam.reprogrammer)
        print(f'Params {params/1e6}M')
        print('Before loading')
        for k,v in repsam.state_dict().items():
            if 'reprogrammer' in k:
                print(k,v.mean().item())

        print('After loading')

        ckpt = torch.load(args.mrsam,map_location=device)
        print(ckpt.keys())
        wdict = {}
        for k,v in repsam.state_dict().items():
            if 'reprogrammer' in k:
                # k1 = k.replace('reprogrammer','prompter')
                
                k1 = 'module.'+k
                wdict[k] = ckpt[k1]
                
            else:
                wdict[k] = v

        repsam.load_state_dict(wdict)

        for k,v in repsam.state_dict().items():
            if 'reprogrammer' in k:
                print(k,v.mean().item())

        sam_model = repsam.eval()
    elif args.decoder_ckpt is not None:
        print('Loading decoder')
        ckpt = torch.load(args.decoder_ckpt,map_location=device)
        wdict = {}
        for k,v in sam_model.state_dict().items():
            if 'mask_decoder' in k:
                # k1 = k.replace('reprogrammer','prompter')
                
                k1 = 'module.'+k
                wdict[k] = ckpt[k1]
                
                
            else:
                wdict[k] = v
        

        sam_model.load_state_dict(wdict)
        sam_model = sam_model.eval()
        

    dice_repsam=[]
    iou_metric = monai.metrics.MeanIoU(include_background=False, reduction="mean")
    
    for images,masks,bboxes in tqdm(dataloader):
        images = images.to(device)
        masks = masks.to(device)
        bboxes = bboxes.numpy()
        H, W = masks.shape[-2], masks.shape[-1]
        B = images.shape[0]

        if not args.use_bbox:
            bboxes = np.array([[0,0,W,H]]*B)
            fullbox = True
        else:
            fullbox = False
        
        prediction = predictor(sam_model,images,bboxes,fullbox)
        
        # dice_metric(prediction,masks)
        dice_repsam.append(dice_coefficient(prediction,masks).item())
        

        iou_metric(prediction,masks)

    print(f"mDice = {np.array(dice_repsam).mean():.2f}")
    print(f"mIoU = {iou_metric.aggregate().item():.2f}")
           




   
  

if __name__ == "__main__":
    main()
