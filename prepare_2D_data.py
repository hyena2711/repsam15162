import numpy as np
import SimpleITK as sitk
import os
join = os.path.join 
from skimage import transform, io, segmentation
from tqdm import tqdm
import torch
from PIL import Image
import json

from sklearn.model_selection import train_test_split
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
def get_file_list(dataset):
    set_seed(2023)
    names = dataset['training']
    np.random.shuffle(names)
    
    train_names, test_names = train_test_split(names, test_size=0.2, random_state=2023)
    
    
    # train_names,test_names  = names[:int(len(names)*0.8)], names[int(len(names)*0.8):]
    print('Train: ', len(train_names))
    print('Test: ', len(test_names))
    


    return train_names, test_names

def preprocess_ct(root, gt_name, image_name, label_id, image_size):
        gt_sitk = sitk.ReadImage(join(root, gt_name))
        gt_data = sitk.GetArrayFromImage(gt_sitk)
        gt_data = np.uint8(gt_data==label_id)
        imgs = []
        gts =  []
        if np.sum(gt_data)>1000:
            
            img_embeddings = []
            assert np.max(gt_data)==1 and np.unique(gt_data).shape[0]==2, 'ground truth should be binary'
            img_sitk = sitk.ReadImage(join(root, image_name))
            image_data = sitk.GetArrayFromImage(img_sitk)
            # print(image_data.shape, gt_data.shape)
            # nii preprocess start
            lower_bound = -500
            upper_bound = 1000
            image_data_pre = np.clip(image_data, lower_bound, upper_bound)
            image_data_pre = (image_data_pre - np.min(image_data_pre))/(np.max(image_data_pre)-np.min(image_data_pre))*255.0
            image_data_pre[image_data==0] = 0
            image_data_pre = np.uint8(image_data_pre)
            
            z_index, _, _ = np.where(gt_data>0)
            z_min, z_max = np.min(z_index), np.max(z_index)
            # print('----',z_min, z_max,gt_data.shape,image_data_pre.shape,image_data.shape)
            for i in range(z_min, z_max):
                gt_slice_i = gt_data[i,:,:]
                # print(gt_slice_i.shape,image_data_pre.shape)
                gt_slice_i = transform.resize(gt_slice_i, (image_size, image_size), order=0, preserve_range=True, mode='constant', anti_aliasing=False)
                if np.sum(gt_slice_i)>100:
                    img_slice_i = transform.resize(image_data_pre[i,:,:], (image_size, image_size), order=3, preserve_range=True, mode='constant', anti_aliasing=True)
                   
                    # convert to three channels
                    img_slice_i = np.uint8(np.repeat(img_slice_i[:,:,None], 3, axis=-1))
                    assert len(img_slice_i.shape)==3 and img_slice_i.shape[2]==3, 'image should be 3 channels'
                    assert img_slice_i.shape[0]==gt_slice_i.shape[0] and img_slice_i.shape[1]==gt_slice_i.shape[1], 'image and ground truth should have the same size'
                    imgs.append(img_slice_i)
                    assert np.sum(gt_slice_i)>100, 'ground truth should have more than 100 pixels'
                    gts.append(gt_slice_i)
                    
       
        return imgs, gts

def preprocess_nonct(root, gt_name, image_name, label_id, image_size):
    
    gt_sitk = sitk.ReadImage(join(root, gt_name))
    gt_data = sitk.GetArrayFromImage(gt_sitk)
    gt_data = np.uint8(gt_data==label_id)
    imgs = []
    gts =  []
    if np.sum(gt_data)>1000:
        
        img_embeddings = []
        assert np.max(gt_data)==1 and np.unique(gt_data).shape[0]==2, 'ground truth should be binary'
        img_sitk = sitk.ReadImage(join(root, image_name))
        image_data = sitk.GetArrayFromImage(img_sitk)
        # nii preprocess start
        lower_bound, upper_bound = np.percentile(image_data, 0.5), np.percentile(image_data, 99.5)
        image_data_pre = np.clip(image_data, lower_bound, upper_bound)
        image_data_pre = (image_data_pre - np.min(image_data_pre))/(np.max(image_data_pre)-np.min(image_data_pre))*255.0
        image_data_pre[image_data==0] = 0
        image_data_pre = np.uint8(image_data_pre)
        
        z_index, _, _ = np.where(gt_data>0)
        z_min, z_max = np.min(z_index), np.max(z_index)
       
        if image_data_pre.ndim==4:
            image_data_pre = image_data_pre[0,:,:,:]
        
        for i in range(z_min, z_max):
            gt_slice_i = gt_data[i,:,:]
            
            gt_slice_i = transform.resize(gt_slice_i, (image_size, image_size), order=0, preserve_range=True, mode='constant', anti_aliasing=True)
            if np.sum(gt_slice_i)>100:
                # resize img_slice_i to 256x256
                img_slice_i = transform.resize(image_data_pre[i,:,:], (image_size, image_size), order=3, preserve_range=True, mode='constant', anti_aliasing=True)
                # convert to three channels
                img_slice_i = np.uint8(np.repeat(img_slice_i[:,:,None], 3, axis=-1))
                assert len(img_slice_i.shape)==3 and img_slice_i.shape[2]==3, 'image should be 3 channels'
                assert img_slice_i.shape[0]==gt_slice_i.shape[0] and img_slice_i.shape[1]==gt_slice_i.shape[1], 'image and ground truth should have the same size'
                imgs.append(img_slice_i)
                assert np.sum(gt_slice_i)>100, 'ground truth should have more than 100 pixels'
                gts.append(gt_slice_i)
                


    return imgs, gts


def save_files(root, files, label_id, image_size,preprocess_func=None):
    for mode in files.keys():
        c = 0
        print('Processing {} data'.format(mode))
        for name in tqdm(files[mode]):
        
            gt_name = name['label']
            image_name = name['image']
            imgs, gts = preprocess_func(root, gt_name, image_name, label_id, image_size)
            
            if not os.path.exists(os.path.join(root,mode)):
                os.makedirs(os.path.join(root,mode))
            if not os.path.exists(os.path.join(root,mode,'images')):
                os.makedirs(os.path.join(root,mode,'images'))
            if not os.path.exists(os.path.join(root,mode,'masks')):
                os.makedirs(os.path.join(root,mode,'masks'))
            if len(imgs)==0:
                continue
            imgs = np.stack(imgs,axis=0)
            gts = np.stack(gts,axis=0)
            # print(imgs.shape, gts.shape)
            

            for i in range(len(imgs)):
                img_idx = imgs[i,:,:,:]
                gt_idx = gts[i,:,:]
                # bd = segmentation.find_boundaries(gt_idx, mode='inner')
                # img_idx[bd, :] = [255, 0, 0]
                io.imsave(os.path.join(root,mode,'images',root.split('/')[-1].lower()+'_'+str(c)+'.png'), img_idx, check_contrast=False)
                io.imsave(os.path.join(root,mode,'masks',root.split('/')[-1].lower()+'_'+str(c)+'.png'), gt_idx, check_contrast=False)
                c+=1
                
            
        print(len(os.listdir(os.path.join(root,mode,'images'))))
        print(len(os.listdir(os.path.join(root,mode,'masks'))))


if __name__ == "__main__":

    root= '/data/datasets/sam_data/Task01_BrainTumour'
    # root = '/data/datasets/sam_data/Task02_Heart'
    # root= '/data/datasets/sam_data/Task04_Hippocampus'
    # root= '/data/datasets/sam_data/Task05_Prostate'
    # root = '/data/datasets/sam_data/Task06_Lung'
    # root = '/data/datasets/sam_data/Task09_Spleen'
    # root = '/data/datasets/sam_data/Task10_Colon'
    # root = '/data/datasets/sam_data/xray_hipjoints'
    import json

    json_path = os.path.join(root, 'dataset.json')
    with open (json_path,'rb') as f:
                dataset = json.load(f)
    print(dataset['modality'],'CT' in dataset['modality'].values())
    print(dataset.keys())
    if 'CT' in dataset['modality'].values():
        preprocess_func = preprocess_ct
        print('CT data')
    else:
        preprocess_func = preprocess_nonct
        print('Non CT data')
    for label_id in dataset['labels'].keys():
        label_id = int(label_id)
        if label_id==0:
            continue
        train_names,  test_names= get_file_list(dataset)
        files = {f'train_{label_id}': train_names,f'test_{label_id}': test_names}
        image_size = 256    
        save_files(root, files, label_id, image_size,preprocess_func=preprocess_func)

    
    
   

# train_dataset = MedicalDecathlonDataset('/data/datasets/sam_data/Task09_Spleen', train_names, image_size=256, label_id=label_id)