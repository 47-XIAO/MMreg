import os, glob
import torch, sys
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pickle
import numpy as np
import SimpleITK as sitk
import random
import nibabel as nib
import torch.nn.functional as F


class CTMRI_SR_REG_Dataset(Dataset):
    def __init__(self, data_path, transforms):
        # self.paths = data_path
        self.transforms = transforms
        self.ct_mri_pairs = self._generate_ct_mri_pairs(data_path)

    def _generate_ct_mri_pairs(self, data_path, ct_suffix="_ct", mri_suffix="_mr"):
        prefix_to_ct = {}  
        prefix_to_mri = {}  
        all_ct_files = []  
        all_mri_files = []  
        
        for filename in os.listdir(data_path):
            file_full_path = os.path.join(data_path, filename)
            if not os.path.isfile(file_full_path):
                continue  

            name_without_ext = filename
            while os.path.splitext(name_without_ext)[1]:
                name_without_ext = os.path.splitext(name_without_ext)[0]
            
            # 判断模态并提取前缀
            if name_without_ext.endswith(ct_suffix):
                prefix = name_without_ext[:-len(ct_suffix)]
                modality = "CT"
            elif name_without_ext.endswith(mri_suffix):
                prefix = name_without_ext[:-len(mri_suffix)]
                modality = "MRI"
            
            if prefix is None or modality is None:
                continue 
            
            # 分类存储
            if modality == "CT":
                prefix_to_ct[prefix] = file_full_path
                all_ct_files.append(file_full_path)
            elif modality == "MRI":
                prefix_to_mri[prefix] = file_full_path
                all_mri_files.append(file_full_path)
        

        valid_prefixes = set(prefix_to_ct.keys()) & set(prefix_to_mri.keys())
        if not valid_prefixes:
            raise ValueError("No matching original CT MRI pair with the same filename was found")
        print(f"Found {len(valid_prefixes)} matching CT MRI pair")
        
        ct_path_to_mate_mri = {}  
        mri_path_to_mate_ct = {}  
        seg_path = {}  
        for prefix in valid_prefixes:
            ct_path = prefix_to_ct[prefix]
            mri_path = prefix_to_mri[prefix]
            ct_path_to_mate_mri[ct_path] = mri_path
            mri_path_to_mate_ct[mri_path] = ct_path
            seg_name = f"{prefix}.nii.gz"
            seg_path[ct_path] = os.path.join(data_path, seg_name)
            seg_path[mri_path] = os.path.join(data_path, seg_name)

        
        non_mate_pairs = []
        for ct_path in all_ct_files:
            mate_mri_path = ct_path_to_mate_mri.get(ct_path)  
            for mri_path in all_mri_files:
                if mri_path != mate_mri_path:  # 排除自身
                    non_mate_pairs.append((ct_path, mri_path, seg_path[ct_path], seg_path[mri_path]))

        if len(non_mate_pairs) == 0:
            raise ValueError("No matching original CT MRI pair with the same filename was found")
        print(f"Get {len(non_mate_pairs)} CT MRI pairs")
        
        return non_mate_pairs


    def __getitem__(self, index):
        # path = self.paths[index]
        ct_path, mri_path,ct_seg_path,mri_seg_path = self.ct_mri_pairs[index]
        # tar_list = self.paths.copy()
        # tar_list.remove(path)
        # random.shuffle(tar_list)
        # tar_file = tar_list[0]
        x = nib.load(ct_path)
        x_seg = nib.load(ct_seg_path.replace("volumes", "segs"))

        y = nib.load(mri_path)
        y_seg = nib.load(mri_seg_path.replace("volumes", "segs"))

        # y_data_file = os.path.dirname(path.replace("CT", "MRI"))
        # y_data_path = y_data_file + '/' + random.choice(os.listdir(y_data_file))
        x = x.get_fdata()  
        y = y.get_fdata()  
        x_seg = x_seg.get_fdata()  
        y_seg = y_seg.get_fdata()  
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)

        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.ct_mri_pairs)

class CTMRIABDDataset(Dataset):
    def __init__(self, data_path, transforms):
        # self.paths = data_path
        self.transforms = transforms
        self.ct_mri_pairs = self._generate_ct_mri_pairs(data_path)

    def pad(self, x, target_size=(160, 160, 128)):

        is_batch = len(x.shape) == 5
        if is_batch:
            B, C, D, H, W = x.shape
            x = x.view(C, D, H, W) 
        
        C, D, H, W = x.shape
        tD, tH, tW = target_size
        pad_d = max(0, tD - x.shape[1])
        pad_h = max(0, tH - x.shape[2])
        pad_w = max(0, tW - x.shape[3])
        pad = (pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2, pad_d//2, pad_d - pad_d//2)
        x = F.pad(x, pad, mode='constant', value=0)

        if is_batch:
            x = x.view(B, C, tD, tH, tW) 
        return x

    def _generate_ct_mri_pairs(self, data_path):
        ct_mri_pairs = []
        ct_folder = os.path.join(data_path, "CT_resample")
        ct_files = os.listdir(ct_folder)
        mri_folder = os.path.join(data_path, "MRI_resample")
        for ct_path in ct_files:
            mri_files = os.listdir(mri_folder)
            for mri_path in mri_files:
                ct_full_path = os.path.join(ct_folder, ct_path) 
                mri_full_path = os.path.join(mri_folder, mri_path) 
                ct_mri_pairs.append((ct_full_path, mri_full_path))
        if len(ct_mri_pairs) == 0:
            raise ValueError("No matching original CT MRI pair with the same filename was found")
        return ct_mri_pairs

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        # path = self.paths[index]
        ct_path, mri_path = self.ct_mri_pairs[index]
        # tar_list = self.paths.copy()
        # tar_list.remove(path)
        # random.shuffle(tar_list)
        # tar_file = tar_list[0]
        x = nib.load(ct_path)
        x_seg = nib.load(ct_path.replace("images", "labels"))

        y = nib.load(mri_path)
        y_seg = nib.load(mri_path.replace("images", "labels"))

        # y_data_file = os.path.dirname(path.replace("CT", "MRI"))
        # y_data_path = y_data_file + '/' + random.choice(os.listdir(y_data_file))
        x = x.get_fdata()  
        y = y.get_fdata()  
        x_seg = x_seg.get_fdata()  
        y_seg = y_seg.get_fdata()  
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        x = self.pad(x)
        y = self.pad(y)
        x_seg = self.pad(x_seg)
        y_seg = self.pad(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.ct_mri_pairs)


class CTMRIABDInferDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms
        self.ct_mri_pairs = self._generate_ct_mri_pairs()

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out
    
    def pad(self, x, target_size=(160, 160, 128)):

        is_batch = len(x.shape) == 5
        if is_batch:
            B, C, D, H, W = x.shape
            x = x.view(C, D, H, W) 

        C, D, H, W = x.shape
        tD, tH, tW = target_size
        pad_d = max(0, tD - x.shape[1])
        pad_h = max(0, tH - x.shape[2])
        pad_w = max(0, tW - x.shape[3])
        pad = (pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2, pad_d//2, pad_d - pad_d//2)
        x = F.pad(x, pad, mode='constant', value=0)

        if is_batch:
            x = x.view(B, C, tD, tH, tW)  
        return x

    def _generate_ct_mri_pairs(self):
        """
        CT: AbdomenMRCT_0002_0001.nii.gz
        MRI: AbdomenMRCT_0002_0000.nii.gz
        """
        ct_folder = os.path.join(self.paths, "CT_resample/")
        mri_folder = os.path.join(self.paths, "MRI_resample/")

        ct_files = [
            os.path.join(ct_folder, f)
            for f in os.listdir(ct_folder)
            if f.endswith((".nii", ".nii.gz")) and "_0001" in f  
        ]
        ct_files.sort()  
        if not ct_files:
            raise ValueError(f"{ct_folder}don't find valid files")
        

        ct_mri_pairs = []
        for ct_path in ct_files:
            ct_filename = os.path.basename(ct_path)
            mri_filename = ct_filename.replace("_0001", "_0000")
            mri_path = os.path.join(mri_folder, mri_filename)

            ct_mri_pairs.append((ct_path, mri_path))

        return ct_mri_pairs

    def __getitem__(self, index):
        ct_path, mri_path = self.ct_mri_pairs[index]
        x = nib.load(ct_path)
        x_seg = nib.load(ct_path.replace("images", "labels"))
        
        y = nib.load(mri_path)
        y_seg = nib.load(mri_path.replace("images", "labels"))
        x = x.get_fdata()  
        y = y.get_fdata()  
        x_seg = x_seg.get_fdata()  
        y_seg = y_seg.get_fdata()  
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        x = self.pad(x)
        y = self.pad(y)
        x_seg = self.pad(x_seg)
        y_seg = self.pad(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.ct_mri_pairs)
    
class CTMRI_AMOS_Dataset(Dataset):
    def __init__(self, data_path, transforms):
        # self.paths = data_path
        self.transforms = transforms
        self.ct_mri_pairs = self._generate_ct_mri_pairs(data_path)

    def _generate_ct_mri_pairs(self, data_path):
        ct_mri_pairs = []
        ct_folder = os.path.join(data_path, "CT_resample")
        ct_files = os.listdir(ct_folder)
        mri_folder = os.path.join(data_path, "MRI_resample")
        for ct_path in ct_files:
            mri_files = os.listdir(mri_folder)
            for mri_path in mri_files:
                ct_full_path = os.path.join(ct_folder, ct_path) 
                mri_full_path = os.path.join(mri_folder, mri_path) 
                ct_mri_pairs.append((ct_full_path, mri_full_path))
        if len(ct_mri_pairs) == 0:
            raise ValueError("No matching original CT MRI pair with the same filename was found")
        return ct_mri_pairs

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        # path = self.paths[index]
        ct_path, mri_path = self.ct_mri_pairs[index]
        x = nib.load(ct_path)
        x_seg = nib.load(ct_path.replace("images", "labels"))

        y = nib.load(mri_path)
        y_seg = nib.load(mri_path.replace("images", "labels"))

        x = x.get_fdata()  
        y = y.get_fdata()  
        x_seg = x_seg.get_fdata()  
        y_seg = y_seg.get_fdata()  
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.ct_mri_pairs)
    

data_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(data_root, os.path.pardir))

def ToTensor(img):
    return torch.from_numpy(img).float()


def ToTensor1(img):
    return torch.from_numpy((img - img.min()) / (img.max() - img.min())).float()

def norm_mr(img):
    mean = np.mean(img)
    std = np.std(img)
    return (img - mean) / (std + 1e-8) 

def norm(img):
    return (img - img.min()) / (img.max() - img.min())

