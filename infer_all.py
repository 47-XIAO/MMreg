import glob
import os, losses, utils
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms

from natsort import natsorted
# from models_rdp import RDP
from model import PRC as network
import random
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
#正确

def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


same_seeds(24)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0
        self.vals.append(val)
        self.std = np.std(self.vals) if len(self.vals) > 0 else 0.0


def calculate_assd_hd95(seg1, seg2, voxel_spacing=None):
    """
    计算ASSD(平均对称表面距离)和HD95(95%豪斯多夫距离)——二值 mask 输入
    """
    try:
        from medpy.metric.binary import assd, hd95
        if voxel_spacing is None:
            voxel_spacing = [1.0, 1.0, 1.0]

        seg1 = seg1.astype(bool)
        seg2 = seg2.astype(bool)

        if np.sum(seg1) == 0 or np.sum(seg2) == 0:
            return float('nan'), float('nan')

        a = assd(seg1, seg2, voxelspacing=voxel_spacing)
        h = hd95(seg1, seg2, voxelspacing=voxel_spacing)
        return a, h

    except ImportError:
        print("medpy not installed. pip install medpy")
        return float('nan'), float('nan')
    except Exception as e:
        print("ASSD/HD95 error:", e)
        return float('nan'), float('nan')


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # val_dir = '/hy-tmp/xsq/AbdomenMRCT_exp/imagesTs_norm/'
    # val_dir = '/hy-tmp/xsq/amos22_exp/val/images'
    val_dir = '/hy-tmp/xsq/SR-Reg/test/volumes_center_norm_crop'
    weights = [1, 1]
    lr = 0.0001
    model_idx = -1
    model_folder = 'prwhc_1_reg_1_lr_0.0001_sr_30e/'
    model_dir = 'RDP-main2/experiments_sr/' + model_folder

    # img_size = (160, 192, 160)
    # img_size = (160, 160, 128)
    # img_size = (208, 144, 128)
    img_size = (144,176,160)
    # num_rois = 56  # labels 1..56

    # --- load model
    model = network(img_size, channels=16)
    ckpt_list = natsorted(os.listdir(model_dir))
    best_model_path = os.path.join(model_dir, ckpt_list[model_idx])
    ckpt = torch.load(best_model_path)
    best_model = ckpt.get('state_dict', ckpt)  # robust
    print('Best model:', best_model_path)
    model.load_state_dict(best_model)
    model.cuda()
    model.eval()

    reg_model = utils.register_model(img_size, 'nearest')
    reg_model.cuda()
    reg_model.eval()

    # test_composed = transforms.Compose([
    #     trans.Seg_norm(),
    #     trans.NumpyType((np.float32, np.int16)),
    # ])
    # test_set = datasets.LPBABrainInferDatasetS2S(glob.glob(val_dir + '*.pkl'), transforms=test_composed)
    # test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
    test_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16))])
    # test_dataset = datasets.CTMRIABDInferDataset(val_dir, transforms=test_composed)
    test_dataset = datasets.CTMRI_SR_REG_Dataset(val_dir, transforms=test_composed)
    # test_dataset = datasets.CTMRI_AMOS_Dataset(val_dir, transforms=test_composed)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, num_workers=4)

    # metrics
    eval_dsc_def = AverageMeter()
    eval_dsc_raw = AverageMeter()
    eval_det = AverageMeter()
    eval_neg_jac_percentage = AverageMeter()
    eval_assd_def = AverageMeter()
    eval_assd_raw = AverageMeter()
    eval_hd95_def = AverageMeter()
    eval_hd95_raw = AverageMeter()

    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            # move to GPU
            data = [t.cuda() for t in data]
            x = data[0]        # moving image
            y = data[1]        # fixed image
            x_seg = data[2]    # moving seg (labels)
            y_seg = data[3]    # fixed seg (labels)

            seg_normalizer = trans.Seg_norm()
            x_seg = seg_normalizer(x_seg.cpu().numpy())
            x_seg = torch.from_numpy(x_seg).long().to(device)
            y_seg = seg_normalizer(y_seg.cpu().numpy())
            y_seg = torch.from_numpy(y_seg).long().to(device)

            # forward
            x_def, flow,_,_,_,_,_,_,_,_ = model(x, y)  # x_def unused for metrics; flow used to warp seg
            # warp segmentation with reg_model
            # reg_model expects [seg, flow] per earlier usage
            def_out = reg_model([x_seg.float(), flow])  # may be float due to interpolation

            # convert to numpy label maps
            # def_out might be interpolated floats → round to nearest int to get labels
            def_out_np = np.rint(def_out.detach().cpu().numpy()[0, 0]).astype(np.int32)
            x_seg_np = x_seg.detach().cpu().numpy()[0, 0].astype(np.int32)
            y_seg_np = y_seg.detach().cpu().numpy()[0, 0].astype(np.int32)

            # jacobian determinant (expect utils.jacobian_determinant_vxm returns numpy array)
            try:
                jac_det = utils.jacobian_determinant_vxm(flow.detach().cpu().numpy()[0])
                fold_count = np.sum(jac_det <= 0)
                neg_jac_percentage = (fold_count / float(np.prod(jac_det.shape))) * 100.0
            except Exception as e:
                print(f"Jacobian error at idx {idx}: {e}")
                fold_count = 0.0
                neg_jac_percentage = 0.0

            eval_det.update(float(fold_count), 1)
            eval_neg_jac_percentage.update(float(neg_jac_percentage), 1)

            # Dice: we compute per-label Dice and average across present labels (per-sample mean)
            dscs_def = []
            dscs_raw = []
            assd_def_list = []
            assd_raw_list = []
            hd_def_list = []
            hd_raw_list = []

            for label in range(1, 19):
                gt_mask = (y_seg_np == label).astype(np.uint8)
                if gt_mask.sum() == 0:
                    continue  # label not present in this sample

                pred_def_mask = (def_out_np == label).astype(np.uint8)
                pred_raw_mask = (x_seg_np == label).astype(np.uint8)

                # Dice
                inter_def = np.sum(pred_def_mask & gt_mask)
                denom_def = np.sum(pred_def_mask) + np.sum(gt_mask)
                dsc_def = 2.0 * inter_def / denom_def if denom_def > 0 else 0.0

                inter_raw = np.sum(pred_raw_mask & gt_mask)
                denom_raw = np.sum(pred_raw_mask) + np.sum(gt_mask)
                dsc_raw = 2.0 * inter_raw / denom_raw if denom_raw > 0 else 0.0

                dscs_def.append(dsc_def)
                dscs_raw.append(dsc_raw)

                # ASSD / HD95 (only if both non-empty)
                if pred_def_mask.sum() > 0:
                    a_def, h_def = calculate_assd_hd95(pred_def_mask, gt_mask)
                    if not np.isnan(a_def):
                        assd_def_list.append(a_def)
                        hd_def_list.append(h_def)
                if pred_raw_mask.sum() > 0:
                    a_raw, h_raw = calculate_assd_hd95(pred_raw_mask, gt_mask)
                    if not np.isnan(a_raw):
                        assd_raw_list.append(a_raw)
                        hd_raw_list.append(h_raw)

            # update per-sample averages (only if lists non-empty)
            if len(dscs_def) > 0:
                eval_dsc_def.update(float(np.mean(dscs_def)), 1)
            if len(dscs_raw) > 0:
                eval_dsc_raw.update(float(np.mean(dscs_raw)), 1)
            if len(assd_def_list) > 0:
                eval_assd_def.update(float(np.mean(assd_def_list)), 1)
            if len(assd_raw_list) > 0:
                eval_assd_raw.update(float(np.mean(assd_raw_list)), 1)
            if len(hd_def_list) > 0:
                eval_hd95_def.update(float(np.mean(hd_def_list)), 1)
            if len(hd_raw_list) > 0:
                eval_hd95_raw.update(float(np.mean(hd_raw_list)), 1)

            # optional per-sample print
            if idx % 5 == 0:
                print(f"[{idx}] sample dsc_def={eval_dsc_def.val:.4f}, dsc_raw={eval_dsc_raw.val:.4f}, neg_jac%={neg_jac_percentage:.4f}")

        # final prints
        print('\n====== Final Results ======')
        print('Deformed DSC: {:.3f} ± {:.3f}'.format(eval_dsc_def.avg, eval_dsc_def.std))
        print('Affine DSC: {:.3f} ± {:.3f}'.format(eval_dsc_raw.avg, eval_dsc_raw.std))
        print('Negative Jacobian Percentage: {:.6f}% ± {:.6f}%'.format(eval_neg_jac_percentage.avg, eval_neg_jac_percentage.std))
        print('Fold count: {:.1f} ± {:.1f}'.format(eval_det.avg, eval_det.std))

        if eval_assd_def.count > 0:
            print('Deformed ASSD: {:.3f} ± {:.3f}'.format(eval_assd_def.avg, eval_assd_def.std))
            print('Deformed HD95: {:.3f} ± {:.3f}'.format(eval_hd95_def.avg, eval_hd95_def.std))
            print('Raw ASSD: {:.3f} ± {:.3f}'.format(eval_assd_raw.avg, eval_assd_raw.std))
            print('Raw HD95: {:.3f} ± {:.3f}'.format(eval_hd95_raw.avg, eval_hd95_raw.std))



if __name__ == '__main__':
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        print('     GPU #' + str(GPU_idx) + ': ' + torch.cuda.get_device_name(GPU_idx))
    torch.cuda.set_device(GPU_iden)
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    main()
