"""生成模型输入数据."""

import argparse
import glob
import os

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from multiprocessing import Pool

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='./train_data/origin_data')
    parser.add_argument('--tgt_path', type=str, default='./train_data/processed_data')
    args = parser.parse_args()
    return args


def load_scans(dcm_path):
    reader = sitk.ImageSeriesReader()
    name = reader.GetGDCMSeriesFileNames(dcm_path)
    reader.SetFileNames(name)
    img = reader.Execute()
    return img


def gen_lst(save_path, task, all_pids):
    save_file = os.path.join(save_path, task+'.txt')
    data_list = glob.glob(os.path.join(save_path, '*.npz'))
    num = 0
    with open(save_file, 'w') as f:
        for pid in all_pids:
            data = os.path.join(save_path, pid+".npz")
            if data in data_list:
                num+=1
                f.writelines(data.replace("\\","/") + '\n')
    print('num of data: ', num)


def mask2point(mask, n):
    points = np.zeros_like(mask)
    for i in range(1,n+1): 
        loc = np.array(list(zip(*np.where(mask==i))))
        loc_center = np.mean(loc, 0)
        loc_center = tuple((loc_center + 0.5).astype(np.int64))
        points[loc_center] = i
    return points


def process_single(input):
    mask_path, dcm_path, tgt_path, pid = input
    img_itk = sitk.ReadImage(dcm_path)
    mask_itk = sitk.ReadImage(mask_path)
    if img_itk.GetSize() == mask_itk.GetSize():
        img = sitk.GetArrayFromImage(img_itk)
        mask = sitk.GetArrayFromImage(mask_itk)
        mask = mask2point(mask, 5).astype('uint8')
        np.savez_compressed(os.path.join(tgt_path, f'{pid}.npz'), img=img, mask=mask)



if __name__ == '__main__':
    args = parse_args()
    src_path = args.src_path
    for task in ["train", "valid"]:
        print("\nBegin gen %s data!"%(task))
        src_dicom_path = os.path.join(args.src_path, task, "dcm_nii")
        src_mask_path = os.path.join(args.src_path, task, "mask_nii")
        tgt_path = args.tgt_path
        os.makedirs(tgt_path, exist_ok=True)
        inputs = []
        for pid in tqdm(os.listdir(src_mask_path)):
            mask_path = os.path.join(src_mask_path, pid)           
            pid = pid.replace('.mask.nii.gz', '')
            dcm_path = os.path.join(src_dicom_path, pid+".nii.gz")
            inputs.append([mask_path, dcm_path, tgt_path, pid])
        processed_pids = [pid.replace('.mask.nii.gz', '') for pid in os.listdir(src_mask_path)]
        pool = Pool(8)
        pool.map(process_single, inputs)
        pool.close()
        pool.join()
        # 生成Dataset所需的数据列表
        gen_lst(tgt_path, task, processed_pids)
