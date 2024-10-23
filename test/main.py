import argparse
import os
import sys
import shutil
import numpy as np
from tqdm import tqdm
from skimage.morphology import dilation
import SimpleITK as sitk
from scipy.ndimage import rotate
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predictor import Predictor

def parse_args():
    parser = argparse.ArgumentParser(description='Test Knee Auto Localization')

    parser.add_argument('--device', default="cuda:0", type=str)
    parser.add_argument('--input_path', default='../test/data/input/testdata', type=str)
    parser.add_argument('--output_path', default='../test/data/output/temp/ResUNET+Refine+AUG+DirectionLoss', type=str)

    parser.add_argument(
        '--model_file',
        type=str,
        # default='../train/checkpoints/trt_model/model.engine'
        default='../train/checkpoints/ResUNET+Refine+AUG+DirectionLoss/170.pth' 
    )
    parser.add_argument(
        '--config_file',
        type=str,
        default='./test_config.yaml'
    )
    args = parser.parse_args()
    return args


def load_scans(dcm_path):
    reader = sitk.ImageSeriesReader()
    name = reader.GetGDCMSeriesFileNames(dcm_path)
    reader.SetFileNames(name)
    sitk_img = reader.Execute()
    return sitk_img


def mask2point(mask, n):
    points = []
    for i in range(1,n+1): 
        loc = np.array(list(zip(*np.where(mask==i))))
        loc_center = np.mean(loc, 0)
        loc_center = list((loc_center + 0.5).astype(np.int64))
        points.append(loc_center)
    return np.array(points)


def main(input_path, output_path, device, args):
    # TODO: 适配参数输入
    predictor = Predictor(
        device=device,
        model_f=args.model_file,
        config_f=args.config_file
        )

    dcm_dir = os.path.join(input_path, "dcms")
    lbl_dir = os.path.join(input_path, "labels")

    for pid in tqdm(os.listdir(dcm_dir)):
        if pid != "14900.nii.gz":
            continue
        try:
            pid_path = os.path.join(dcm_dir, pid)
            if pid.endswith(".nii.gz"):
                sitk_img = sitk.ReadImage(pid_path)
            else: 
                sitk_img = load_scans(pid_path)

            pid = pid.replace(".nii.gz","")
            volume = sitk.GetArrayFromImage(sitk_img).astype('float32') 
            pred_array, heatmap, points = predictor.predict(volume)
            keypoint_itk = sitk.GetImageFromArray(pred_array)
            keypoint_itk.CopyInformation(sitk_img)

            # heatmap_itk = sitk.GetImageFromArray(heatmap[2].astype("float32"))
            # heatmap_itk.CopyInformation(sitk_img)
            # sitk.WriteImage(heatmap_itk, os.path.join(output_path, pid, f'{pid}.heatmap.nii.gz'))
            # shift_itk = sitk.GetImageFromArray(shift[2].astype("float32"))
            # shift_itk.CopyInformation(sitk_img)
            # sitk.WriteImage(shift_itk, os.path.join(output_path, pid, f'{pid}.shift.nii.gz'))

            os.makedirs(os.path.join(output_path, pid), exist_ok=True)
            sitk.WriteImage(sitk_img, os.path.join(output_path, pid, f'{pid}.nii.gz'))
            sitk.WriteImage(keypoint_itk, os.path.join(output_path, pid, f'{pid}.kp.nii.gz'))

            lbl_path = os.path.join(lbl_dir, f"{pid}.mask.nii.gz")
            if os.path.exists(lbl_path):
                data_for_metrics = os.path.join(output_path, "data_for_metrics")
                os.makedirs(data_for_metrics, exist_ok=True)
                lbl_arr = sitk.GetArrayFromImage(sitk.ReadImage(lbl_path))
                lbl_points = mask2point(lbl_arr, 5)
                lbl_mask = np.zeros(lbl_arr.shape, dtype="uint8")
                for i in range(lbl_points.shape[0]):
                    mask_i = np.zeros(lbl_arr.shape, dtype="uint8")
                    mask_i[tuple(points[i])] = 1
                    mask_dilate = dilation(mask_i, np.ones([2, 4, 4]))
                    lbl_mask[mask_dilate==1] = i+1
                lbl_itk = sitk.GetImageFromArray(lbl_mask)
                lbl_itk.CopyInformation(sitk_img)
                sitk.WriteImage(lbl_itk, os.path.join(output_path, pid, f'{pid}.mask.nii.gz'))

                spacing=np.array(sitk_img.GetSpacing())
                origin=np.array(sitk_img.GetOrigin())
                pred = points[:,::-1] * spacing + origin
                label = lbl_points[:,::-1] * spacing + origin
                imgsize = np.array(lbl_arr.shape)[::-1] * spacing
                np.savez_compressed(os.path.join(data_for_metrics, f'{pid}.npz'), pred=pred, label=label, imgsize=imgsize)
        except Exception as e:
            print(pid+" predict failed! ")
            print(e)

def main2(input_path, output_path, device, args):
    # TODO: 适配参数输入
    predictor = Predictor(
        device=device,
        model_f=args.model_file,
        config_f=args.config_file
        )

    dcm_dir = os.path.join(input_path, "consistency2")

    for pid in tqdm(os.listdir(dcm_dir)):
        try:
            pid_path = os.path.join(dcm_dir, pid)
            if pid.endswith(".nii.gz"):
                sitk_img = sitk.ReadImage(pid_path)
            else: 
                sitk_img = load_scans(pid_path)

            pid = pid.replace(".nii.gz","")
            temp_output_path = output_path + "/" + pid
            os.makedirs(temp_output_path, exist_ok=True)

            volume = sitk.GetArrayFromImage(sitk_img).astype('float32') 
            test_angles = [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50]
            for angle in test_angles:
                rotated_volume = rotate(volume, angle=angle, axes=(1, 2), reshape=False)
                rotated_sitk_img = sitk.GetImageFromArray(rotated_volume)
                rotated_sitk_img.CopyInformation(sitk_img)

                pred_array, heatmap, points = predictor.predict(rotated_volume)

                keypoint_itk = sitk.GetImageFromArray(pred_array)
                keypoint_itk.CopyInformation(sitk_img)

                os.makedirs(os.path.join(temp_output_path, str(angle)), exist_ok=True)
                sitk.WriteImage(rotated_sitk_img, os.path.join(temp_output_path, str(angle), f'{angle}.nii.gz'))
                sitk.WriteImage(keypoint_itk, os.path.join(temp_output_path, str(angle), f'{angle}.kp.nii.gz'))

        except Exception as e:
            print(pid+" predict failed! ")
            print(e)

if __name__ == '__main__':
    args = parse_args()
    main(
        input_path=args.input_path,
        output_path=args.output_path,
        device=args.device,
        args=args,
    )