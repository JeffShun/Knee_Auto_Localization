import argparse
import os
import sys
from tqdm import tqdm
import SimpleITK as sitk
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predictor import Predictor

def parse_args():
    parser = argparse.ArgumentParser(description='Test Knee Auto Localization')

    parser.add_argument('--device', default="cuda:0", type=str)
    parser.add_argument('--input_path', default='../test/data/input', type=str)
    parser.add_argument('--output_path', default='../test/data/output', type=str)

    parser.add_argument(
        '--model_file',
        type=str,
        # default='../train/checkpoints/trt_model/model.engine'
        default='../train/checkpoints/v1/100.pth' 
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


def main(input_path, output_path, device, args):
    # TODO: 适配参数输入
    predictor = Predictor(
        device=device,
        model_f=args.model_file,
        config_f=args.config_file
        )

    os.makedirs(output_path, exist_ok=True)
    for pid in tqdm(os.listdir(input_path)):
        try:
            pid_path = os.path.join(input_path, pid)
            if pid.endswith(".nii.gz"):
                sitk_img = sitk.ReadImage(pid_path)
            else: 
                sitk_img = load_scans(pid_path)

            pid = pid.replace(".nii.gz","")
            volume = sitk.GetArrayFromImage(sitk_img).astype('float32') 
            pred_array, heatmap = predictor.predict(volume)
            keypoint_itk = sitk.GetImageFromArray(pred_array)
            keypoint_itk.CopyInformation(sitk_img)

            # heatmap_itk = sitk.GetImageFromArray(heatmap[4].astype("float32"))
            # heatmap_itk.CopyInformation(sitk_img)

            os.makedirs(os.path.join(output_path, pid), exist_ok=True)
            sitk.WriteImage(sitk_img, os.path.join(output_path, pid, f'{pid}.nii.gz'))
            sitk.WriteImage(keypoint_itk, os.path.join(output_path, pid, f'{pid}.mask.nii.gz'))
            # sitk.WriteImage(heatmap_itk, os.path.join(output_path, pid, f'{pid}.heatmap.nii.gz'))
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