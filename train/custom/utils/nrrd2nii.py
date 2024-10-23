import os
import SimpleITK as sitk
from tqdm import tqdm

if __name__ == '__main__':
    srcDir = r'F:\Code\Knee_Auto_Localization\Knee_Auto_Localization\train\train_data\data'
    for pid in tqdm(os.listdir(srcDir)):
        for file in os.listdir(os.path.join(srcDir, pid)):
            if "local3d_Knee_Center" in file: 
                volume_nrrd_file = os.path.join(srcDir, pid, file)
                volume = sitk.ReadImage(volume_nrrd_file)
                sitk.WriteImage(volume,  volume_nrrd_file.replace(file, '{}.nii.gz'.format(pid)))

            if "Segmentation" in file: 
                mask_nrrd_file = os.path.join(srcDir, pid, file)
                mask = sitk.ReadImage(mask_nrrd_file)
                sitk.WriteImage(mask,  mask_nrrd_file.replace(file, '{}.mask.nii.gz'.format(pid)))
