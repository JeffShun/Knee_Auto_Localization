import os
import SimpleITK as sitk

if __name__ == '__main__':
    srcDir = r'F:\Code\Knee_Auto_Localization\fancyviewer3D\data2'
    for pid in os.listdir(srcDir):
        volume_nrrd_file = os.path.join(srcDir, pid, "{}0000.nrrd".format(pid))
        volume = sitk.ReadImage(volume_nrrd_file)
        sitk.WriteImage(volume,  volume_nrrd_file.replace('{}0000.nrrd'.format(pid), '{}.nii.gz'.format(pid)))
    
        mask_nrrd_file = os.path.join(srcDir, pid, "Segmentation.seg.nrrd")
        mask = sitk.ReadImage(mask_nrrd_file)
        sitk.WriteImage(mask,  mask_nrrd_file.replace('Segmentation.seg.nrrd', '{}.mask.nii.gz'.format(pid)))
