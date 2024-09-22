import os
import numpy as np
from tqdm import tqdm
from skimage.measure import regionprops
from skimage.morphology import ball, dilation
import zarr

def create_pseudo_nuclei(root, experiment_date, cp_model_name, well_num, overwrite_flag=True, center_rad=3):

    file_prefix = experiment_date + f"_well{well_num:04}"
    # get path to full masks
    mask_zarr_path = os.path.join(root, "built_data", "stitched_labels", cp_model_name, experiment_date,
                                  file_prefix + "_labels_stitched.zarr")
    mask_zarr = zarr.open(mask_zarr_path, mode='r')
    # if out_shape is None:
    #     out_shape = mask_zarr.shape
    # create new zarr to store nuclei
    out_zarr_path = os.path.join(root, "built_data", "centroid_labels",   cp_model_name, experiment_date,
                                 file_prefix + "_centroids.zarr")
    dtype = mask_zarr.dtype

    if (not os.path.exists(out_zarr_path)) | overwrite_flag:
        out_zarr = zarr.open(out_zarr_path, mode='w', shape=mask_zarr.shape, dtype=dtype,
                             chunks=(1,) + mask_zarr.shape[1:])
    else:
        out_zarr = zarr.open(out_zarr_path, mode='a', chunks=(1,) + mask_zarr.shape[1:])

    fp = ball(center_rad)
    # Load and resize
    print("Loading time points...")
    for m in tqdm(range(mask_zarr.shape[0])):

        lb_flag = np.any(out_zarr[m] > 0)

        if (not lb_flag) | overwrite_flag:
            frame_arr = np.asarray(mask_zarr[m])
            regions = regionprops(frame_arr)
            new_mask_array = np.zeros(frame_arr.shape, dtype=dtype)
            for region in regions:
                centroid = np.asarray(region.centroid).astype(int)
                new_mask_array[centroid[0], centroid[1], centroid[2]] = region.label

            new_mask = dilation(new_mask_array, fp)
            out_zarr[m] = new_mask

    return {}


if __name__ == '__main__':

    experiment_date = "20240223"
    root = "E:/Nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/fin_morphodynamics/"
    overwrite_flag = True
    well_num = 12
    segmentation_model = "log-v5"

    create_pseudo_nuclei(root, experiment_date, well_num=well_num, cp_model_name=segmentation_model, overwrite_flag=False)