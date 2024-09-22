import numpy as np
import scipy.ndimage as ndi
import zarr
from skimage.registration import phase_cross_correlation
from tqdm import tqdm
import os
import glob2 as glob
from src.utilities.functions import path_leaf

def register_timelapse(video: np.ndarray, mask_thresh=None) -> np.ndarray:

    # registration using channel 1
    shift_array = np.zeros((video.shape[0], 3))
    for t in tqdm(range(video.shape[0] - 1), "registering images..."):

        if mask_thresh is None:
            shift, error, _ = phase_cross_correlation(
                video[t].astype(np.float32),
                video[t + 1].astype(np.float32),
                normalization=None,
                upsample_factor=4,
                overlap_ratio=0.25,
            )

        else:
            shift, error, _ = phase_cross_correlation(
                video[t].astype(np.float32),
                video[t + 1].astype(np.float32),
                normalization=None,
                reference_mask=video[t]>=mask_thresh,
                moving_mask=video[t+1]>=mask_thresh,
                upsample_factor=4,
                overlap_ratio=0.25,
            )

        shift_array[t + 1, :] = shift
        # if out_array is not None:
        video[t+1] = ndi.shift(video[t+1], (shift), order=1)

    return video, shift_array

def registration_wrapper(root, experiment_date, model_name,register_masks=True,  scale_vec=None, overwrite=False):

    if scale_vec is None:
        scale_vec = np.asarray([2.0, 0.55, 0.55])

    metadata_dir = os.path.join(root, "metadata", "")
    # path to zarr files
    data_directory = os.path.join(root, "built_data", "zarr_image_files", experiment_date, '')
    reg_data_directory = os.path.join(root, "built_data", "zarr_image_files_registered", experiment_date, '')
    if not os.path.isdir(reg_data_directory):
        os.makedirs(reg_data_directory)

    cellpose_directory = os.path.join(root, "built_data", "cellpose_output", model_name, experiment_date, '')
    stitched_directory = os.path.join(root, "built_data", "stitched_labels", model_name, experiment_date, '')
    # get list of images
    image_list = sorted(glob.glob(data_directory + "*.zarr"))
    register_list = [i for i in range(len(image_list)) if i not in [2, 12]]

    for well_index in register_list:

        # prob_zarr = zarr.open(prob_name, mode="r")
        zarr_path = image_list[well_index]
        im_name = path_leaf(zarr_path)
        print("processing " + im_name)
        # read the image data
        data_zarr = zarr.open(zarr_path, mode="a")

        # generate zarr files
        file_prefix = experiment_date + f"_well{well_index:04}"

        # frame_vec = np.arange(0, data_zarr.shape[0])
        saved_frames = sorted(glob.glob(os.path.join(zarr_path, "*")))
        frame_nums = []
        for f in saved_frames:
            filename = path_leaf(f)
            ind = filename.find(".")
            if ind > 0:
                frame_nums.append(int(filename[:ind]))

        frame_nums = sorted(frame_nums)
        last_frame = np.max(frame_nums)

        data_zarr = data_zarr[:last_frame]
        # register dataset
        registered_data, shift_array = register_timelapse(data_zarr)
        data_zarr[:last_frame] = registered_data

        # save shift array
        np.save(os.path.join(metadata_dir, "registration", experiment_date + "_shift_array.npy"), shift_array)

        # apply shifts to the mask datasets
        if register_masks:
            # look for raw cellpose outputs first
            prob_name = os.path.join(cellpose_directory, file_prefix + "_probs.zarr")
            mask_name = os.path.join(cellpose_directory, file_prefix + "_labels.zarr")
            grad_name = os.path.join(cellpose_directory, file_prefix + "_grads.zarr")
            if os.path.isdir(prob_name):
                prob_zarr = zarr.open(prob_name, mode="a")
                mask_zarr = zarr.open(mask_name, mode="a")
                grad_zarr = zarr.open(grad_name, mode="a")
                for t in tqdm(range(0, last_frame-1), "Registering CellPose output..."):
                    prob_zarr[t+1] = ndi.shift(prob_zarr[t+1], (shift_array[t+1, :]), order=1)
                    mask_zarr[t + 1] = ndi.shift(mask_zarr[t + 1], (shift_array[t + 1, :]), order=0)
                    grad_zarr[t + 1] = ndi.shift(grad_zarr[t + 1], (0,) + tuple(shift_array[t + 1, :]), order=1)

            # now check for stitched labels
            stitch_name = os.path.join(stitched_directory, file_prefix + "_labels_stitched.zarr")
            if os.path.isdir(prob_name):
                stitch_zarr = zarr.open(stitch_name, mode="a")
                for t in tqdm(range(0, last_frame-1), "Registering stitched labels..."):
                    stitch_zarr[t + 1] = ndi.shift(stitch_zarr[t + 1], (shift_array[t + 1, :]), order=0)


if __name__ == "__main__":
    overwrite = False

    # set read/write paths
    root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/fin_morphodynamics/"
    experiment_date = "20240223"
    model_name = "log-v5"

    registration_wrapper(root=root, experiment_date=experiment_date, model_name=model_name,
                         register_masks=False, overwrite=overwrite)