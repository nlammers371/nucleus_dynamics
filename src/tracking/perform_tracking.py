import napari
import os
import numpy as np
from tqdm import tqdm
from skimage.transform import resize
from ultrack.imgproc.intensity import robust_invert
from ultrack.imgproc.segmentation import detect_foreground
import time
from ultrack.utils.array import array_apply
from ultrack import MainConfig, load_config, track, to_tracks_layer, tracks_to_zarr
import nd2
import zarr
from ultrack.utils import estimate_parameters_from_labels, labels_to_edges
import glob2 as glob
from skimage.segmentation import watershed

# get path to raw nd2 file


def perform_tracking(root, experiment_date, well_num, tracking_config, scale_vec, seg_model, start_i=0, stop_i=None,
                     overwrite_flag=False, use_centroids=False, add_label_spacers=False):

    # imObject = nd2.ND2File(nd2_path)
    # res_raw = imObject.voxel_size()
    # scale_vec = np.asarray(res_raw)[::-1]

    tracking_name = tracking_config.replace(".txt", "")

    # get path to zarr file
    file_prefix = experiment_date + f"_well{well_num:04}"

    # set parameters
    # data_zarr = os.path.join(root, "built_data", "zarr_image_files", experiment_date, file_prefix + ".zarr")
    if not use_centroids:
        mask_zarr_path = os.path.join(root, "built_data", "stitched_labels", seg_model, experiment_date, file_prefix + "_labels_stitched.zarr")
        suffix = ""
    else:
        mask_zarr_path = os.path.join(root, "built_data", "centroid_labels", seg_model, experiment_date,
                                      file_prefix + "_centroids.zarr")
        suffix = "_centroid"

    if add_label_spacers:
        suffix += "_spacer"

    # data_tzyx_full = zarr.open(data_zarr, mode='r')
    mask_tzyx_full = zarr.open(mask_zarr_path, mode='r')
    if stop_i is None:
        stop_i = mask_tzyx_full.shape[0]

    # get path to metadata
    metadata_path = os.path.join(root, "metadata", "tracking")

    # set output path for tracking results
    project_path = os.path.join(root, "tracking", experiment_date,  tracking_name, f"well{well_num:04}" + suffix, "")
    if not os.path.isdir(project_path):
        os.makedirs(project_path)

    # specify time points to load
    # data_tzyx = data_tzyx_full[start_i:stop_i, :, :, :]
    if well_num == 12:
        mask_tzyx = mask_tzyx_full[start_i:stop_i, :, 400:850, 80:520]
    elif well_num == 2:
        mask_tzyx = mask_tzyx_full[start_i:stop_i, :, 300:700, 100:500]
    else:
        mask_tzyx = mask_tzyx_full

    # load tracking config file
    cfg = load_config(os.path.join(metadata_path, tracking_config))
    cfg.data_config.working_dir = project_path
    if use_centroids:
        cfg.segmentation_config.min_area = 90
        cfg.segmentation_config.max_area = 200

    # get tracking inputs
    detection, boundaries = labels_to_edges(mask_tzyx)

    if add_label_spacers: # seperate labeld components
        mask_tzyx_eroded = mask_tzyx.copy()
        for t in range(mask_tzyx_eroded.shape[0]):
            mask_temp = mask_tzyx_eroded[t]
            mask_temp[boundaries[t] > 0] = 0
            mask_tzyx_eroded[t] = mask_temp

        # cellpose_directory = os.path.join(root, "built_data", "cellpose_output", seg_model, experiment_date, '')
        # prob_name = os.path.join(cellpose_directory, file_prefix + "_probs.zarr")
        #
        # # mask_zarr = zarr.open(mask_name, mode="r")
        # prob_zarr = zarr.open(prob_name, mode="r")[start_i:stop_i, :, 400:850, 80:520]
        #
        # wt_array = watershed(image=-prob_zarr[0], markers=mask_tzyx_eroded, mask=mask_tzyx[0]>0, watershed_line=False)
        detection, boundaries = labels_to_edges(mask_tzyx_eroded)

    # Perform tracking
    print("Performing tracking...")
    track(
        cfg,
        detection=detection,
        edges=boundaries,
        scale=scale_vec,
        overwrite=True,
    )

    print("Saving results...")
    tracks_df, graph = to_tracks_layer(cfg)
    tracks_df.to_csv(project_path + "tracks.csv", index=False)

    segments = tracks_to_zarr(
        cfg,
        tracks_df,
        store_or_path=project_path + "segments.zarr",
        overwrite=True,
    )
    print("Done.")

    return segments


if __name__ == '__main__':

    scale_vec = np.asarray([2.0, 0.55, 0.55])
    experiment_date = "20240223"
    root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/fin_morphodynamics/"
    overwrite_flag = True
    well_num = 2
    use_centroids = False
    tracking_config = "tracking_jordao_frontier.txt"
    segmentation_model = "log-v5"
    add_label_spacer = False
    perform_tracking(root, experiment_date, well_num, tracking_config, scale_vec=scale_vec,
                     seg_model=segmentation_model, start_i=0, stop_i=101, overwrite_flag=overwrite_flag,
                     use_centroids=use_centroids, add_label_spacers=add_label_spacer)
# print("Saving downsampled image data...")
# np.save(project_path + "image_data_ds.npy", data_tzyx)