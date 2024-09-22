import napari
import os
import numpy as np
from tqdm import tqdm
import scipy.ndimage as ndi
from skimage.transform import resize
from ultrack.imgproc.intensity import robust_invert
from ultrack.imgproc.segmentation import detect_foreground
import time
from ultrack.utils.array import array_apply
from ultrack import MainConfig, load_config, track, to_tracks_layer, tracks_to_zarr
from src.utilities.register_image_stacks import registration_wrapper
import zarr
from ultrack.utils import estimate_parameters_from_labels, labels_to_edges
import glob2 as glob
from skimage.segmentation import watershed

# get path to raw nd2 file


def perform_tracking(root, experiment_date, well_num, tracking_config, seg_model, start_i=0, stop_i=None,
                     overwrite_tracking=False, overwrite_registration=False, use_stack_flag=False, suffix = ""):

    # imObject = nd2.ND2File(nd2_path)
    # res_raw = imObject.voxel_size()
    # scale_vec = np.asarray(res_raw)[::-1]

    tracking_name = tracking_config.replace(".txt", "")

    # get path to zarr file
    file_prefix = experiment_date + f"_well{well_num:04}"

    # set parameters
    # data_zarr = os.path.join(root, "built_data", "zarr_image_files", experiment_date, file_prefix + ".zarr")
    if use_stack_flag:
        mask_zarr_path = os.path.join(root, "built_data", "mask_stacks", seg_model, experiment_date, file_prefix + "_mask_stacks.zarr")
        mask_zarr_path_r = os.path.join(root, "built_data", "mask_stacks", seg_model, experiment_date, file_prefix + "_mask_stacks_registered.zarr")
    else:
        mask_zarr_path = os.path.join(root, "built_data", "mask_stacks", seg_model, experiment_date,
                                      file_prefix + "_mask_aff.zarr")
        mask_zarr_path_r = os.path.join(root, "built_data", "mask_stacks", seg_model, experiment_date,
                                        file_prefix + "_mask_aff_registered.zarr")


    # set output path for tracking results
    project_path = os.path.join(root, "tracking", experiment_date, tracking_name, f"well{well_num:04}" + suffix, "")
    if not os.path.isdir(project_path):
        os.makedirs(project_path)
    # get path to metadata
    metadata_path = os.path.join(root, "metadata", "tracking")

    # perform registration if necessary
    if not os.path.exists(mask_zarr_path_r) or overwrite_registration:

        # data_tzyx_full = zarr.open(data_zarr, mode='r')
        mask_tzyx_full = zarr.open(mask_zarr_path, mode='r')
        if stop_i is None:
            stop_i = mask_tzyx_full.shape[0]

        # register timerseries data. Load previous registration if it exists
        shift_array = registration_wrapper(root, experiment_date, well_index=well_num, model_name=seg_model, overwrite_flag=True)

        # initialize empty array
        store = zarr.DirectoryStore(mask_zarr_path_r)
        if use_stack_flag:
            mask_tzyx_r = zarr.open(store, mode='a', shape=mask_tzyx_full.shape, dtype=np.uint16, chunks=(1, 1,) + mask_tzyx_full.shape[2:])
            # shift masks to match registrationllyx_fu
            for i in tqdm(range(mask_tzyx_r.shape[1]), "Aligning masks..."):
                mask_tzyx_r[0, i] = mask_tzyx_full[0, i]
                for t in range(mask_tzyx_full.shape[0] - 1):
                    mask_tzyx_r[t + 1, i]= ndi.shift(mask_tzyx_full[t + 1, i], (shift_array[t + 1, :]), order=0)
        else:
            mask_tzyx_r = zarr.open(store, mode='a', shape=mask_tzyx_full.shape, dtype=np.uint16,
                                    chunks=(1,) + mask_tzyx_full.shape[1:])
            # shift masks to match registrationllyx_fu
            mask_tzyx_r[0] = mask_tzyx_full[0]
            for t in tqdm(range(mask_tzyx_r.shape[0] - 1), "Aligning masks..."):
                mask_tzyx_r[t + 1] = ndi.shift(mask_tzyx_full[t + 1], (shift_array[t + 1, :]), order=0)


        print("saving registered masks.")
        # initialize zarr file to save mask hierarchy
        meta_keys = mask_tzyx_full.attrs.keys()
        for meta_key in meta_keys:
            mask_tzyx_r.attrs[meta_key] = mask_tzyx_full.attrs[meta_key]

    else:
        store = zarr.DirectoryStore(mask_zarr_path_r)
        mask_tzyx_r = zarr.open(store, mode='r')
        if stop_i is None:
            stop_i = mask_tzyx_r.shape[0]

    # convert masks to list of arrays
    if use_stack_flag:
        mask_list = [mask_tzyx_r[start_i:stop_i, i, :, :, :] for i in range(mask_tzyx_r.shape[1])]
        out_shape = mask_list[0].shape
    else:
        mask_list = mask_tzyx_r[start_i:stop_i]
        out_shape = mask_list.shape

    scale_vec = mask_tzyx_r.attrs["voxel_size_um"]

    # load tracking config file
    cfg = load_config(os.path.join(metadata_path, tracking_config))
    cfg.data_config.working_dir = project_path

    # get tracking inputs
    dstore = zarr.DirectoryStore(project_path + "detection.zarr")
    bstore = zarr.DirectoryStore(project_path + "boundaries.zarr")
    detection = zarr.open(store=dstore, mode='a', shape=out_shape, dtype=bool, chunks=(1,) + out_shape[1:])
    boundaries = zarr.open(store=bstore, mode='a', shape=out_shape, dtype=np.uint16, chunks=(1,) + out_shape[1:])

    d, b = labels_to_edges(mask_list)
    for t in range(d.shape[0]):
        detection[t] = d[t]
        boundaries[t] = b[t]
    # Perform tracking
    print("Performing tracking...")
    track(
        cfg,
        detection=detection,
        edges=boundaries,
        scale=scale_vec,
        overwrite=overwrite_tracking,
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

    experiment_date = "20240620"
    root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/"
    well_num = 3
    use_centroids = False
    tracking_config = "tracking_jordao_frontier.txt"
    segmentation_model = "tdTom-bright-log-v5"
    add_label_spacer = False
    perform_tracking(root, experiment_date, well_num, tracking_config,
                     seg_model=segmentation_model, start_i=0, stop_i=None, overwrite_registration=None)
