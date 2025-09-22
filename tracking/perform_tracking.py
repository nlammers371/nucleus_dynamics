import napari
import os
import numpy as np
from ultrack import MainConfig, load_config, track, to_tracks_layer, tracks_to_zarr
from zarr.errors import ContainsArrayError
import zarr
from skimage.measure import regionprops
import napari
import dask.array as da
import pandas as pd
from src.build_killi.build_utils import labels_to_contours_nl
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from functools import partial
from tqdm.contrib.concurrent import process_map
import multiprocessing
from dask.diagnostics import ProgressBar
from zarr.sync import ProcessSynchronizer

def copy_zarr(frame, src, dst):
    dst[frame] = np.copy(src[frame])

def reindex_mask(frame, seg_zarr, lookup, track_df=None):

    if track_df is not None:
        # get taret frame IDs
        frame_tracks = track_df.loc[track_df["t"] == frame, "track_id"].values  # get the track ids for the current frame
        props = regionprops(seg_zarr[frame])
        label_vec = np.asarray([pr.label for pr in props])
        if np.all(np.isin(label_vec[1:], frame_tracks)):
            # if all labels are already in the frame_tracks, no need to reindex.
            return
    # Apply the lookup table to relabel the current frame.
    seg_zarr[frame] = lookup[seg_zarr[frame]]

def write_dask_to_zarr(frame, out_zarr, dask_array, frames_to_reindex, lookup):

    if frame not in frames_to_reindex:
        out_zarr[frame] = dask_array[frame].compute()
    else:
        out_zarr[frame] = lookup[dask_array[frame].compute()]

def combine_tracking_results(root, project_name, tracking_config, track_range1, track_range2, handoff_index=None,
                             suffix="", well_num=None, par_flag=False, n_workers=None, overwrite_flag=False):

    if well_num is None:
        well_num = 0

    if handoff_index is None:
        handoff_index = track_range2[0]  # default to end of first range

    if n_workers is None:
        total_cpus = multiprocessing.cpu_count()
        # Limit yourself to 33% of CPUs (rounded down, at least 1)
        n_workers = max(1, total_cpus // 3)


    tracking_name = tracking_config.replace(".txt", "")

    start_i1, stop_i1 = track_range1[0], track_range2[1]
    start_i2, stop_i2 = track_range2[0], track_range2[1]

    # get paths to tracking results
    project_path = os.path.join(root, "tracking", project_name, tracking_name, f"well{well_num:04}", "")
    project_sub_path1 = os.path.join(project_path, f"track_{start_i1:04}" + f"_{stop_i1:04}" + suffix, "")
    project_sub_path2 = os.path.join(project_path, f"track_{start_i2:04}" + f"_{stop_i2:04}" + suffix, "")

    # load tracking masks
    seg_zarr1 = zarr.open(os.path.join(project_sub_path1, "segments.zarr"), mode="r")
    seg_zarr2 = zarr.open(os.path.join(project_sub_path2, "segments.zarr"), mode="r")

    # load tracking DFs
    tracks_df1 = pd.read_csv(os.path.join(project_sub_path1, "tracks.csv"))
    tracks_df2 = pd.read_csv(os.path.join(project_sub_path2, "tracks.csv"))

    # initialize output paths
    combined_sub_path = os.path.join(project_path, f"track_{start_i1:04}" + f"_{stop_i2:04}" + "_cb", "")
    os.makedirs(combined_sub_path, exist_ok=True)

    # figure out track id mappings
    hi_rel1 = handoff_index - start_i1
    hi_rel2 = handoff_index - start_i2
    ref_frame1 = seg_zarr1[hi_rel1]  # get the reference frame from the first run02_segment
    ref_frame2 = seg_zarr2[hi_rel2]  # get the reference frame from the second run02_segment

    # generate array of track mask overlaps
    props1 = regionprops(ref_frame1)
    props2 = regionprops(ref_frame2)

    label_vec1 = np.array([prop.label for prop in props1])  # labels in the first frame
    label_vec2 = np.array([prop.label for prop in props2])  # labels in the second frame

    mask_overlap_array = np.zeros((len(props1), len(props2)), dtype=np.int32)
    for i, prop2 in enumerate(tqdm(props2)):
        coords2 = prop2.coords
        ref1_pixels = ref_frame1[coords2[:, 0], coords2[:, 1], coords2[:, 2]]  # get the pixels in the first frame
        ref1_pixels = ref1_pixels[ref1_pixels > 0]  # only consider valid pixels (greater than 0)
        lbu, lbc = np.unique(ref1_pixels,  return_counts=True)
        for l, lb in enumerate(lbu):
            mask_overlap_array[label_vec1 == lb, i] = lbc[l]  # fill the overlap array

    # solve for optimal mapping
    cost_matrix = -mask_overlap_array

    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # first, set mapping for those tracks that connect directly to prior tracks
    track2_label_index = np.unique(tracks_df2.loc[tracks_df2["t"] >=hi_rel2, "track_id"].values)   # unique track ids in the second frame
    track1_label_map = dict({})
    for i in range(len(row_ind)):
        track1_label_map[label_vec2[col_ind[i]]] = label_vec1[row_ind[i]]  # map the track ids from the first frame to the second

    # now, assign new labels to tracks that are new to the second piece
    start_id = np.max(ref_frame1) + 1
    ids_sorted = np.sort(track2_label_index[~np.isin(track2_label_index, list(track1_label_map.keys()))])  # find the track ids in the second frame that are not in the first
    new_ids = start_id + np.arange(0, len(ids_sorted))
    for i in range(len(ids_sorted)):
        track1_label_map[ids_sorted[i]] = new_ids[i]

    # make new track ID data frames
    tracks_df1_new = tracks_df1.copy()
    tracks_df1_new = tracks_df1_new.loc[tracks_df1_new["t"] < hi_rel1, :].reset_index(drop=True)  # only keep tracks before the handoff index
    tracks_df2_new = tracks_df2.copy()
    tracks_df2_new = tracks_df2_new.loc[tracks_df2_new["t"] >= hi_rel2, :].reset_index(drop=True) # only keep tracks after the handoff index
    tracks_df2_new["track_id_orig"] = tracks_df2_new["track_id"].copy()  # keep a copy of the original track ids
    tracks_df2_new["parent_track_id_orig"] = tracks_df2_new["parent_track_id"] .copy()  # keep a copy of the original parent track ids
    tracks_df1_ho = tracks_df1.loc[tracks_df1["t"] == hi_rel1, :]

    # First let's update track info for the continuation tracks. Specificaly, we need to adjust parent info
    ct_labels = label_vec2[col_ind]
    mapped_parent_filter = np.zeros((tracks_df2_new.shape[0],), dtype=bool)  # filter for mapped parent tracks
    for _, lb in enumerate(tqdm(ct_labels)):
        new_id = track1_label_map[lb]  # get the new track id for this label
        parent_id = tracks_df1_ho.loc[tracks_df1_ho["track_id"] == new_id, "parent_track_id"].to_numpy()[0]
        track_ft = (tracks_df2_new["track_id"] == lb)
        tracks_df2_new.loc[track_ft, "parent_track_id"] = parent_id  # update the parent id for the new track
        mapped_parent_filter = mapped_parent_filter | track_ft  # update the filter for mapped parent tracks

    # Now uddate all other track labels
    new_track_ids = tracks_df2_new["track_id"].map(track1_label_map).values.astype(int) # [track1_label_map[tri] for tri in tracks_df2_new["track_id"].values]  # get the new track ids for the second frame
    new_track_ids[new_track_ids < 0] = -1
    tracks_df2_new["track_id"] = new_track_ids

    new_parent_ids = tracks_df2_new.loc[~mapped_parent_filter, "parent_track_id"].map(track1_label_map).values.astype(int)
    tracks_df2_new.loc[~mapped_parent_filter, "parent_track_id"] = new_parent_ids

    # update frame index in tracks2
    tracks_df2_new["t"] = tracks_df2_new["t"] - np.min(tracks_df2_new["t"]) + hi_rel1

    # combine tracks
    tracks_df_cb = pd.concat([tracks_df1_new, tracks_df2_new], ignore_index=True)
    tracks_df_cb.to_csv(os.path.join(combined_sub_path, "tracks.csv"), index=False)

    # create new segments zarr for combined results
    combined_seg_zarr_path = os.path.join(combined_sub_path, "segments.zarr")
    # combined_seg_zarr = zarr.open(combined_seg_zarr_path, mode='a', shape=out_size, dtype=np.uint16, chunks=(1,) + out_size[1:])

    # Wrap them as Dask arrays. (Assuming their chunk sizes are already appropriate.)
    da1 = da.from_array(seg_zarr1, seg_zarr1.chunks) #chunks=(1,) + out_size[1:])
    da2 = da.from_array(seg_zarr2, seg_zarr1.chunks)

    # Concatenate them along the desired axis (for example, axis 0).
    combined = da.concatenate([da1[:hi_rel1], da2[hi_rel2:]], axis=0)

    # relabel frames that come from tracks2
    max_label = np.max(da2[-1])
    # Create an identity lookup table (i.e. each label maps to itself)
    lookup = np.arange(max_label + 1, dtype=da1.dtype)

    # Update the lookup table with your mapping.
    for old_label, new_label in track1_label_map.items():
        # It's assumed that old_label is within [0, max_label]
        lookup[old_label] = new_label

    try:
        with ProgressBar():
            combined.to_zarr(combined_seg_zarr_path, overwrite=overwrite_flag)
    except ContainsArrayError:
        print(f"An array already exists at {combined_seg_zarr_path}. Skipping writing and continuing.")

    # reopen
    combined_seg_zarr = zarr.open(combined_seg_zarr_path, mode='a')

    if par_flag:
        reindex_run = partial(reindex_mask, seg_zarr=combined_seg_zarr, lookup=lookup, track_df=tracks_df_cb)
        process_map(reindex_run, range(hi_rel1, combined.shape[0]), max_workers=n_workers, chunksize=1)
    else:
        # Apply the lookup table to relabel the combined array.
        for frame in tqdm(range(hi_rel1, combined.shape[0]), "Reindexing run02_segment labels"):
            reindex_mask(frame=frame, seg_zarr=combined_seg_zarr, lookup=lookup, track_df=tracks_df_cb)

    return {}

# create function to load and visualize tracking results using napari
def check_tracking(root, project_name, tracking_config, seg_model="", suffix="", well_num=None, start_i=0, stop_i=None,
                   use_marker_masks=False, use_stack_flag=False, use_fused=True, view_range=None, tracks_only=False):

    # get path to zarr file
    if well_num is not None:
        file_prefix = project_name + f"_well{well_num:04}"
        subfolder = project_name
    else:
        file_prefix = project_name
        subfolder = ""
        well_num = 0

    # get name
    tracking_name = tracking_config.replace(".txt", "")

    # data_zarr = os.path.join(root, "built_data", "zarr_image_files", project_name, file_prefix + ".zarr")
    if use_stack_flag:
        mask_zarr_path = os.path.join(root, "built_data", "mask_stacks", seg_model, subfolder,
                                      file_prefix + "_mask_stacks.zarr")
        # mask_zarr_path_r = os.path.join(root, "built_data", "mask_stacks", seg_model, project_name, file_prefix + "_mask_stacks_registered.zarr")
    elif use_fused:
        mask_zarr_path = os.path.join(root, "built_data", "mask_stacks", seg_model, subfolder,
                                      file_prefix + "_mask_fused.zarr")
    elif use_marker_masks:
        mask_zarr_path = os.path.join(root, "built_data", "mask_stacks", seg_model, subfolder,
                                      file_prefix + "_marker_masks.zarr")
    else:
        mask_zarr_path = os.path.join(root, "built_data", "mask_stacks", seg_model, project_name,
                                      file_prefix + "_mask_aff.zarr")

    # image_zarr_path = os.path.join(root, "built_data", "zarr_image_files", file_prefix + "_fused.zarr")
    # image_zarr = zarr.open(image_zarr_path, mode='r')
    if use_marker_masks:
        project_name += "_marker"

    # set output path for tracking results
    project_path = os.path.join(root, "tracking", project_name, tracking_name, f"well{well_num:04}", "")
    project_sub_path = os.path.join(project_path, f"track_{start_i:04}" + f"_{stop_i:04}" + suffix, "")

    # load input mask
    mask_tzyx = zarr.open(mask_zarr_path, mode='r')
    if "voxel_size_um" not in mask_tzyx.attrs.keys():
        ad = mask_tzyx.attrs
        scale_vec = [ad["PhysicalSizeZ"], ad["PhysicalSizeY"], ad["PhysicalSizeX"]]
    else:
        scale_vec = mask_tzyx.attrs["voxel_size_um"]

    # load tracking masks
    label_path = os.path.join(project_sub_path, "segments.zarr")
    seg_zarr = zarr.open(label_path, mode='r')

    if view_range is None:
        view_range = np.arange(start_i, stop_i)

    # load image data
    metadata_path = os.path.join(root, "metadata", "tracking")
    cfg = load_config(os.path.join(metadata_path, tracking_config + ".txt"))

    # load tracks
    tracks_df = pd.read_csv(os.path.join(project_sub_path, "tracks.csv"))

    tracks_df_plot = tracks_df.loc[np.isin(tracks_df["t"], view_range), :]
    tracks_df_plot.loc[:, "t"] = tracks_df_plot.loc[:, "t"] - view_range[0]

    viewer = napari.Viewer(ndisplay=3)  # view_image(data_zarr_da, scale=tuple(scale_vec))

    viewer.add_tracks(
        tracks_df_plot[["track_id", "t", "z", "y", "x"]],
        name="tracks",
        scale=tuple(scale_vec),
        translate=(0, 0, 0, 0),
        visible=False,
    )
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "um"
    # out_dir = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\slides\\killifish\\20241001\\nls_frames\\"
    # for frame in tqdm(range(0, 1600, 2)):
    #     viewer.dims.current_step = (frame, 0, 0, 0)
    #     viewer.screenshot(out_dir + f"frame{frame:04}.tif", canvas_only=True, scale=5)

    if not tracks_only:
        seg_zarr_plot = seg_zarr[view_range]
        viewer.add_labels(
            seg_zarr_plot,
            name="segments",
            scale=tuple(scale_vec),
            translate=(0, 0, 0, 0),
        ).contour = 2

    return viewer


def perform_tracking(root, project_name, tracking_config, seg_model, well_num=None, start_i=0, stop_i=None,
                     use_stack_flag=False, use_marker_masks=False, use_fused=True, suffix="", par_seg_flag=True, last_filter_start_i=None):

    tracking_name = tracking_config.replace(".txt", "")

    # get path to zarr file
    if well_num is not None:
        file_prefix = project_name + f"_well{well_num:04}"
        subfolder = project_name
    else:
        file_prefix = project_name
        subfolder = ""
        well_num = 0

    # set parameters
    # data_zarr = os.path.join(root, "built_data", "zarr_image_files", project_name, file_prefix + ".zarr")
    if use_stack_flag:
        mask_zarr_path = os.path.join(root, "built_data", "mask_stacks", seg_model, subfolder, file_prefix + "_mask_stacks.zarr")
        # mask_zarr_path_r = os.path.join(root, "built_data", "mask_stacks", seg_model, project_name, file_prefix + "_mask_stacks_registered.zarr")
    elif use_fused:
        mask_zarr_path = os.path.join(root, "built_data", "mask_stacks", seg_model, subfolder,
                                      file_prefix + "_mask_fused.zarr")
    elif use_marker_masks:
        mask_zarr_path = os.path.join(root, "built_data", "mask_stacks", seg_model, subfolder,
                                      file_prefix + "_marker_masks.zarr")
    else:
        mask_zarr_path = os.path.join(root, "built_data", "mask_stacks", seg_model, subfolder,
                                      file_prefix + "_mask_aff.zarr")

    # get path to metadata
    metadata_path = os.path.join(root, "metadata", "tracking")

    mask_tzyx = zarr.open(mask_zarr_path, mode='a')
    mask_tzyx_da = da.from_zarr(mask_tzyx)

    if stop_i is None:
        stop_i = mask_tzyx.shape[0]

    # set output path for tracking results
    if use_marker_masks:
        project_name += "_marker"

    seg_path = os.path.join(root, "tracking", project_name, "segmentation", f"well{well_num:04}", "")
    project_path = os.path.join(root, "tracking", project_name, tracking_name, f"well{well_num:04}", "")
    project_sub_path = os.path.join(project_path, f"track_{start_i:04}" + f"_{stop_i:04}" + suffix, "")
    os.makedirs(project_sub_path, exist_ok=True)
    full_shape = mask_tzyx.shape

    if (len(list(mask_tzyx.attrs.keys())) == 0) and use_fused:
        ref_path = os.path.join(root, "built_data", "mask_stacks", seg_model,
                                      file_prefix + "_side1_mask_aff.zarr")
        mask_ref = zarr.open(ref_path, mode='r')
        for key in mask_ref.attrs.keys():
            mask_tzyx.attrs[key] = mask_ref.attrs[key]

    if "voxel_size_um" not in mask_tzyx.attrs.keys():
        ad = mask_tzyx.attrs
        scale_vec = [ad["PhysicalSizeZ"], ad["PhysicalSizeY"], ad["PhysicalSizeX"]]
    else:
        scale_vec = mask_tzyx.attrs["voxel_size_um"]

    # load tracking config file
    cfg = load_config(os.path.join(metadata_path, tracking_config + ".txt"))
    cfg.data_config.working_dir = project_sub_path

    # get tracking inputs
    # segment_flag = not os.path.isdir()

    #  figur e out which indices to write
    # get all indices
    all_indices = set(range(start_i, stop_i))

    # List files directly within zarr directory (recursive search):
    if not os.path.isdir(seg_path + "detection.zarr"):
        existing_chunks = []
    else:
        existing_chunks = os.listdir(seg_path + "detection.zarr")

    # Extract time indices from chunk filenames:
    written_indices = set(int(fname.split('.')[0])
                          for fname in existing_chunks if fname[0].isdigit())

    segment_indices = np.asarray(sorted(all_indices - written_indices))

    # segment_flag = np.any(empty_indices)

    dstore = zarr.DirectoryStore(seg_path + "detection.zarr")
    bstore = zarr.DirectoryStore(seg_path + "boundaries.zarr")
    detection = zarr.open(store=dstore, mode='a', shape=full_shape, dtype=bool, chunks=(1,) + full_shape[1:])
    boundaries = zarr.open(store=bstore, mode='a', shape=full_shape, dtype=np.uint16, chunks=(1,) + full_shape[1:])

    if segment_indices.size > 0:
        d, b = labels_to_contours_nl(mask_tzyx_da, segment_indices, par_flag=par_seg_flag,
                                     last_filter_start_i=last_filter_start_i, scale_vec=scale_vec)
        for t in range(d.shape[0]):
            detection[segment_indices[t]] = d[t]
            boundaries[segment_indices[t]] = b[t]

    detection = da.from_zarr(detection)
    boundaries = da.from_zarr(boundaries)

    detection = detection[start_i:stop_i]
    boundaries = boundaries[start_i:stop_i]

    # Perform tracking
    print("Performing tracking...")
    track(
        cfg,
        detection=detection,
        edges=boundaries,
        scale=scale_vec
    )

    print("Saving results...")
    tracks_df, graph = to_tracks_layer(cfg)
    tracks_df.to_csv(project_sub_path + "tracks.csv", index=False)

    segments = tracks_to_zarr(
        cfg,
        tracks_df,
        store_or_path=project_sub_path + "segments.zarr",
        overwrite=True,
    )
    print("Done.")

    return segments


if __name__ == '__main__':

    project_name = "20240620"
    # root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/"
    # well_num = 3
    # use_centroids = False
    # tracking_config = "tracking_jordao_frontier.txt"
    # segmentation_model = "tdTom-bright-log-v5"
    # add_label_spacer = False
    # perform_tracking(root, project_name, well_num, tracking_config,
    #                  seg_model=segmentation_model, start_i=0, stop_i=None, overwrite_registration=None)
