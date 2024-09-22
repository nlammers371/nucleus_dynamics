import os
import zarr
from utilities.functions import path_leaf
import numpy as np
from omnipose.core import compute_masks
from cellpose.core import use_gpu
from skimage.segmentation import watershed
from scipy.ndimage import zoom
from tqdm import tqdm
from skimage import morphology
import pandas as pd
import glob2 as glob

def restitch_masks(mask_stack_zarr_path, prob_zarr_path, thresh_range, min_mask_size=15):

    # open the zarr file
    mask_stack_zarr = zarr.open(mask_stack_zarr_path, mode="a")
    mask_aff_zarr_path = mask_stack_zarr_path.replace("stacks.zarr", "aff.zarr")
    mask_aff_zarr = zarr.open(mask_aff_zarr_path, mode="r+")
    prob_zarr = zarr.open(prob_zarr_path, mode="r")
    prob_levels = mask_stack_zarr.attrs["prob_levels"]
    if len(prob_levels) == 0:
        prob_levels = dict({})
        for t in range(0, mask_stack_zarr.shape[0]):
            prob_levels[str(int(t))] = [-4, -2, 0, 2, 4, 6, 8]

        mask_stack_zarr.attrs["prob_levels"] = prob_levels
        mask_aff_zarr.attrs["prob_levels"] = prob_levels

    # iterate through time points
    for t in tqdm(range(mask_stack_zarr.shape[0])):

        prob_levels_t = prob_levels[str(t)]
        indices_to_use = np.where(np.isin(prob_levels_t, thresh_range))[0]

        # initialize
        masks_curr = mask_stack_zarr[t, indices_to_use[0], :, :, :]  # start with the most permissive mask

        for m in tqdm(indices_to_use[1:], "Performing affinity stitching..."):
            # get next layer of labels
            aff_labels = mask_stack_zarr[t, m, :, :, :]

            # get union of two masks
            mask_u = (masks_curr + aff_labels) > 0

            # get label vectors
            curr_vec = masks_curr[mask_u]
            next_vec = aff_labels[mask_u]

            # get index vec
            u_indices = np.where(mask_u)

            # get lists of unique labels
            labels_u_curr = np.unique(curr_vec)

            # for each label in the new layer, find label in prev layer that it most overlaps
            lb_df = pd.DataFrame(next_vec, columns=["next"])
            lb_df["curr"] = curr_vec

            # get most frequent curr label for each new label
            m_df = lb_df.groupby(by=["next"]).agg(lambda x: pd.Series.mode(x)[0]).reset_index()
            top_label_vec = m_df.loc[:, "curr"].to_numpy()

            # get most frequent new label for each curr label
            # m_df2 = lb_df.groupby(by=["curr"]).agg(lambda x: pd.Series.mode(x)[0]).reset_index()

            # merge info back on
            lb_df = lb_df.merge(m_df.rename(columns={"curr": "top_curr"}), how="left", on="next")
            # lb_df = lb_df.merge(m_df2.rename(columns={"next": "top_next"}), how="left", on="curr")

            # initialize mask and marker arrays for watershed
            mask_array = (masks_curr > 0) * 1  # this dictates the limits of what can be foreground

            # initialize marker array for watershed seeding
            marker_array = np.zeros(masks_curr.shape, dtype=np.uint16)

            # get indices to populate
            ft = (lb_df.loc[:, "curr"] == lb_df.loc[:, "top_curr"]) & (lb_df.loc[:, "next"] > 0)
            lb_indices = tuple(u[ft] for u in u_indices)

            # generate new label set
            _, new_labels = np.unique(next_vec[ft], return_inverse=True)
            marker_array[lb_indices] = new_labels + 1

            # add markers from base that do not appear in new layer
            included_base_labels = np.unique(top_label_vec)
            max_lb_curr = np.max(marker_array) + 1
            missing_labels = np.asarray(list(set(labels_u_curr) - set(included_base_labels)))

            ft2 = np.isin(curr_vec, missing_labels) & (~ft)
            lb_indices2 = tuple(u[ft2] for u in u_indices)

            _, missed_labels = np.unique(curr_vec[ft2], return_inverse=True)
            marker_array[lb_indices2] = missed_labels + 1 + max_lb_curr

            # finally, expand the mask array to accommodate markers from the new labels that are not in the reference
            mask_array = (mask_array + marker_array) > 0

            # calculate watershed
            wt_array = watershed(image=-prob_zarr[t], markers=marker_array, mask=mask_array, watershed_line=True)

            masks_curr = wt_array

        masks_out = morphology.remove_small_objects(masks_curr, min_mask_size)

        mask_aff_zarr[t] = masks_out
        ams = mask_aff_zarr.attrs["prob_levels"]
        ams[str(int(t))] = list(thresh_range)
        mask_aff_zarr.attrs["prob_levels"] = ams

def do_affinity_stitching(prob_array, grad_array, scale_vec, seg_res=None, prob_thresh_range=None,
                                                    niter=100, min_mask_size=5, max_mask_size=1e5):

    if prob_thresh_range is None:
        raise Exception("No threshold range was provided")

    # get resizing info
    shape_orig = np.asarray(prob_array.shape)
    shape_iso = shape_orig.copy()
    iso_factor = scale_vec[0] / scale_vec[1]
    shape_iso[0] = int(shape_iso[0] * iso_factor)

    if seg_res is not None:
        rs_factor = scale_vec[1] / seg_res
        shape_iso = (shape_iso * rs_factor).astype(int)
    else:
        rs_factor = 1.0

    use_GPU = use_gpu()

    # get device info
    device = (
        "cuda"
        if use_GPU
        else "cpu"
    )

    print("Resizing arrays...")
    zoom_factor = np.divide(shape_iso, shape_orig)
    grad_array_rs = zoom(grad_array, (1,) + tuple(zoom_factor), order=1) * rs_factor
    prob_array_rs = zoom(prob_array, zoom_factor, order=1)  # resize(prob_array, shape_iso, preserve_range=True, order=1)

    grad_array_rs[0] = grad_array_rs[0]
    # list of prob thresholds to use
    # prob_thresh_range = list(range(min_prob, max_prob + prob_increment, prob_increment))
    seg_hypothesis_array = np.zeros((len(prob_thresh_range),) + prob_array_rs.shape, dtype=np.uint16)

    for m, mask_threshold in enumerate(tqdm(prob_thresh_range, "Calculating affinity masks...")):
        mask_aff, _, _, _, _ = compute_masks(grad_array_rs, prob_array_rs,
                                             do_3D=True,
                                             niter=niter,
                                             boundary_seg=False,
                                             affinity_seg=True,
                                             min_size=10,
                                             max_size=max_mask_size,
                                             mask_threshold=mask_threshold,
                                             verbose=False,
                                             interp=True,
                                             omni=True,
                                             cluster=False,
                                             use_gpu=use_GPU,
                                             device=device,
                                             nclasses=2,
                                             dim=3)

        seg_hypothesis_array[m] = mask_aff

    ######
    # performing hierarchical watershed

    # initialize
    masks_curr = seg_hypothesis_array[0]  # start with the most permissive mask

    for m in tqdm(range(1, len(prob_thresh_range)), "Performing affinity stitching..."):
        # get next layer of labels
        aff_labels = seg_hypothesis_array[m]

        # get union of two masks
        mask_u = (masks_curr + aff_labels) > 0

        # get label vectors
        curr_vec = masks_curr[mask_u]
        next_vec = aff_labels[mask_u]

        # get index vec
        u_indices = np.where(mask_u)

        # get lists of unique labels
        labels_u_curr = np.unique(curr_vec)

        # for each label in the new layer, find label in prev layer that it most overlaps
        lb_df = pd.DataFrame(next_vec, columns=["next"])
        lb_df["curr"] = curr_vec

        # get most frequent curr label for each new label
        m_df = lb_df.groupby(by=["next"]).agg(lambda x: pd.Series.mode(x)[0]).reset_index()
        top_label_vec = m_df.loc[:, "curr"].to_numpy()

        # get most frequent new label for each curr label
        # m_df2 = lb_df.groupby(by=["curr"]).agg(lambda x: pd.Series.mode(x)[0]).reset_index()

        # merge info back on
        lb_df = lb_df.merge(m_df.rename(columns={"curr": "top_curr"}), how="left", on="next")
        # lb_df = lb_df.merge(m_df2.rename(columns={"next": "top_next"}), how="left", on="curr")

        # initialize mask and marker arrays for watershed
        mask_array = (masks_curr > 0) * 1  # this dictates the limits of what can be foreground

        # initialize marker array for watershed seeding
        marker_array = np.zeros(masks_curr.shape, dtype=np.uint16)

        # get indices to populate
        ft = (lb_df.loc[:, "curr"] == lb_df.loc[:, "top_curr"]) & (lb_df.loc[:, "next"] > 0)
        lb_indices = tuple(u[ft] for u in u_indices)

        # generate new label set
        _, new_labels = np.unique(next_vec[ft], return_inverse=True)
        marker_array[lb_indices] = new_labels + 1

        # add markers from base that do not appear in new layer
        included_base_labels = np.unique(top_label_vec)
        max_lb_curr = np.max(marker_array) + 1
        missing_labels = np.asarray(list(set(labels_u_curr) - set(included_base_labels)))

        ft2 = np.isin(curr_vec, missing_labels) & (~ft)
        lb_indices2 = tuple(u[ft2] for u in u_indices)

        _, missed_labels = np.unique(curr_vec[ft2], return_inverse=True)
        marker_array[lb_indices2] = missed_labels + 1 + max_lb_curr

        # finally, expand the mask array to accommodate markers from the new labels that are not in the reference
        mask_array = (mask_array + marker_array) > 0

        # calculate watershed
        wt_array = watershed(image=-prob_array_rs, markers=marker_array, mask=mask_array, watershed_line=True)

        masks_curr = wt_array

    # resize
    masks_out_rs = zoom(masks_curr, zoom_factor**-1, order=0)
    masks_out_rs = morphology.remove_small_objects(masks_out_rs, min_mask_size)

    # resize each hypothesis
    seg_hypothesis_array_rs = zoom(seg_hypothesis_array, (1,) + tuple(zoom_factor**-1), order=0)

    return masks_out_rs, seg_hypothesis_array_rs


def stitch_cellpose_labels(root, model_name, experiment_date, well_range=None, prob_thresh_range=None, overwrite=False,
                           seg_res=None):

    if prob_thresh_range is None:
        prob_thresh_range = np.arange(-8, 9, 4)

    if seg_res is None:
        seg_res = 0.65

    # get raw data dir
    raw_directory = os.path.join(root, "built_data", "zarr_image_files", experiment_date, '')
    # get path to cellpose output
    cellpose_directory = os.path.join(root, "built_data", "cellpose_output", model_name, experiment_date, '')
    # make directory to write stitched labels
    out_directory = os.path.join(root, "built_data", "mask_stacks", model_name, experiment_date, '')
    if not os.path.isdir(out_directory):
        os.makedirs(out_directory)

    # load curation data if we have it
    # curation_path = os.path.join(root, "metadata", "curation", experiment_date + "_curation_info.csv")
    # has_curation_info = os.path.isfile(curation_path)
    # if has_curation_info:
    #     curation_df = pd.read_csv(curation_path)
    #     curation_df_long = pd.melt(curation_df,
    #                                id_vars=["series_number", "notes", "tbx5a_flag", "follow_up_flag"],
    #                                var_name="time_index", value_name="qc_flag")
    #     curation_df_long["time_index"] = curation_df_long["time_index"].astype(int)

    # get list of wells with labels to stitch
    well_list = sorted(glob.glob(cellpose_directory + "*_probs.zarr"))

    if well_range is not None:
        well_range = [w for w in well_range if w < len(well_list)]
        well_list = [well_list[w] for w in well_range]

    for _, well in enumerate(well_list):

        # get well index
        # well_index = well.find("_well")
        # well_num = int(well[well_index + 5:well_index + 9])

        #########
        file_prefix = path_leaf(well).replace("_probs.zarr", "")
        print("Stitching data from " + file_prefix)
        raw_name = os.path.join(raw_directory, file_prefix + ".zarr")
        prob_name = os.path.join(cellpose_directory, file_prefix + "_probs.zarr")
        grad_name = os.path.join(cellpose_directory, file_prefix + "_grads.zarr")

        prob_zarr = zarr.open(prob_name, mode="a")
        grad_zarr = zarr.open(grad_name, mode="a")
        if (experiment_date == "20240425") or (experiment_date == "20240424"):
            data_zarr = zarr.open(raw_name, mode="r")
        else:
            data_zarr = prob_zarr

        time_indices0 = np.arange(prob_zarr.shape[0])

        # generate zarr store for stitched masks
        multi_mask_zarr_path = os.path.join(out_directory, file_prefix + "_mask_stacks.zarr")
        aff_mask_zarr_path = os.path.join(out_directory, file_prefix + "_mask_aff.zarr")
        prev_flag = os.path.isdir(multi_mask_zarr_path)
        
        # initialize zarr file to save mask hierarchy
        multi_mask_zarr = zarr.open(multi_mask_zarr_path, mode='a', shape=(prob_zarr.shape[0],) + (len(prob_thresh_range),) + tuple(prob_zarr.shape[1:]),
                                dtype=np.uint16, chunks=(1, 1,) + prob_zarr.shape[1:])
        
        # initialize zarr to save current best mask
        aff_mask_zarr = zarr.open(aff_mask_zarr_path, mode='a', shape=prob_zarr.shape,
                                    dtype=np.uint16, chunks=(1,) + prob_zarr.shape[1:])

        # transfer metadata from raw data to cellpose products
        prob_keys = prob_zarr.attrs.keys()
        meta_keys = data_zarr.attrs.keys()

        for meta_key in meta_keys:
            multi_mask_zarr.attrs[meta_key] = data_zarr.attrs[meta_key]
            aff_mask_zarr.attrs[meta_key] = data_zarr.attrs[meta_key]
            if "voxel_size_um" not in prob_keys:
                prob_zarr.attrs[meta_key] = data_zarr.attrs[meta_key]
                grad_zarr.attrs[meta_key] = data_zarr.attrs[meta_key]

        multi_mask_zarr.attrs["prob_levels"] = dict({})
        aff_mask_zarr.attrs["prob_levels"] = dict({})

        if "model_path" not in prob_keys:
            prob_zarr.attrs["model_path"] = model_name
            grad_zarr.attrs["model_path"] = model_name

        scale_vec = data_zarr.attrs["voxel_size_um"]

        # determine which indices to stitch
        print("Determining which time points need stitching...")
        if overwrite | (not prev_flag):
            write_indices = time_indices0
        else:
            write_indices = []
            for t in time_indices0:
                nz_flag = np.any(multi_mask_zarr[t, :, :, :] != 0)
                if not nz_flag:
                    write_indices.append(t)
            write_indices = np.asarray(write_indices)

        # iterate through time points
        print("Stitching labels...")
        for time_int in tqdm(write_indices):

            # use affinity graph method from omnipose core to stitch masks at different probability levels
            # do the stitching
            grad_array = grad_zarr[time_int, :, :, :, :]
            prob_array = prob_zarr[time_int, :, :, :]

            if np.any(prob_array != 0):
                # perform stitching
                stitched_labels, mask_stack = do_affinity_stitching(prob_array, grad_array,  scale_vec=scale_vec,
                                                                        prob_thresh_range=prob_thresh_range, seg_res=seg_res)  # NL: these were used for 202404 min_prob=-2, max_prob=8,

                # save
                multi_mask_zarr[time_int] = mask_stack
                aff_mask_zarr[time_int] = stitched_labels

                mms = multi_mask_zarr.attrs["prob_levels"]
                mms[int(time_int)] = list(prob_thresh_range)
                multi_mask_zarr.attrs["prob_levels"] = mms

                ams = aff_mask_zarr.attrs["prob_levels"]
                ams[int(time_int)] = list(prob_thresh_range)
                aff_mask_zarr.attrs["prob_levels"] = ams
                # aff_mask_zarr.attrs["prob_levels"][time_int] = prob_thresh_range
            else:
                print(f"Skipping time point {time_int:04}: no cellpose output found")


if __name__ == "__main__":
    # root = "E:\\Nick\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\pecfin_dynamics\\"
    root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/"
    experiment_date = "20240223"
    model_name = "log-v5"
    overwrite = False

    stitch_cellpose_labels(root, model_name, experiment_date, overwrite)