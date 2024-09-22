import glob2 as glob
import os
from typing import Any
from typing import Dict
from utilities.functions import path_leaf
import pandas as pd
import nd2
# import openpyxl
import numpy as np
import dask.array as da

def parse_plate_metadata(root, experiment_date, sheet_names=None):

    plate_directory = os.path.join(root, "metadata", "plate_maps", experiment_date + "_plate_map.xlsx")

    if sheet_names is None:
        sheet_names = ["series_number_map", "genotype", "start_age_hpf", "chem_perturbation"]

    row_letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
    col_nums = [i for i in range(12)]

    well_coord_list = []
    for row in row_letters:
        for col in col_nums:
            well_coord_list.append(row + f"{col:02}")

    well_coord_list = np.asarray(well_coord_list)

    xl_temp = pd.ExcelFile(plate_directory)
    # extract nd2 series info
    series_vec_raw = xl_temp.parse(sheet_names[0]).iloc[0:8, 1:13].values.ravel()
    nn_indices = np.where(~np.isnan(series_vec_raw))[0]
    series_vec = series_vec_raw[nn_indices]
    # extract staging info
    start_age_hpf = xl_temp.parse(sheet_names[2]).iloc[0:8, 1:13].values.ravel()
    start_age_hpf = start_age_hpf[nn_indices]
    # extract genotype info
    genotype_vec = xl_temp.parse(sheet_names[1]).iloc[0:8, 1:13].values.ravel()
    genotype_vec = genotype_vec[nn_indices]

    # extract chemical info
    chem_vec = xl_temp.parse(sheet_names[3]).iloc[0:8, 1:13].values.ravel()
    chem_vec = chem_vec[nn_indices]

    # get list of well ID coordinates
    well_coord_list = well_coord_list[nn_indices]

    plate_df = pd.DataFrame(series_vec[:, np.newaxis], columns=["nd2_series"])
    plate_df["well_id"] = well_coord_list
    plate_df["genotype"] = genotype_vec
    plate_df["chem_i"] = chem_vec
    plate_df["start_age_hpf"] = start_age_hpf

    return plate_df

def parse_nd2_metadata(nd2_path):

    imObject = nd2.ND2File(nd2_path)
    im_raw_dask = imObject.to_dask()
    n_channels = len(imObject.metadata.channels)

    # [well, channel, time, z, y, x]
    if len(im_raw_dask.shape) == 4 and n_channels == 1:
        im_raw_dask = im_raw_dask[:, None, None, :, :, :]

    elif len(im_raw_dask.shape) == 5 and n_channels == 1:
        im_raw_dask = im_raw_dask[:, None, :, :, :, :]

    elif len(im_raw_dask.shape) == 5 and n_channels == 2:
        im_raw_dask = da.transpose(im_raw_dask, (0, 2, 1, 3, 4))
        im_raw_dask = im_raw_dask[:, :, None, :, :, :]

    elif len(im_raw_dask.shape) == 6 and n_channels == 2:
        im_raw_dask = da.transpose(im_raw_dask, (0, 2, 1, 3, 4, 5))

    im_shape = im_raw_dask.shape
    n_z_slices = im_shape[3]
    n_time_points = im_shape[2]
    n_wells = im_shape[0]

    scale_vec = imObject.voxel_size()

    # extract frame times
    n_frames_total = imObject.frame_metadata(0).contents.frameCount
    frame_time_vec = [imObject.frame_metadata(i).channels[0].time.relativeTimeMs / 1000 for i in
                      range(0, n_frames_total, n_z_slices)]

    # check for common nd2 artifact where time stamps jump midway through
    dt_frame_approx = (imObject.frame_metadata(n_z_slices).channels[0].time.relativeTimeMs -
                       imObject.frame_metadata(0).channels[0].time.relativeTimeMs) / 1000
    jump_ind = np.where(np.diff(frame_time_vec) > 2 * dt_frame_approx)[0]  # typically it is multiple orders of magnitude to large
    if len(jump_ind) > 0:
        jump_ind = jump_ind[0]
        # prior to this point we will just use the time stamps. We will extrapolate to get subsequent time points
        nf = jump_ind - 1 - int(jump_ind / 2)
        dt_frame_est = (frame_time_vec[jump_ind - 1] - frame_time_vec[int(jump_ind / 2)]) / nf
        base_time = frame_time_vec[jump_ind - 1]
        for f in range(jump_ind, len(frame_time_vec)):
            frame_time_vec[f] = base_time + dt_frame_est * (f - jump_ind)
    frame_time_vec = np.asarray(frame_time_vec)

    # get well positions
    stage_zyx_array = np.empty((n_wells * n_time_points, 3))
    for t in range(n_time_points):
        for w in range(n_wells):
            base_ind = t * n_wells + w
            slice_ind = base_ind * n_z_slices

            stage_zyx_array[base_ind, :] = np.asarray(
                imObject.frame_metadata(slice_ind).channels[0].position.stagePositionUm)[::-1]

    ###################
    # Pull it together into dataframe
    ###################
    well_df = pd.DataFrame(np.tile(range(1, n_wells + 1), n_time_points)[:, np.newaxis], columns=["nd2_series"])
    well_df["time_index"] = np.repeat(range(n_time_points), n_wells)
    well_df["well_index"] = np.tile(range(n_wells), n_time_points)

    # add additional info
    well_df["time"] = frame_time_vec
    well_df["stage_z_um"] = stage_zyx_array[:, 0]
    well_df["stage_y_um"] = stage_zyx_array[:, 1]
    well_df["stage_x_um"] = stage_zyx_array[:, 2]

    well_df["x_res_um"] = scale_vec[0]
    well_df["y_res_um"] = scale_vec[1]
    well_df["z_res_um"] = scale_vec[2]

    imObject.close()

    return well_df
def parse_curation_metadata(root, experiment_date):
    curation_path = os.path.join(root, "metadata", "curation", experiment_date + "_curation_info.xlsx")
    if os.path.isfile(curation_path):
        curation_xl = pd.ExcelFile(curation_path)
        curation_df = curation_xl.parse(curation_xl.sheet_names[0])
        curation_df_long = pd.melt(curation_df,
                                   id_vars=["series_number", "notes", "tbx5a_flag", "follow_up_flag"],
                                   var_name="time_index", value_name="qc_flag")
        # time_ind_vec = [int(t[1:]) for t in curation_df_long["time_string"].values]
        # curation_df_long["time_index"] = time_ind_vec
        curation_df_long = curation_df_long.rename(columns={"series_number": "nd2_series"})

    else:
        curation_df_long = None
        curation_df = None

    return curation_df_long, curation_df

def extract_frame_metadata(
    root: str,
    experiment_date: str
) -> Dict[str, Any]:


    raw_directory = os.path.join(root, "raw_data", experiment_date, '')

    save_directory = os.path.join(root, "metadata", "frame_metadata", '')
    if not os.path.isdir(save_directory):
        os.makedirs(save_directory)

    # get list of images
    image_list = sorted(glob.glob(raw_directory + "*.nd2"))

    if not os.path.isdir(save_directory):
        os.makedirs(save_directory)

    if len(image_list) > 1:
        raise Exception("Multiple .nd2 files were found in target directory. Make sure to put fullembryo images into a subdirectory")

    nd2_path = image_list[0]
    imObject = nd2.ND2File(nd2_path)
    im_raw_dask = imObject.to_dask()

    n_channels = len(imObject.metadata.channels)

    # [well, channel, time, z, y, x]
    if len(im_raw_dask.shape) == 4 and n_channels == 1:
        im_raw_dask = im_raw_dask[:, None, None, :, :, :]

    elif len(im_raw_dask.shape) == 5 and n_channels == 1:
        im_raw_dask = im_raw_dask[:, None, :, :, :, :]

    elif len(im_raw_dask.shape) == 5 and n_channels == 2:
        im_raw_dask = da.transpose(im_raw_dask, (0, 2, 1, 3, 4))
        im_raw_dask = im_raw_dask[:, :, None, :, :, :]

    elif len(im_raw_dask.shape) == 6 and n_channels == 2:
        im_raw_dask = da.transpose(im_raw_dask, (0, 2, 1, 3, 4, 5))

    nd2_shape = im_raw_dask.shape

    metadata = dict({})
    metadata["n_time_points"] = nd2_shape[2]
    metadata["n_wells"] = nd2_shape[0]
    n_z = nd2_shape[-3]
    n_x = nd2_shape[-1]
    n_y = nd2_shape[-2]

    metadata["n_channels"] = n_channels
    metadata["zyx_shape"] = tuple([n_z, n_y, n_x])
    metadata["voxel_size_um"] = tuple(np.asarray(imObject.voxel_size())[::-1])

    im_name = path_leaf(nd2_path)
    print("processing " + im_name)

    ####################
    # Process information from plate map
    plate_df = parse_plate_metadata(root, experiment_date)

    ####################
    # Process extract information from nd2 metadata
    ####################
    # join on plate info using series id
    well_df = parse_nd2_metadata(nd2_path)
    plate_cols = plate_df.columns
    well_cols = well_df.columns
    well_df = well_df.merge(plate_df, on="nd2_series", how="left")

    # reorder columns
    col_union = plate_cols.tolist() + well_cols.tolist()
    col_u = []
    [col_u.append(col) for col in col_union if col not in col_u]
    well_df = well_df.loc[:, col_u]

    ################
    # Finally, add curation info
    curation_df_long, curation_df_wide = parse_curation_metadata(root, experiment_date)
    if curation_df_long is not None:
        well_df = well_df.merge(curation_df_long, on=["nd2_series", "time_index"], how="left")
    well_df["estimated_stage_hpf"] = well_df["start_age_hpf"] + well_df["time"]/3600

    # save
    well_df.to_csv(os.path.join(root, "metadata", "frame_metadata", experiment_date + "_master_metadata_df.csv"), index=False)

    return metadata

if __name__ == "__main__":

    # set path to CellPose model to use
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\pecfin_dynamics\\"
    experiment_date = "20240223"


    extract_frame_metadata(root=root, experiment_date=experiment_date)
