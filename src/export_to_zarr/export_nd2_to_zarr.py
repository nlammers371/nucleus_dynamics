from aicsimageio import AICSImage
import zarr
import numpy as np
import napari
from skimage.transform import resize
import glob2 as glob
import sys
import threading
import scipy.ndimage as ndi
import os
import nd2
import dask.array as da
from tqdm import tqdm
from src.utilities.register_image_stacks import register_timelapse
from src.utilities.extract_frame_metadata import extract_frame_metadata

def export_nd2_to_zarr(root,experiment_date, overwrite_flag, metadata_only=False, nuclear_channel=None, channel_names=None):

    out_dir = os.path.join(root, "built_data", "zarr_image_files", experiment_date, "")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    metadata = extract_frame_metadata(root, experiment_date)

    if not metadata_only:
        metadata["nuclear_channel"] = nuclear_channel
        if channel_names is not None:
            metadata["channel_names"] = channel_names
        # extract key metadata info
        nd2_files = glob.glob(os.path.join(root, "raw_data", experiment_date, "*nd2"))

        imObject = nd2.ND2File(nd2_files[0])
        im_array_dask = imObject.to_dask()
        nd2_shape = im_array_dask.shape

        n_time_points = metadata["n_time_points"]
        n_wells = metadata["n_wells"]
        n_channels = metadata["n_channels"]

        well_shape = tuple([nd2_shape[0]]) + nd2_shape[2:]
        if len(nd2_shape) == 4:
            im_array_dask = im_array_dask[None, :, :, :, :]
            nd2_shape = im_array_dask.shape
            well_shape = tuple([nd2_shape[0]]) + nd2_shape[2:]

        elif len(nd2_shape) == 5 and n_channels == 2:
            im_array_dask = da.transpose(im_array_dask, (0, 2, 1, 3, 4))
            im_array_dask = im_array_dask[None, :, :, :, :, :]
            nd2_shape = im_array_dask.shape
            well_shape = tuple([nd2_shape[2]]) + tuple([nd2_shape[0]]) + nd2_shape[3:]

        # get list of metadata keys to pass to wll zarrs
        meta_keys = list(metadata.keys())
        meta_keys = [key for key in meta_keys if key!="n_wells"]
        multichannel_flag = n_channels > 1
        if multichannel_flag and nuclear_channel is None:
            raise ValueError("nuclear channel must be provided for multichannel experiments")

        dtype = im_array_dask.dtype

        for well_num in tqdm(range(n_wells), "Exporting well time series to zarr..."):

            # initialize zarr data store
            filename = experiment_date + f"_well{well_num:04}.zarr"
            zarr_file = os.path.join(out_dir, filename)

            # check for existing zarr file
            prev_flag = os.path.isdir(zarr_file)
            # Initialize zarr array
            if not multichannel_flag:
                well_zarr = zarr.open(zarr_file, mode='a', shape=well_shape, dtype=dtype, chunks=(1,) + nd2_shape[2:])
            else:
                well_zarr = zarr.open(zarr_file, mode='a', shape=well_shape, dtype=dtype, chunks=tuple([1, 1]) + tuple(well_shape[-3:]))

            # add metadata
            for key in meta_keys:
                well_zarr.attrs[key] = metadata[key]

            # check for pre-existing data
            if overwrite_flag | (not prev_flag):
                write_indices = np.arange(n_time_points)
            else:
                write_indices = []
                for t in tqdm(range(n_time_points), "Checking which frames to segment..."):
                    if multichannel_flag:
                        nz_flag_to = np.any(well_zarr[0, t] != 0)
                    else:
                        nz_flag_to = np.any(well_zarr[t] != 0)
                    if not nz_flag_to:  # if the cellpose output is all zeros
                        write_indices.append(t)

            for t, ti in tqdm(enumerate(write_indices), "Writing to zarr..."):
                if not multichannel_flag:
                    data_zyx = im_array_dask[ti, well_num, :, :, :].compute()
                    well_zarr[ti] = data_zyx
                else:
                    for chi in range(n_channels):
                        data_zyx = np.squeeze(im_array_dask[ti, well_num, chi, :, :, :].compute())
                        well_zarr[chi, ti] = data_zyx

            # register frames
            # if not multichannel_flag:
            #     _, shift_array = register_timelapse(well_zarr)
            # else:
            #     _, shift_array = register_timelapse(np.squeeze(well_zarr[nuclear_channel]))
            #
            # # apply shifts
            # for t in tqdm(range(1, well_zarr.shape[1]), "Registering image data..."):
            #
            #     if not multichannel_flag:
            #         well_zarr[t] = ndi.shift(well_zarr[t], (shift_array[t, :]), order=1)
            #     else:
            #         for chi in range(n_channels):
            #             well_zarr[chi, t] = ndi.shift(np.squeeze(well_zarr[chi, t]), (shift_array[t, :]), order=1)
            #
            # print("check")

        imObject.close()

if __name__ == '__main__':
    experiment_date_vec = ["20240619", "20240620"]
    root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/"
    overwrite_flag = True
    nuclear_channel = 0
    channel_names = ["H2B-tdTom"]
    for experiment_date in experiment_date_vec:
        export_nd2_to_zarr(root, experiment_date, overwrite_flag, nuclear_channel=0, channel_names=channel_names)