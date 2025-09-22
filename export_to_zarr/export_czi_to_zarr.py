import numpy as np
import json
import os
import glob2 as glob
from skimage.transform import resize
from tqdm import tqdm
from skimage import io
from src.utilities.functions import path_leaf
from tqdm.contrib.concurrent import process_map
from functools import partial
import zarr
import re
from bioio import BioImage
import bioio_ome_zarr
import bioio_czi

import dask


_CHUNK_KEY_RE = re.compile(r'^(\d+)\..*$')  # capture leading time index

def _existing_time_indices(zarr_dir):
    """Return a set of time indices that already have at least one chunk written."""
    seen = set()
    try:
        for name in os.listdir(zarr_dir):
            m = _CHUNK_KEY_RE.match(name)
            if m:
                seen.add(int(m.group(1)))
    except FileNotFoundError:
        # No store yet
        pass
    return seen

def get_prefix_list(raw_data_root):
    image_list = sorted(glob.glob(os.path.join(raw_data_root, f"*.czi")))
    stripped_names = [re.sub(r"\(.*\).czi", "", os.path.basename(p)) for p in image_list]
    prefix_list = np.unique(stripped_names).tolist()
    prefix_list = [p for p in prefix_list if p != ""]
    return prefix_list


def initialize_zarr_store(zarr_path, image_list, resampling_scale,
                          channel_to_use=None, channels_to_keep=None,
                          overwrite_flag=False, last_i=None):

    imObject = BioImage(image_list[0])
    image_data = np.squeeze(imObject.data)  # Could be (C, Z, Y, X) or (Z, Y, X)
    multichannel_flag = len(image_data.shape) == 4

    # Apply channel filtering before shape calc
    if multichannel_flag and channels_to_keep is not None:
        image_data = image_data[np.array(channels_to_keep, dtype=bool)]
        if image_data.shape[0] == 1:
            image_data = np.squeeze(image_data[0])
        multichannel_flag = len(image_data.shape) == 4
    elif multichannel_flag and channel_to_use is not None:
        image_data = np.squeeze(image_data[np.asarray(channel_to_use)])
        multichannel_flag = len(image_data.shape) == 4


    raw_scale_vec = np.asarray(imObject.physical_pixel_sizes)
    if np.max(raw_scale_vec) <= 1e-5:
        raw_scale_vec = raw_scale_vec * 1e6  # assume input was meters → convert to microns

    dims_orig = image_data.shape
    rs_factors = np.divide(raw_scale_vec, resampling_scale)

    # Apply channel filtering before shape calc
    if not multichannel_flag:
        # dims_orig: (Z,Y,X)
        out_spatial = tuple(np.round(np.multiply(dims_orig, rs_factors)).astype(int))
        inner_shape = out_spatial
        chunks = (1,) + inner_shape
    else:
        # dims_orig: (C,Z,Y,X)
        out_spatial = tuple(np.round(np.multiply(dims_orig[1:], rs_factors)).astype(int))
        inner_shape = (dims_orig[0],) + out_spatial  # (C, Z, Y, X)
        chunks = (1, 1) + inner_shape[1:]

    # if not multichannel_flag:
    #     shape = tuple(np.round(np.multiply(dims_orig, rs_factors)).astype(int))
    # else:
    #     shape = (dims_orig[0],) + tuple(np.round(np.multiply(dims_orig[1:], rs_factors)).astype(int))
    T = len(image_list) if last_i is None else int(last_i)
    shape = (T,) + inner_shape
    dtype = np.uint16
    mode = "w" if overwrite_flag else "a"

    zarr_file = zarr.open(
        zarr_path,
        mode=mode,
        shape=shape,
        dtype=dtype,
        chunks=chunks
    )

    # Figure out which timepoints still need writing
    if overwrite_flag:
        indices_to_write = list(range(T))
    else:
        # Scan the chunk filenames in the root of the array directory
        already = _existing_time_indices(zarr_path)
        # Clip to [0, T-1] just in case the directory has leftovers from old shapes
        already = {i for i in already if 0 <= i < T}
        indices_to_write = sorted(set(range(T)) - already)

    return zarr_file, indices_to_write


def write_zarr(t, zarr_file, image_list, file_prefix, tres, resampling_scale,
               channel_names=None, channels_to_keep=None):

    f_string = path_leaf(image_list[t])
    time_string = f_string.replace(file_prefix, "")
    time_string = time_string.replace(".czi", "")
    time_point = int(time_string[1:-1]) - 1


    imObject = BioImage(image_list[t])
    image_data = np.squeeze(imObject.data)

    # Apply channel filtering if multichannel
    multichannel_flag = len(image_data.shape) == 4
    if multichannel_flag and channels_to_keep is not None:
        image_data = image_data[np.array(channels_to_keep, dtype=bool)]
        if image_data.shape[0] == 1:
            image_data = np.squeeze(image_data[0])

    shape = zarr_file.shape
    frame_shape = shape[1:]
    multichannel_flag = len(shape) > 4

    if not multichannel_flag:
        image_data_rs = np.round(resize(image_data, frame_shape,
                                        preserve_range=True, order=1,
                                        anti_aliasing=True)).astype(np.uint16)
    else:
        image_data_rs = np.empty(frame_shape, dtype=np.uint16)
        for c in range(frame_shape[0]):
            image_data_rs[c] = np.round(resize(image_data[c], frame_shape[1:],
                                               preserve_range=True, order=1,
                                               anti_aliasing=True)).astype(np.uint16)

    if t == 0:
        if channel_names is None:
            channel_names = [f"channel{c:02}" for c in range(frame_shape[0])]
        n_time_points = len(image_list)
        project_name = path_leaf(image_list[t]).replace(".czi", "")
        metadata = {
            "DimOrder": "tzyx",
            "Dims": [n_time_points, image_data_rs.shape[0], image_data_rs.shape[1], image_data_rs.shape[2]]
                    if multichannel_flag else
                    [n_time_points, image_data_rs.shape[0], image_data_rs.shape[1]],
            "TimeRes": tres,
            "PhysicalSizeX": resampling_scale[2],
            "PhysicalSizeY": resampling_scale[1],
            "PhysicalSizeZ": resampling_scale[0],
            "pixel_scale_um": resampling_scale.tolist(),
            "ProjectName": project_name,
            "Channels": channel_names
        }
        for key, val in metadata.items():
            zarr_file.attrs[key] = val

    zarr_file[time_point] = image_data_rs


def export_czi_to_zarr(raw_data_root, file_prefix, project_name, save_root, tres,
                       par_flag=True, last_i=None, overwrite_flag=False,
                       resampling_scale=None, channel_names=None,
                       channel_to_use=None, channels_to_keep=None,
                       n_workers=8):

    if resampling_scale is None:
        resampling_scale = np.asarray([1.5, 1.5, 1.5])

    if channels_to_keep is not None:
        if channel_names is None:
            raise ValueError("channel_names must be provided if channels_to_keep is used.")
        if len(channels_to_keep) != len(channel_names):
            raise ValueError("channels_to_keep must match length of channel_names.")
        channel_names = [ch for ch, keep in zip(channel_names, channels_to_keep) if keep]

    zarr_path = os.path.join(save_root, "built_data", "zarr_image_files", project_name + '.zarr')
    if not os.path.isdir(zarr_path):
        os.makedirs(zarr_path)

    image_list = sorted(glob.glob(os.path.join(raw_data_root, file_prefix + f"(*).czi")))

    zarr_file, times_to_write = initialize_zarr_store(zarr_path, image_list,
                                      resampling_scale=resampling_scale,
                                      channel_to_use=channel_to_use,
                                      channels_to_keep=channels_to_keep,
                                      overwrite_flag=overwrite_flag,
                                      last_i=last_i)

    times_to_write = np.asarray(times_to_write)
    if last_i is not None:
        times_to_write = times_to_write[times_to_write <= last_i]

    # get list of image timestamps and use this to firgure out which indices to write
    image_time_stamps = []
    for _, image_path in enumerate(image_list):
        f_string = path_leaf(image_path)
        time_string = f_string.replace(file_prefix, "")
        time_string = time_string.replace(".czi", "")
        image_time_stamps.append(int(time_string[1:-1]) - 1)

    indices_to_write = np.where(np.isin(np.asarray(image_time_stamps), times_to_write))[0]

    if par_flag:
        process_map(
            partial(write_zarr, zarr_file=zarr_file, image_list=image_list,
                    channel_names=channel_names,
                    channels_to_keep=channels_to_keep,
                    file_prefix=file_prefix, tres=tres,
                    resampling_scale=resampling_scale),
            indices_to_write, max_workers=n_workers, chunksize=1
        )
    else:
        for i in tqdm(indices_to_write, "Exporting raw images to zarr..."):
            write_zarr(i, zarr_file=zarr_file, image_list=image_list, channel_names=channel_names,
                       channels_to_keep=channels_to_keep,
                       file_prefix=file_prefix, tres=tres,
                       resampling_scale=resampling_scale)

    print("Done.")



if __name__ == "__main__":

    # da_chunksize = (1, 207, 256, 256)
    resampling_scale = np.asarray([1.5, 1.5, 1.5])
    tres = 123.11  # time resolution in seconds

    # set path parameters
    raw_data_root = "D:\\Syd\\240611_EXP50_NLS-Kikume_24hpf_2sided_NuclearTracking\\" #"D:\\Syd\\240219_LCP1_67hpf_to_"
    file_prefix_vec = ["E2_2024_11_14__20_21_18_968_G1", "E2_Timelapse_2024_06_11__22_51_41_085_G2"] #"E3_186_TL_start93hpf_2024_02_20__19_13_43_218"

    # Specify the path to the output OME-Zarr file and metadata file
    save_root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
    project_name_vec = ["20240611_NLS-Kikume_24hpf_side1", "20240611_NLS-Kikume_24hpf_side2"]
    overwrite = False

    for i in range(len(project_name_vec)):
        file_prefix = file_prefix_vec[i]
        project_name = project_name_vec[i]
        export_czi_to_zarr(raw_data_root, file_prefix, project_name, save_root, tres, par_flag=True,
                           channel_to_use=0, overwrite_flag=overwrite)