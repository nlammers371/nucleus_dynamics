import numpy as np
import json
import os
import glob2 as glob
from aicsimageio import AICSImage
from skimage.transform import resize
from tqdm import tqdm
from skimage import io
from src.utilities.functions import path_leaf
from tqdm.contrib.concurrent import process_map
from functools import partial
import zarr
from src.utilities.register_image_stacks import register_timelapse
import dask
# def create_dummy_data(shape):
#     return da.random.random(shape, chunks=(100, 100, 100))

def initialize_zarr_store(zarr_path, image_list, resampling_scale, channel_to_use=None, overwrite_flag=False):

    # Load image
    imObject = AICSImage(image_list[0])
    image_data = np.squeeze(imObject.data)
    if (len(image_data.shape) == 4) and (channel_to_use is not None):
        image_data = np.squeeze(image_data[channel_to_use])
    raw_scale_vec = np.asarray(imObject.physical_pixel_sizes)
    if np.max(raw_scale_vec) <= 1e-5:    # check for weird issue with units
        raw_scale_vec = raw_scale_vec * 1e6
    dims_orig = image_data.shape
    rs_factors = np.divide(raw_scale_vec, resampling_scale)
    dims_new = np.round(np.multiply(dims_orig, rs_factors)).astype(int)

    shape = tuple(dims_new)
    dtype = np.uint16

    if overwrite_flag:
        zarr_file = zarr.open(zarr_path, mode='w', shape=(len(image_list),) + shape, dtype=dtype, chunks=(1,) + shape)
    else:
        zarr_file = zarr.open(zarr_path, mode='a', shape=(len(image_list),) + shape, dtype=dtype, chunks=(1,) + shape)

    return zarr_file


def write_zarr(t, zarr_file, image_list, overwrite_flag, file_prefix, tres, resampling_scale,
               channel_to_use=None):

    f_string = path_leaf(image_list[t])
    time_string = f_string.replace(file_prefix, "")
    time_string = time_string.replace(".czi", "")
    time_point = int(time_string[1:-1]) - 1
    # readPath = os.path.join(raw_data_root, file_prefix + f"({time_point}).czi")

    if (not np.any(zarr_file[time_point] > 0)) | overwrite_flag:
        # Load image
        imObject = AICSImage(image_list[t])
        image_data = np.squeeze(imObject.data)
        if (len(image_data.shape) == 4) and (channel_to_use is not None):
            image_data = np.squeeze(image_data[channel_to_use])

        raw_scale_vec = np.asarray(imObject.physical_pixel_sizes)
        if np.max(raw_scale_vec) <= 1e-5:
            raw_scale_vec = raw_scale_vec * 1e6
        # Resize
        dims_orig = image_data.shape
        rs_factors = np.divide(raw_scale_vec, resampling_scale)
        dims_new = np.round(np.multiply(dims_orig, rs_factors)).astype(int)
        image_data_rs = np.round(resize(image_data, dims_new, preserve_range=True, order=1)).astype(np.uint16)

        # Export the Dask array to the OME-Zarr file
        if t == 0:
            n_time_points = len(image_list)
            project_name = path_leaf(image_list[t])
            project_name = project_name.replace(".czi", "")
            metadata = {
                "DimOrder": "tzyx",
                "Dims": [n_time_points, image_data_rs.shape[0], image_data_rs.shape[1], image_data_rs.shape[2]],
                # Assuming there is only one timepoint per array, adjust if needed
                "TimeRes": tres,
                "PhysicalSizeX": resampling_scale[2],
                "PhysicalSizeY": resampling_scale[1],
                "PhysicalSizeZ": resampling_scale[0],
                "ProjectName": project_name
            }
            meta_keys = list(metadata.keys())
            meta_keys = [key for key in meta_keys if key != "n_wells"]
            for key in meta_keys:
                zarr_file.attrs[key] = metadata[key]

        zarr_file[time_point] = image_data_rs


def export_czi_to_zarr(raw_data_root, file_prefix, project_name, save_root, tres, par_flag=True, overwrite_flag=False,
                       resampling_scale=None, channel_to_use=0, n_workers=6):

    if resampling_scale is None:
        resampling_scale = np.asarray([1.5, 1.5, 1.5])

    zarr_path = os.path.join(save_root, "built_data", "zarr_image_files", project_name + '.zarr')

    if not os.path.isdir(zarr_path):
        os.makedirs(zarr_path)

    image_list = sorted(glob.glob(os.path.join(raw_data_root, file_prefix + f"(*).czi")))

    # Resize
    zarr_file = initialize_zarr_store(zarr_path, image_list, resampling_scale=resampling_scale,
                                      channel_to_use=channel_to_use, overwrite_flag=overwrite_flag)

    # Specify time index and pixel resolution
    # print("Exporting image arrays...")
    if par_flag:
        process_map(
            partial(write_zarr, zarr_file=zarr_file, image_list=image_list,
                                 overwrite_flag=overwrite_flag,
                                 file_prefix=file_prefix, tres=tres, resampling_scale=resampling_scale,
                                 channel_to_use=channel_to_use),
                    range(len(image_list)), max_workers=n_workers)
    else:
        for i in tqdm(range(len(image_list)), "Exporting raw images to zarr..."):
            write_zarr(i, zarr_file=zarr_file, image_list=image_list,
                                 overwrite_flag=overwrite_flag,
                                 file_prefix=file_prefix, tres=tres, resampling_scale=resampling_scale,
                                 channel_to_use=channel_to_use)

    # for t in range(len(image_list)):
    #     write_image(t, image_list=image_list, project_path=project_path,
    #                          overwrite_flag=overwrite_flag, metadata_file_path=metadata_file_path,
    #                          file_prefix=file_prefix, tres=tres, resampling_scale=resampling_scale)

    # Export each Dask array to the same OME-Zarr file one at a time
    # for t in tqdm(range(len(image_list))):


    print("Done.")


if __name__ == "__main__":

    # da_chunksize = (1, 207, 256, 256)
    resampling_scale = np.asarray([1.5, 1.5, 1.5])
    tres = 123.11  # time resolution in seconds

    # set path parameters
    # raw_data_root = "D:\\Syd\\231016_EXP40_LCP1_UVB_300mJ\\PreUVB_Timelapse_Raw\\"
    raw_data_root = "D:\\Syd\\240611_EXP50_NLS-Kikume_24hpf_2sided_NuclearTracking\\" #"D:\\Syd\\240219_LCP1_67hpf_to_"
    file_prefix_vec = ["E2_Timelapse_2024_06_11__22_51_41_085_G1", "E2_Timelapse_2024_06_11__22_51_41_085_G2"] #"E3_186_TL_start93hpf_2024_02_20__19_13_43_218"

    # Specify the path to the output OME-Zarr file and metadata file
    save_root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
    project_name_vec = ["20240611_NLS-Kikume_24hpf_side1", "20240611_NLS-Kikume_24hpf_side2"]
    overwrite = False

    for i in range(len(project_name_vec)):
        file_prefix = file_prefix_vec[i]
        project_name = project_name_vec[i]
        export_czi_to_zarr(raw_data_root, file_prefix, project_name, save_root, tres, par_flag=True,
                           channel_to_use=0, overwrite_flag=overwrite)