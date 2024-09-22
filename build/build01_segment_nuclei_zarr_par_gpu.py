"""
Image segmentation via Cellpose library
"""
import logging
import glob2 as glob
import os
import time
# import pyclesperanto as cle
from typing import Any
from typing import Dict
from typing import Literal
from typing import Optional
import numpy as np
import torch.cuda
from cellpose import models
from cellpose.core import use_gpu
import pandas as pd
from skimage.transform import resize
from utilities.functions import path_leaf
import zarr
from utilities.image_utils import calculate_LoG
from functools import partial
from tqdm.contrib.concurrent import process_map

# logging = logging.getlogging(__name__)
logging.basicConfig(level=logging.NOTSET)
# __OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__

def process_time_point(t, t0, data_tzyx, model_name, mask_zarr, prob_zarr, grad_zarr, anisotropy_raw, cell_diameter,
                       flow_threshold, model_type, min_size, zarr_path, device_list=["cpu"]):

    device = device_list[t-t0]

    # extract image
    data_zyx_raw = data_tzyx[t]

    if np.any(data_zyx_raw > 0):
        # rescale data
        dims_orig = data_zyx_raw.shape
        if xy_ds_factor > 1.0:
            dims_new = np.round([dims_orig[0], dims_orig[1] / xy_ds_factor, dims_orig[2] / xy_ds_factor]).astype(int)
            data_zyx = resize(data_zyx_raw, dims_new, order=1)
        else:
            dims_new = dims_orig
            data_zyx = data_zyx_raw.copy()

        if ("log" in model_name) or ("bkg" in model_name):
            im_log, im_bkg = calculate_LoG(data_zyx=data_zyx, scale_vec=pixel_res_raw)
        if "log" in model_name:
            data_zyx = im_log
        elif "bkg" in model_name:
            print("bkg")
            data_zyx = im_bkg

        anisotropy = anisotropy_raw * dims_new[1] / dims_orig[1]

        # Select 2D/3D behavior and set some parameters
        do_3D = data_zyx.shape[0] > 1

        # Preliminary checks on Cellpose model
        if pretrained_model is None:
            if model_type not in ["nuclei", "cyto2", "cyto"]:
                raise ValueError(f"ERROR model_type={model_type} is not allowed.")
        else:
            if not os.path.exists(pretrained_model):
                raise ValueError(f"{pretrained_model=} does not exist.")

        logging.info(
            f"mask will have shape {data_zyx.shape} "
        )

        # Initialize cellpose
        gpu = use_gpu()
        if pretrained_model:
            model = models.CellposeModel(
                gpu=gpu, pretrained_model=pretrained_model, device=device
            )
        else:
            model = models.CellposeModel(gpu=gpu, model_type=model_type, device=device)

        # Initialize other things
        logging.info(f"Start cellpose_segmentation task for {zarr_path}")
        logging.info(f"do_3D: {do_3D}")
        logging.info(f"use_gpu: {gpu}")
        logging.info(f"model_type: {model_type}")
        logging.info(f"pretrained_model: {pretrained_model}")
        logging.info(f"anisotropy: {anisotropy}")

        # Execute illumination correction
        image_mask, image_probs, image_grads = segment_FOV(
            data_zyx,  # data_zyx.compute(),
            model=model,
            do_3D=do_3D,
            anisotropy=anisotropy,
            label_dtype=np.uint32,
            diameter=cell_diameter / xy_ds_factor,
            cellprob_threshold=cellprob_threshold,
            flow_threshold=flow_threshold,
            min_size=min_size,
            pretrain_flag=(pretrained_model != None)
        )

        if xy_ds_factor > 1.0:
            image_mask_out = resize(image_mask, dims_orig, order=0, anti_aliasing=False, preserve_range=True)
            image_probs_out = resize(image_probs, dims_orig, order=1)
            image_grads_out = resize(image_grads, (3,) + dims_orig, order=1)
        else:
            image_mask_out = image_mask
            image_probs_out = image_probs
            image_grads_out = image_grads

        mask_zarr[t] = image_mask_out
        prob_zarr[t] = image_probs_out
        grad_zarr[t] = image_grads_out

        logging.info(f"End file save process, exit")

    else:
        logging.info(f"No image data found. Skipping time point.")
def segment_FOV(
    column: np.ndarray,
    model=None,
    do_3D: bool = True,
    anisotropy=None,
    diameter: float = 15,
    cellprob_threshold: float = 0.0,
    flow_threshold: float = 0.4,
    min_size=None,
    label_dtype=None,
    pretrain_flag=False
):
    """
    Internal function that runs Cellpose segmentation for a single ROI.

    :param column: Three-dimensional numpy array
    :param model: TBD
    :param do_3D: TBD
    :param anisotropy: TBD
    :param diameter: TBD
    :param cellprob_threshold: TBD
    :param flow_threshold: TBD
    :param min_size: TBD
    :param label_dtype: TBD
    """

    # Write some debugging info
    logging.info(
        f"[segment_FOV] START Cellpose |"
        f" column: {type(column)}, {column.shape} |"
        f" do_3D: {do_3D} |"
        f" model.diam_mean: {model.diam_mean} |"
        f" diameter: {diameter} |"
        f" flow threshold: {flow_threshold}"
    )

    # Actual labeling
    t0 = time.perf_counter()
    if not pretrain_flag:
        mask, flows, styles = model.eval(
            column,
            channels=[0, 0],
            do_3D=do_3D,
            net_avg=False,
            augment=False,
            diameter=diameter,
            anisotropy=anisotropy,
            cellprob_threshold=cellprob_threshold,
            flow_threshold=flow_threshold,
        )
    else:
        mask, flows, styles = model.eval(
            column,
            channels=[0, 0],
            do_3D=do_3D,
            min_size=min_size,
            diameter=diameter,
            anisotropy=anisotropy,
            cellprob_threshold=cellprob_threshold,
            # net_avg=False,
            augment=False
        )
    if not do_3D:
        mask = np.expand_dims(mask, axis=0)
    t1 = time.perf_counter()

    # Write some debugging info
    logging.info(
        f"[segment_FOV] END   Cellpose |"
        f" Elapsed: {t1-t0:.4f} seconds |"
        f" mask shape: {mask.shape},"
        f" mask dtype: {mask.dtype} (before recast to {label_dtype}),"
        f" max(mask): {np.max(mask)} |"
        f" model.diam_mean: {model.diam_mean} |"
        f" diameter: {diameter} |"
        f" anisotropy: {anisotropy} |"
        f" flow threshold: {flow_threshold}"
    )

    probs = flows[2]
    grads = flows[1]

    return mask.astype(label_dtype), probs, grads


def cellpose_segmentation(
    *,
    # Fractal arguments
    root: str,
    experiment_date: str,
    cell_diameter: float = 30,
    cellprob_threshold: float = 0,
    flow_threshold: float = 0.4,
    model_type: Literal["nuclei", "cyto", "cyto2"] = "nuclei",
    pretrained_model: Optional[str] = None,
    overwrite: Optional[bool] = False,
    xy_ds_factor: Optional[float] = 1.0,
    nuclear_channel: int = 0,
) -> Dict[str, Any]:
    """
    Run cellpose segmentation on the ROIs of a single OME-NGFF image

    Full documentation for all arguments is still TBD, especially because some
    of them are standard arguments for Fractal tasks that should be documented
    in a standard way. Here are some examples of valid arguments::

        input_paths = ["/some/path/*.zarr"]
        component = "some_plate.zarr/B/03/0"
        metadata = {"num_levels": 4, "coarsening_xy": 2}

    :param data_directory: path to directory containing zarr folders for images to segment
    :param seg_channel_label: Identifier of a channel based on its label (e.g.
                          ``DAPI``). If not ``None``, then ``wavelength_id``
                          must be ``None``.
    :param cell_diameter: Initial diameter to be passed to
                            ``CellposeModel.eval`` method (after rescaling from
                            full-resolution to ``level``).
    :param output_label_name: output name for labels
    :param cellprob_threshold: Parameter of ``CellposeModel.eval`` method.
    :param flow_threshold: Parameter of ``CellposeModel.eval`` method.
    :param model_type: Parameter of ``CellposeModel`` class.
    :param pretrained_model: Parameter of ``CellposeModel`` class (takes
                             precedence over ``model_type``).
    """

    # Read useful parameters from metadata
    min_size = 5  # let's be conservative here     # (cell_diameter/4)**3 / xy_ds_factor**2

    # check for available gpus
    n_gpu = torch.cuda.device_count()
    gpu_list = []
    t_increment = 1
    if n_gpu == 0:
        gpu_list = ["cpu"]
    elif n_gpu == 1:
        gpu_list = ["cuda"]
    elif n_gpu == 2:
        gpu_list = ["cuda:0", "cuda:1"]
        t_increment = 2

    # path to zarr files
    data_directory = os.path.join(root, "built_data", "zarr_image_files", experiment_date, '')

    # load curation data
    curation_path = os.path.join(root, "metadata", "curation", experiment_date + "_curation_info.csv")
    has_curation_info = os.path.isfile(curation_path)
    if has_curation_info:
        curation_df = pd.read_csv(curation_path)
        curation_df_long = pd.melt(curation_df,
                                   id_vars=["series_number", "notes", "tbx5a_flag", "follow_up_flag"],
                                   var_name="time_index", value_name="qc_flag")
        curation_df_long["time_index"] = curation_df_long["time_index"].astype(int)

    model_name = path_leaf(pretrained_model)
    save_directory = os.path.join(root, "built_data", "cellpose_output", model_name, experiment_date, '')
    if not os.path.isdir(save_directory):
        os.makedirs(save_directory)

    # get list of images
    image_list = sorted(glob.glob(data_directory + "*.zarr"))

    for well_index in range(0, len(image_list)):

        zarr_path = image_list[well_index]
        im_name = path_leaf(zarr_path)
        print("processing " + im_name)
        # read the image data
        data_tzyx = zarr.open(zarr_path, mode="r")
        if len(data_tzyx.shape) == 5:
            data_tzyx = data_tzyx[nuclear_channel]
        # n_time_points = data_tzyx.shape[0]

        # make sure we are not accidentally up-sampling
        assert xy_ds_factor >= 1.0
        pixel_res_raw = data_tzyx.attrs["voxel_size_um"]
        anisotropy_raw = pixel_res_raw[0] / pixel_res_raw[1]

        # generate zarr files
        file_prefix = experiment_date + f"_well{well_index:04}"
        mask_zarr_path = os.path.join(save_directory, file_prefix + "_labels.zarr")
        prev_flag = os.path.isdir(mask_zarr_path)

        mask_zarr = zarr.open(mask_zarr_path, mode='a', shape=data_tzyx.shape, dtype=np.uint16, chunks=(1,) + data_tzyx.shape[1:])

        grad_zarr_path = os.path.join(save_directory, file_prefix + "_grads.zarr")
        grad_zarr = zarr.open(grad_zarr_path, mode='a',
                              shape=(data_tzyx.shape[0], 3, data_tzyx.shape[1], data_tzyx.shape[2], data_tzyx.shape[3]),
                              dtype=np.float32, chunks=(1, 3,) + data_tzyx.shape[1:])


        prob_zarr_path = os.path.join(save_directory, file_prefix + "_probs.zarr")
        prob_zarr = zarr.open(prob_zarr_path, mode='a', shape=data_tzyx.shape, dtype=np.float32,
                              chunks=(1,) + data_tzyx.shape[1:])

        # transfer metadata from raw data to cellpose products
        meta_keys = data_tzyx.attrs.keys()
        for meta_key in meta_keys:
            prob_zarr.attrs[meta_key] = data_tzyx.attrs[meta_key]
            mask_zarr.attrs[meta_key] = data_tzyx.attrs[meta_key]
            grad_zarr.attrs[meta_key] = data_tzyx.attrs[meta_key]

        prob_zarr.attrs["model_path"] = pretrained_model
        mask_zarr.attrs["model_path"] = pretrained_model
        grad_zarr.attrs["model_path"] = pretrained_model

        # if has_curation_info:
        #     time_indices0 = curation_df_long.loc[(curation_df_long['series_number'] == well_index) & \
        #                                          (curation_df_long['qc_flag'] < 2), 'time_index'].values

        time_indices0 = range(data_tzyx.shape[0])

        # determine which indices to segment
        if overwrite | (not prev_flag):
            write_indices = time_indices0
        else:
            write_indices = []
            for t in time_indices0:
                nz_flag = np.any(prob_zarr[t, :, :, :] != 0)
                if not nz_flag:
                    write_indices.append(t)
            write_indices = np.asarray(write_indices)

        if t_increment == 2:
            write_indices = write_indices[::2]

        for t in write_indices:

            if n_gpu == 2:
                emb_df_temp = process_map(
                    partial(process_time_point, t, data_tzyx, model_name, mask_zarr, prob_zarr, grad_zarr, anisotropy_raw, cell_diameter,
                           flow_threshold, model_type, min_size, zarr_path, device_list=gpu_list),
                    [t, t+1], max_workers=8, chunksize=1)
            else:
                process_time_point(t, t, data_tzyx, model_name, mask_zarr, prob_zarr, grad_zarr, anisotropy_raw, cell_diameter,
                    flow_threshold, model_type, min_size, zarr_path, device_list = gpu_list)



    return {}

if __name__ == "__main__":
    # s0rt some hyperparameters
    overwrite = False
    xy_ds_factor = 1
    cell_diameter = 10
    cellprob_threshold = 0.0
    pixel_res_raw = [2, 0.55, 0.55]

    # set path to CellPose model to use
    pretrained_model = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/built_data/cellpose_training/20240424_tdTom/log/models/log-v3"

    # set read/write paths
    root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/"
    experiment_date_vec = ["20240424", "20240425"]

    for experiment_date in experiment_date_vec:
        cellpose_segmentation(root=root, experiment_date=experiment_date, pixel_res_raw=pixel_res_raw,
                              xy_ds_factor=xy_ds_factor, cell_diameter=cell_diameter, nuclear_channel=1,
                              cellprob_threshold=cellprob_threshold, pretrained_model=pretrained_model, overwrite=overwrite)
