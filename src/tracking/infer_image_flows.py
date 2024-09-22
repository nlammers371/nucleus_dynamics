import napari
import os
import numpy as np
from ultrack import MainConfig, load_config, track, to_tracks_layer, tracks_to_zarr
import zarr
from ultrack.utils import estimate_parameters_from_labels, labels_to_edges
from ultrack.imgproc.flow import timelapse_flow, advenct_from_quasi_random, trajectories_to_tracks
import torch
import shutil
import pandas as pd
from cellpose.core import use_gpu
from ultrack.utils.cuda import import_module, torch_default_device
from ultrack.tracks.stats import tracks_df_movement


def infer_flows(root, experiment_date, well_num, start_i=0, stop_i=None, overwrite_flag=False):


    # get path to zarr file
    file_prefix = experiment_date + f"_well{well_num:04}"

    # set parameters
    data_zarr_path = os.path.join(root, "built_data", "zarr_image_files", experiment_date, file_prefix + ".zarr")
    flow_zarr_path = os.path.join(root, "built_data", "flow", experiment_date, file_prefix + ".zarr")
    # mask_zarr = os.path.join(root, "built_data", "stitched_labels", seg_model, experiment_date, file_prefix + "_labels_stitched.zarr")

    data_tzyx_full = zarr.open(data_zarr_path, mode='r')
    # mask_tzyx_full = zarr.open(mask_zarr, mode='r')
    if stop_i is None:
        stop_i = data_tzyx_full.shape[0]

    # specify time points to load
    data_tzyx = data_tzyx_full[start_i:stop_i]

    # infer flows
    if not use_gpu():
        device = torch.device("cpu")
    else:
        device = torch_default_device()

    if os.path.isdir(flow_zarr_path):
        shutil.rmtree(flow_zarr_path)
    flows = timelapse_flow(data_tzyx, store_or_path=flow_zarr_path, n_scales=3, lr=1e-2, num_iterations=2_000,
                           device=device)

    # trajectory = advenct_from_quasi_random(flows, data_tzyx.shape[-3:], n_samples=1000)
    # flow_tracklets = pd.DataFrame(
    #     trajectories_to_tracks(trajectory),
    #     columns=["track_id", "t", "z", "y", "x"],
    # )
    #
    # flow_tracklets[["z", "y", "x"]] += 0.5  # napari was crashing otherwise, might be an openGL issue
    # flow_tracklets[["dz", "dy", "dx"]] = tracks_df_movement(flow_tracklets)
    # flow_tracklets["angles"] = np.arctan2(flow_tracklets["dy"], flow_tracklets["dx"])
    #
    # viewer.add_tracks(
    #     flow_tracklets[["track_id", "t", "z", "y", "x"]],
    #     name="flow vectors",
    #     visible=True,
    #     tail_length=25,
    #     features=flow_tracklets[["angles", "dy", "dx"]],
    #     colormap="hsv",
    # ).color_by = "angles"

    return flows


if __name__ == '__main__':

    scale_vec = np.asarray([2.0, 0.55, 0.55])
    experiment_date = "20240223"
    root = "E:/Nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/fin_morphodynamics/"
    overwrite_flag = True
    well_num = 12
    infer_flows(root, experiment_date, well_num, start_i=0, stop_i=187, overwrite_flag=overwrite_flag)
