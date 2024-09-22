import napari
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from ultrack import MainConfig, load_config, track, to_tracks_layer, tracks_to_zarr
import glob2 as glob
import zarr
import skimage.io as io
from skimage.transform import resize
import json
import nd2

# # set parameters
# root = "E:/Nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/fin_morphodynamics/"
root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/fin_morphodynamics/"
experiment_date = "20240223"
config_name = "tracking_jordao_frontier.txt"
model ="log-v5"
tracking_folder = config_name.replace(".txt", "")
tracking_folder = tracking_folder.replace(".toml", "")

well_num = 12
start_i = 0
stop_i = 101
scale_vec = np.asarray([2.0, 0.55, 0.55])
centroid_flag = False
label_spacer = False

suffix = ""
if centroid_flag:
    suffix = "_centroid"

if label_spacer:
    suffix += "_spacer"

# get path to metadata
metadata_path = os.path.join(root, "metadata", "tracking")

# set output path for tracking results
project_path = os.path.join(root, "tracking", experiment_date,  tracking_folder, f"well{well_num:04}" + suffix, "")

# path to image data
data_path = os.path.join(root, "built_data", "cellpose_output", model, experiment_date, "")
filename = experiment_date + f"_well{well_num:04}_probs.zarr"

# load tracking results
image_path = os.path.join(data_path, filename)
label_path = os.path.join(project_path, "segments.zarr")
data_zarr = zarr.open(image_path, mode='r')
if well_num == 12:
    data_zarr = data_zarr[start_i:stop_i, :, 400:850, 80:520]
elif well_num == 2:
    data_zarr = data_zarr[start_i:stop_i, :, 300:700, 100:500]


seg_zarr = zarr.open(label_path, mode='r')

cfg = load_config(os.path.join(metadata_path, config_name))
# tracks_df, graph = to_tracks_layer(cfg)
tracks_df = pd.read_csv(os.path.join(project_path, "tracks.csv"))

viewer = napari.view_image(data_zarr, scale=tuple(scale_vec))


viewer.add_tracks(
    tracks_df[["track_id", "t", "z", "y", "x"]],
    name="tracks",
    scale=tuple(scale_vec),
    translate=(0, 0, 0, 0),
    visible=False,
)


viewer.add_labels(
    seg_zarr,
    name="segments",
    scale=tuple(scale_vec),
    translate=(0, 0, 0, 0),
).contour = 2

# filter for the best tracks
track_index, track_counts = np.unique(tracks_df["track_id"], return_counts=True)
min_len = 40
good_tracks = track_index[track_counts >= min_len]


tracks_df_qc = tracks_df.loc[np.isin(tracks_df["track_id"], good_tracks), :]
seg_qc = np.asarray(seg_zarr).copy()
seg_qc[~np.isin(seg_qc, good_tracks)] = 0


viewer.add_tracks(
    tracks_df_qc[["track_id", "t", "z", "y", "x"]],
    name="tracks qc",
    scale=tuple(scale_vec),
    translate=(0, 0, 0, 0),
    visible=False,
)


viewer.add_labels(
    seg_qc,
    name="segments qc",
    scale=tuple(scale_vec),
    translate=(0, 0, 0, 0),
).contour = 2


viewer.window.add_plugin_dock_widget(plugin_name='napari-animation')

if __name__ == '__main__':
    napari.run()