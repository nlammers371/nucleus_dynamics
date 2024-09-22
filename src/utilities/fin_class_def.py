import numpy as np
import pandas as pd
import os

import trimesh

from src.utilities.surface_axes_functions import *
from glob2 import glob
import pickle
import alphashape
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA
from scipy.spatial import distance_matrix
import itertools
from scipy.spatial import Delaunay

class FinData:
    def __init__(self, name: str = "", data_root: str = "", point_features: pd.DataFrame = None, tissue_seg_model: str = "",
                 seg_approved: bool=False, full_point_data: pd.DataFrame = None, full_point_data_raw: pd.DataFrame = None,
                 axis_body: pd.DataFrame=None, axis_body_approved: bool=False, axis_fin: pd.DataFrame = None,
                 axis_fin_approved: bool=False, yolk_surf_points: np.ndarray=None, yolk_surf_params: np.ndarray=None,
                 fin_surf_points: np.ndarray=None, fin_surf_faces: np.ndarray=None, fin_surf_params: np.ndarray=None,
                 fin_alpha_surf: trimesh.Trimesh=None, fin_core_points: np.ndarray=None, handle_scale: float=125):
        """
        Initialize the FinData class.

        Parameters:
        data (dict): A dictionary to hold financial data. Default is None.
        """
        self.name = name
        self.data_root = data_root
        self.point_features = point_features
        self.tissue_seg_model = tissue_seg_model
        self.seg_approved = seg_approved
        self.full_point_data = full_point_data
        self.full_point_data_raw = full_point_data_raw
        # self.fin_data = fin_data
        self.axis_fin = axis_fin
        self.axis_fin_approved = axis_fin_approved
        # self.axis_fin_array = axis_fin_array
        self.axis_body = axis_body
        self.axis_body_approved = axis_body_approved
        # self.axis_body_array = axis_body_array
        self.yolk_surf_points = yolk_surf_points
        self.yolk_surf_params = yolk_surf_params
        self.fin_surf_points = fin_surf_points
        self.fin_surf_faces = fin_surf_faces
        self.fin_surf_params = fin_surf_params
        self.fin_alpha_surf = fin_alpha_surf
        self.fin_core_points = fin_core_points
        
        # convenience features for curation
        self.handle_scale = handle_scale

        # check to see if a fin object already exists for this well
        self.save_path = os.path.join(self.data_root, self.data_root, "point_cloud_data", "fin_objects", "")
        self.save_name = self.name + "_fin_object.pkl"
        if os.path.exists(self.save_path + self.save_name):
            self.load_from_file()
        else:
            self.initialization_sequence()

    def load_from_file(self):
        # print("Loading saved pec fin data...")
        """
        Load the entire FinData object from a pickle file.

        Parameters:
        file_path (str): The path to the pickle file containing the serialized object.
        """
        try:
            with open(self.save_path + self.save_name, 'rb') as file:
                obj = pickle.load(file)
            self.__dict__.update(obj.__dict__)
        except FileNotFoundError:
            print(f"Error: The file {self.save_path} was not found.")
        except pickle.UnpicklingError:
            print(f"Error: The file {self.save_path} is not a valid pickle file.")

    def save_to_file(self):
        """
        Save the entire FinData object to the data_root file using pickle.
        """
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        try:
            with open(self.save_path + self.save_name, 'wb') as file:
                pickle.dump(self, file)
            print(f"Object saved to {self.save_path + self.save_name}")
        except Exception as e:
            print(f"Error: Failed to save object to {self.save_path}. {e}")

        # Save key features as separate files to have as backups
        backup_dir = os.path.join(self.save_path, "backup", self.name, "")
        if not os.path.isdir(backup_dir):
            os.makedirs(backup_dir)

        if self.seg_approved:
            self.full_point_data.to_csv(os.path.join(backup_dir, self.name + "_labels.csv"), index=False)
        if self.axis_fin_approved:
            self.axis_fin.to_csv(os.path.join(backup_dir, self.name + "_fin_axes.csv"), index=False)
        if self.axis_body_approved:
            self.axis_body.to_csv(os.path.join(backup_dir, self.name + "_body_axes.csv"), index=False)

    def initialization_sequence(self):
        # load point data and labels
        self.load_point_data()
        # use heuristics to flag outlier points in the fin class
        self.clean_fin_data()
        # calculate initial guess at fin alpha surface
        self.calculate_alpha_surf()
        # use PCA and heuristics to estimate body and fin axes
        self.estimate_body_axes()
        self.estimate_fin_axes()
        # fit surface
        self.fit_fin_surface()

    def curation_update_sequence(self):
        # sequence of functions run if a data attribute is manually updated
        self.calculate_alpha_surf()
        self.fit_fin_surface()

    def load_point_data(self):
        
        point_df_temp = pd.read_csv(os.path.join(self.data_root, "point_cloud_data", "point_features", self.tissue_seg_model, self.name + "_points_features.csv"))
        point_df_temp = strip_dummy_cols(point_df_temp)
        self.point_features = point_df_temp.copy()

        # check to see if manual tissue labels exist (otherwise we'll use model predictions)
        curation_path = os.path.join(self.data_root, "point_cloud_data", "manual_curation", "")
        dir_list = sorted(glob(curation_path + "*"))
        labels_df = []
        for dir_path in dir_list:
            if os.path.isfile(os.path.join(dir_path, self.name + "_labels.csv")):
                labels_df = pd.read_csv(os.path.join(dir_path, self.name + "_labels.csv"))
                labels_df = strip_dummy_cols(labels_df)
                if "fin_label_final" in labels_df.columns:
                    labels_df["fin_label_curr"] = labels_df["fin_label_final"].copy()
                    labels_df.drop(columns=["fin_label_final"], inplace=True)
                # label_path = dir_path + self.name + "_labels.csv"

        if len(labels_df) == 0:
            keep_cols = [col for col in self.point_features.columns if "feat" not in col]
            labels_df = self.point_features.loc[:, keep_cols]
            labels_df["fin_label_curr"] = self.point_features["label_pd"] + 1

        labels_df["approved_flag"] = False

        # self.seg_approved = labels_df.loc[0, "approved_flag"]
        self.full_point_data_raw = labels_df

    def clean_fin_data(self, size_thresh=100, k_nn=3):
        # remove outliers from fin dataset
        ########
        # get spatial outliers
        if not self.seg_approved:
            fin_indices = np.where(self.full_point_data_raw["fin_label_curr"] == 1)[0]
            fin_data = self.full_point_data_raw.loc[fin_indices, ["X", "Y", "Z"]]

            if len(fin_indices) > 25:
                # calculate NN using KD tree
                tree = KDTree(fin_data)
                nearest_dist, nearest_ind = tree.query(fin_data, k=k_nn + 1)

                nn_mean = np.mean(nearest_dist, axis=0)
                nn_scale = nn_mean[1]
                space_outliers = (nearest_dist[:, k_nn] > 2 * nn_scale).ravel()

                #########
                # get size-based outliers
                size_outliers = (self.full_point_data_raw.loc[fin_indices, ["size"]] < size_thresh).to_numpy().ravel()

                # update the data array
                outlier_indices = fin_indices[size_outliers | space_outliers]
                self.full_point_data = self.full_point_data_raw.copy()
                self.full_point_data.loc[outlier_indices, "fin_label_curr"] = 5  # set to outlier class

            else:
                print("No fin data found")
                self.full_point_data = self.full_point_data_raw.copy()
                self.full_point_data.loc[fin_indices, "fin_label_curr"] = 5
        else:
            print("Warning: segmentation is marked as approved. Toggle status using 's' key and rerun")


    def calculate_alpha_surf(self, alpha=10):

        # calculate fin alpha hull
        df = self.full_point_data
        xyz_fin = df.loc[df["fin_label_curr"] == 1, ["X", "Y", "Z"]].to_numpy()

        if np.any(xyz_fin):
            points = xyz_fin / np.max(xyz_fin)

            fin_hull = alphashape.alphashape(points, alpha)  # only works on normalized coordinates
            self.alpha_surf_scale = np.max(xyz_fin)

            fin_hull.vertices = fin_hull.vertices * np.max(xyz_fin)  # shift back to spatial scale

            # use ray-tracing to find intersections
            ray_origins = fin_hull.vertices
            ray_directions = -fin_hull.vertex_normals

            ray_intersections = fin_hull.ray.intersects_location(ray_origins=ray_origins, ray_directions=ray_directions)

            # build normal average array
            centerpoint_array = np.empty(ray_origins.shape)
            centerpoint_array[:] = np.nan
            for i in range(centerpoint_array.shape[0]):
                ray_ft = ray_intersections[1] == i
                if np.sum(ray_ft) >= 2:
                    centerpoint_array[i, :] = np.mean(ray_intersections[0][ray_ft, :], axis=0)

            if np.mean(np.isnan(centerpoint_array)) > 0.5:
                ray_directions = fin_hull.vertex_normals

                ray_intersections = fin_hull.ray.intersects_location(ray_origins=ray_origins,
                                                                     ray_directions=ray_directions)

                # build normal average array
                centerpoint_array = np.empty(ray_origins.shape)
                centerpoint_array[:] = np.nan
                for i in range(centerpoint_array.shape[0]):
                    ray_ft = ray_intersections[1] == i
                    if np.sum(ray_ft) >= 2:
                        centerpoint_array[i, :] = np.mean(ray_intersections[0][ray_ft, :], axis=0)


            self.fin_core_points = centerpoint_array[~np.isnan(centerpoint_array[:, 0]), :]
            self.fin_alpha_surf = fin_hull


    def estimate_body_axes(self):

        if not self.axis_body_approved:
            # get data
            xyz_body = self.full_point_data.loc[self.full_point_data["fin_label_curr"] == 3, ["X", "Y", "Z"]].to_numpy().astype(float)

            # initialize df
            id_vec = ["C", "A", "L", "D", "P", "R", "V"] #+ ["C", "Pr", "L", "D", "Di", "R", "V"]
            # tissue_id_vec = [3]*7 + [1]*7
            # tissue_name_vec = ["body"]*7 + ["fin"]*7
            point_id_float = np.asarray([0, 1, 2, 3, 4, 5, 6]) / 6

            axis_df = pd.DataFrame(id_vec, columns=["point_id"])

            axis_df["point_id_float"] = point_id_float
            # axis_df["tissue_id"] = tissue_id_vec
            # axis_df["tissue_name"] = tissue_name_vec
            axis_df[["X", "Y", "Z"]] = np.nan #np.concatenate((body_centroid[np.newaxis, :], xyz_min_array, xyz_max_array), axis=0)
            # axis_df[["X_pd", "Y_pd", "Z_pd"]] = np.nan
            centroid = np.mean(xyz_body, axis=0)
            # axis_df.loc[0, ["X", "Y", "Z"]] = centroid

            # get body axes
            point_pca = PCA(n_components=3)
            point_pca.fit(xyz_body)

            # pca_array_body = point_pca.transform(xyz_body)
            axes = point_pca.components_

            # orient D-V axis
            if axes[2, 2] < 0:
                axes = -axes

            xyz_min_array = -self.handle_scale * axes + centroid
            xyz_max_array = self.handle_scale * axes + centroid

            # xyz_min_array_pd = -0.66 * handle_scale_global * scale_factor * axes + centroid
            # xyz_max_array_pd = 0.66 * handle_scale_global * scale_factor * axes + centroid

            axis_df.loc[:, ["X", "Y", "Z"]] = np.concatenate((centroid[np.newaxis, :], xyz_min_array, xyz_max_array), axis=0)
            # axis_df.loc[axis_df["tissue_id"] == tissue_flag, ["X_pd", "Y_pd", "Z_pd"]] = np.concatenate(
            #     (centroid[np.newaxis, :], xyz_min_array_pd, xyz_max_array_pd), axis=0)


            self.axis_body = axis_df
            # self.axis_fin_array = axes
        else:
            print("Warning: body axis marked as approved. Skipping PCA estimation. Use 'b' to toggle approved status")
    def fit_yolk_surface(self, dist_thresh=75):

        # calculate distances
        df = self.full_point_data
        yolk_data0 = df.loc[df["fin_label_curr"] == 2, ["X", "Y", "Z"]].to_numpy()
        fin_data0 = df.loc[df["fin_label_curr"] == 1, ["X", "Y", "Z"]].to_numpy()
        dist_mat = distance_matrix(yolk_data0, fin_data0)
        yolk_to_fin_dist = np.min(dist_mat, axis=1)  # just using simple euclidean distance for now

        # filter for nearby surface nuclei
        dist_ft = yolk_to_fin_dist <= dist_thresh
        yolk_data1 = yolk_data0[dist_ft]

        # now fit quadratic surface
        fit = self.polyfit2d(yolk_data1)
        yolk_surf = self.polyval2d(yolk_data1, fit)
        self.yolk_surf_points = yolk_surf
        self.yolk_surf_params = fit


    def polyfit2d(self, xyz, order=2):

        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]
        # general-purpose function for fitting polynomial surfaces to data
        ncols = (order + 1) ** 2
        G = np.zeros((x.size, ncols))
        ij = itertools.product(range(order + 1), range(order + 1))
        for k, (i, j) in enumerate(ij):
            G[:, k] = x ** i * y ** j
        params, _, _, _ = np.linalg.lstsq(G, z)
        return params

    def polyval2d(self, xyz, params):

        x = xyz[:, 0]
        y = xyz[:, 1]

        order = int(np.sqrt(len(params))) - 1
        ij = itertools.product(range(order + 1), range(order + 1))
        z = np.zeros_like(x)
        for a, (i, j) in zip(params, ij):
            z += a * x ** i * y ** j
        return np.c_[x, y, z]

    def calculate_tangent_plane(self, fit_params, point):
        x = point[0]
        y = point[1]

        A, B, C, D, E, F, G, H, I = fit_params

        dfdx = D + E*y + F*y**2 + 2*G*x + 2*H*x*y + 2*I*x*y**2
        dfdy = B + 2*C*y + E*x + 2*F*x*y + H*x**2 + 2*I*x**2*y

        dfdz = -1
        plane_vec_norm = np.asarray([dfdx, dfdy, dfdz])
        plane_vec_norm = plane_vec_norm / np.sqrt(np.sum(plane_vec_norm ** 2))

        # calculate D
        D = -np.dot(plane_vec_norm, point)
        return plane_vec_norm, D
    def estimate_fin_axes(self):

        if not self.axis_fin_approved:
            # we need to know where the yolk surface is for this


            df = self.full_point_data
            fin_data = df.loc[df["fin_label_curr"] == 1, ["X", "Y", "Z"]].to_numpy()


            if np.any(fin_data):
                if self.yolk_surf_points is None:
                    self.fit_yolk_surface()
                yolk_data = self.yolk_surf_points
                # get fin pca
                pca_fin = PCA(n_components=3)
                pca_fin.fit(fin_data)

                # find pec fin nuclei that are near the surface
                dist_array = np.min(distance_matrix(fin_data, yolk_data), axis=1)
                surf_thresh = np.percentile(dist_array, 10)
                surf_indices = np.where(dist_array <= surf_thresh)[0]
                xyz_surf = fin_data[surf_indices]

                # calculate PCA
                pca_surf = PCA(n_components=3)
                pca_surf.fit(xyz_surf)
                lr_axis = pca_surf.components_[0]

                fin_axis0 = pca_fin.components_[0]
                fin_axis1 = pca_fin.components_[1]

                dot0 = np.dot(lr_axis, fin_axis0)
                dot1 = np.dot(lr_axis, fin_axis1)

                pd_ind = 0
                lr_ind = 1
                if np.abs(dot0) > np.abs(dot1):
                    pd_ind = 1
                    lr_ind = 0

                # initialize DF to store fin axes info
                # pca_array_body = point_pca.transform(xyz_body)
                axes = pca_fin.components_
                pca_array = pca_fin.transform(fin_data)
                axes = axes[[pd_ind, lr_ind, 2], :]
                pca_array = pca_array[:, [pd_ind, lr_ind, 2]]
                centroid = np.mean(fin_data, axis=0)

                # orient A-P axis
                cc = np.corrcoef(fin_data[:, 2], pca_array[:, 0])
                if cc[1, 0] < 0:
                    axes[0] = -axes[0]

                # orient D-V axis
                cc2 = np.corrcoef(fin_data[:, 2], pca_array[:, 2])
                if cc2[1, 0] < 0:
                    axes[2] = -axes[2]

                # orient L-R
                axes[1] = np.cross(axes[0], axes[2])

                # calculate ref points
                xyz_min_array = -self.handle_scale * 0.5 * axes + centroid
                xyz_max_array = self.handle_scale * 0.5 * axes + centroid

                # xyz_min_array_pd = -0.66 * handle_scale_global * scale_factor * axes + centroid
                # xyz_max_array_pd = 0.66 * handle_scale_global * scale_factor * axes + centroid
            id_vec = ["C", "Pr", "L", "D", "Di", "R", "V"]
            point_id_float = np.asarray([0, 1, 2, 3, 4, 5, 6]) / 6

            axis_df = pd.DataFrame(id_vec, columns=["point_id"])

            axis_df["point_id_float"] = point_id_float

            if np.any(fin_data):
                axis_df.loc[:, ["X", "Y", "Z"]] = np.concatenate((centroid[np.newaxis, :], xyz_min_array, xyz_max_array),
                                                             axis=0)
            # axis_df.loc[axis_df["tissue_id"] == tissue_flag, ["X_pd", "Y_pd", "Z_pd"]] = np.concatenate(
            #     (centroid[np.newaxis, :], xyz_min_array_pd, xyz_max_array_pd), axis=0)
                self.axis_fin = axis_df
            else:
                axis_df.loc[:, ["X", "Y", "Z"]] = np.nan
                self.axis_fin = axis_df
                self.axis_fin_approved = True
            # self.axis_fin_array = axes
        else:
            print("Warning: fin axes are marked as approved. Skipping PCA estimation. Toggle approval status using 'f'")

    def polyfit2d_fin(self, xyz, order=2, pd_ind=0, reg_flag=False):

        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]
        ncols = 2 * order + 2
        if reg_flag:
            ncols = ncols - 1

        G = np.zeros((x.size, ncols))

        G[:, 0] = 1
        for o in range(order):
            ind = 2 * o + 1
            G[:, ind] = x ** (o + 1)
            G[:, ind + 1] = y ** (o + 1)

        if not reg_flag:
            if pd_ind == 0:
                G[:, -1] = x ** (order + 1)
            else:
                G[:, -1] = y ** (order + 1)
            # print(G.shape)

        fit_params, resid, _, _ = np.linalg.lstsq(G, z)
        return fit_params, resid

    def polyval2d_fin(self, xyz, fit_params, pd_ind=0, reg_flag=False):
        
        x = xyz[:, 0]
        y = xyz[:, 1]
        if not reg_flag:
            order = int((len(fit_params) - 2) / 2)
        else:
            order = int((len(fit_params) - 1) / 2)
        # ij = itertools.product(range(order + 1), range(order + 1))
        z = np.zeros_like(x)

        z += fit_params[0]

        for o in range(order):
            ind = 2 * o + 1
            z += fit_params[ind] * x ** (o + 1)
            z += fit_params[ind + 1] * y ** (o + 1)

        if not reg_flag:
            if pd_ind == 0:
                z += fit_params[-1] * x ** (order + 1)
            else:
                z += fit_params[-1] * y ** (order + 1)

        return np.c_[x, y, z]
        
    def calculate_axis_array(self, axis_df):
        point_array = axis_df.loc[1:, ["X", "Y", "Z"]].to_numpy()
        vec_array = point_array[:3, :] - point_array[3:, :]
        vec_array = np.divide(vec_array, np.linalg.norm(vec_array, axis=1))
        return vec_array 

    def fit_fin_surface(self, samp_frac=0.2, nn_thresh=6.5):
        
        fin_data = self.full_point_data.loc[self.full_point_data["fin_label_curr"] == 1, ["X", "Y", "Z"]].to_numpy()
        if np.any(fin_data):
            fin_centerpoints = self.fin_core_points
            fin_axis_df = self.axis_fin
            fin_axes = self.calculate_axis_array(fin_axis_df)
            # print(fin_axes)
            # pca_fin = PCA(n_components=3)
            # pca_fin.components_ = fin_axes
            pca_centerpoints = np.matmul(fin_centerpoints - np.mean(fin_centerpoints, axis=0), fin_axes.T)
            pca_fin_array = np.matmul(fin_data - np.mean(fin_data, axis=0), fin_axes.T)

            fin_scale = np.mean(np.std(fin_data, axis=0))

            base_thresh = np.percentile(pca_centerpoints[:, 0], 10)
            tip_thresh = np.percentile(pca_fin_array[:, 0], 90)

            n_fin_points = fin_centerpoints.shape[0]

            # select base points
            base_ft = pca_centerpoints[:, 0] <= base_thresh
            base_indices = np.where(base_ft)[0]

            # generate pseudopoints extending toward base of fin
            # lr_vec = pca_fin.components_[1]  # this should be AP axis of fin, more or less

            # calculate centroid
            yolk_surf = self.yolk_surf_points
            surf_cm = np.mean(fin_centerpoints[base_ft], axis=0)
            closest_i = np.argmin(np.sqrt(np.sum((surf_cm - yolk_surf) ** 2, axis=1)))  # find closest surface point
            closest_point = yolk_surf[closest_i]

            surf_normal, _ = self.calculate_tangent_plane(self.yolk_surf_params, closest_point)
            surf_normal = surf_normal / np.linalg.norm(surf_normal)

            if surf_normal[2] > 0:
                surf_normal = -surf_normal

            # generate base pseudo-points to guide curvature of the fin
            np.random.seed(1423)
            n_base_points = int(samp_frac * n_fin_points)
            surf_samples = np.random.choice(base_indices, n_base_points, replace=True)  # , p=dist_vec / np.sum(dist_vec))
            scale_factors = 1 + np.random.rand(n_base_points, 1) * fin_scale / 10
            base_points = fin_centerpoints[surf_samples] - np.multiply(scale_factors * nn_thresh, surf_normal)

            tip_indices = np.where(pca_fin_array[:, 0] >= tip_thresh)[0]
            tip_samples = np.random.choice(tip_indices, n_base_points, replace=True)
            tip_points = fin_data[tip_samples]

            fit_data = np.concatenate((fin_centerpoints, base_points, tip_points), axis=0)
            fit_data_pca = np.matmul(fit_data - np.mean(fit_data, axis=0), fin_axes.T)

            test_1 = np.arange(np.percentile(fit_data_pca[:, 1], 0), np.percentile(fit_data_pca[:, 1], 100), 1)
            test_0 = np.arange(np.percentile(fit_data_pca[:, 0], 0), np.percentile(fit_data_pca[:, 0], 100), 1)
            PCA0, PCA1 = np.meshgrid(test_0, test_1)
            PC0 = PCA0.ravel()
            PC1 = PCA1.ravel()

            # adding heuristic to prevent poor fits
            pca_scales = np.max(pca_fin_array, axis=0) - np.min(pca_fin_array, axis=0) #np.std(pca_fin_array, axis=0)
            reg_flag = (pca_scales[0] < 1.75*pca_scales[2]) | (pca_scales[0] < 100)
            # fin_fit1, loss1 = self.polyfit2d_fin(fit_data_pca, order=1)
            fin_fit, _ = self.polyfit2d_fin(fit_data_pca, order=2, reg_flag=reg_flag)
            PC_array = self.polyval2d_fin(np.c_[PC0, PC1], fin_fit)

            tri = Delaunay(PC_array[:, :2])
            faces = tri.simplices

            # transform back
            # XYZ_surf = np.dot(PC_array, fin_axes) + np.mean(PC_array, axis=0)
            XYZ_surf = np.matmul(PC_array, np.linalg.inv(fin_axes.T)) + np.mean(fit_data, axis=0)
            # store
            self.fin_surf_params = fin_fit
            self.fin_surf_points = XYZ_surf
            self.fin_surf_faces = faces
    
    def manual_axis_shift(self, selected_index, layer):

        data = layer.data
        new_position = layer.data[selected_index].copy()

        new_vec = new_position - data[0, :]
        new_vec = new_vec / np.sqrt(np.sum((new_vec) ** 2))
        if selected_index > 0:

            scale_factor = 1
            if layer.name == "fin axis points":
                scale_factor = 0.5

            # update orientation of target axis
            if selected_index < 4:
                alt_index = selected_index + 3
            else:
                alt_index = selected_index - 3

            new_pos0 = data[0, :] + scale_factor * new_vec * self.handle_scale
            new_pos1 = data[0, :] - scale_factor * new_vec * self.handle_scale

            data[selected_index, :] = new_pos0
            data[alt_index, :] = new_pos1

            D = -np.dot(new_vec, data[0, :])

            # we also need to adjust the other axes for consistency--they all need to be orthogonal
            options = [i for i in range(1, 4) if i not in [alt_index, selected_index]]
            ind = np.random.choice(options, 1)[0]
            curr_point = data[ind, :]
            # v = curr_point - data[0, :]
            dist = np.dot(curr_point, new_vec) + D

            raw_point = curr_point - new_vec * dist
            vec = raw_point - data[0, :]
            vec = vec / np.sqrt(np.sum((vec) ** 2))
            # maintain original polarity
            if np.dot(vec, data[ind, :] - data[ind+3, :]) < 0:
                vec = -vec

            data[ind, :] = data[0, :] + scale_factor * vec * self.handle_scale
            data[ind + 3, :] = data[0, :] - scale_factor * vec * self.handle_scale

            last_i = [i for i in range(1, 4) if i not in [alt_index, selected_index, ind]][0]
            vec1 = np.cross(vec, new_vec)
            vec1 = vec1 / np.sqrt(np.sum((vec1) ** 2))
            if np.dot(vec1, data[last_i, :] - data[last_i+3, :]) < 0:
                vec1 = -vec1
            data[last_i, :] = data[0, :] + scale_factor * vec1 * self.handle_scale
            data[last_i + 3, :] = data[0, :] - scale_factor * vec1 * self.handle_scale

            # update layer
            layer.data = data

            # update appropriate fin data field
            if layer.name == "fin axis points":
                self.axis_fin.loc[:, ["Z", "Y", "X"]] = data
            elif layer.name == "body axis points":
                self.axis_body.loc[:, ["Z", "Y", "X"]] = data

            # call update sequence
            self.curation_update_sequence()

        return layer

    def manual_polarity_switch(self, selected_index, layer):
        if selected_index > 0:
            data = layer.data
            swap0 = data[selected_index, :].copy()
            if selected_index < 4:
                alt_index = selected_index + 3
            else:
                alt_index = selected_index - 3
            swap1 = data[alt_index, :].copy()

            data[selected_index, :] = swap1
            data[alt_index, :] = swap0
            layer.data = data

            if layer.name == "body axis points":
                self.axis_body.loc[:, ["Z", "Y", "X"]] = data
            elif layer.name == "fin axis points":
                self.axis_fin.loc[:, ["Z", "Y", "X"]] = data


        return layer