import numpy as np
from scipy.integrate import quad

def strip_dummy_cols(df):
    cols = df.columns
    keep_cols = [col for col in cols if "Unnamed" not in col]
    df = df[keep_cols]
    return df

def calculate_path_distance(FDECAB, p0, p1):
    dx = p0[0] - p1[0]
    dy = p0[1] - p1[1]

    I = quad(numerical_integral, 0, 1, args=(FDECAB, p0, [dx, dy]))

    return I[1] - I[0]


def solve_for_f(C_fit, cp):
    # C0 = z0 - C1*x0 - C2*y0 - C3*x0*yz - C4*x0**2 - C5*Y0**2
    x0 = cp[0]
    y0 = cp[1]
    z0 = cp[2]
    C0 = z0 - C_fit[0] * x0 - C_fit[1] * y0 - C_fit[2] * x0 * y0 - C_fit[3] * x0 ** 2 - C_fit[4] * y0 ** 2
    C_out = [C0, C_fit[0], C_fit[1], C_fit[2], C_fit[3], C_fit[4]]
    return C_out


def predict_quadratic_surface(xy, DECAB, point):
    C_full = solve_for_f(DECAB, point)
    A = np.c_[np.ones(xy.shape[0]), xy, np.prod(xy, axis=1), xy ** 2]
    surf_pd = np.dot(A, C_full).ravel()
    xyz_out = np.concatenate((xy, np.reshape(surf_pd, (xy.shape[0], 1))), axis=1)
    return xyz_out, C_full


def numerical_integral(t, FDECAB, P0, V):
    F, D, E, C, A, B = FDECAB
    v0, v1 = V
    x0, y0, _ = P0
    integrand = np.sqrt(v0 ** 2 + v1 ** 2 + (
                D * v0 + 2 * A * v0 * (t * v0 + x0) + v1 * (E + 2 * C * t * v0 + 2 * B * t * v1 + C * x0) + (
                    C * v0 + 2 * B * v1) * y0) ** 2)
    return integrand


def calculate_tangent_plane(C_full, point):
    F, D, E, C, A, B = C_full
    dfdx = 2 * A * point[0] + C * point[1] + D
    dfdy = 2 * B * point[1] + C * point[0] + E
    dfdz = -1
    plane_vec_norm = np.asarray([dfdx, dfdy, dfdz])
    plane_vec_norm = plane_vec_norm / np.sqrt(np.sum(plane_vec_norm ** 2))

    # calculate D
    D = -np.dot(plane_vec_norm, point)
    return plane_vec_norm, D


def make_quadratic_array(data):
    A = np.c_[np.ones(data.shape[0]), data[:, :2], np.prod(data[:, :2], axis=1), data[:, :2] ** 2]
    return A


def predict_z(A, C):
    Z = np.dot(A, C)
    return Z


def quadratic_z_loss(C, data):
    A = make_quadratic_array(data)
    Z = predict_z(A, C)
    residuals = data[:, 2] - Z
    return residuals