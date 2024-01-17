import numpy as np
import scipy.stats
import alphashape
import multiprocessing as mp
from contextlib import contextmanager
from functools import partial
from sklearn.neighbors import KDTree
from scipy.spatial import ConvexHull
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import transforms


@contextmanager
def poolcontext(*args, **kwargs):
    pool = mp.Pool(*args, **kwargs)
    yield pool
    pool.terminate()


def approx_normal_vector(neighbours):
    """
    Finding the normal vector with length of 1 for each neighbour point in terms of the local coordinates system
    It will be served as the approximation of the normal vector of the fitted surface for
    Args:
        neighbours:
            the coordinate of neighbours considering the point of interest as centroid
    Returns:
        array of N normal vectors of the fitted PCA plane. The starting point of normal vectors are
        moved to each corresponding points.
    """
    mean = np.mean(neighbours, axis=0)
    centered_system = (neighbours - mean)
    U, S, V = np.linalg.svd(centered_system, full_matrices=False)
    # First Eigen vector
    EV = V[2, :]
    norm_EV = np.sqrt(EV[0] ** 2 + EV[1] ** 2 + EV[1] ** 2)
    normal_base = EV / norm_EV
    normal_vectors = neighbours + normal_base
    return normal_vectors


def compute_relative_cord(centroid, neighbours):
    # Construct local coordinates
    relative_neighbours = neighbours-centroid
    return relative_neighbours


def differential_reconstruction(normal_vectors):
    # Construct the normal vectors make the Z to 1 to ensure x,y is the first order derivative at point p
    result = []
    for n in normal_vectors:
        result.append(n / n[-1] * -1)
    return np.array(result)


def second_order_curve(neighbours, normal_vectors=None):
    '''
    Fit z = r(x,y) = ax^2+by^2+cxy+dx+ey (He et al., 2013) as the local surface

    He, B., Lin, Z., & Li, Y. F. (2013). An automatic registration algorithm for the scattered point clouds based on the curvature feature. Optics & Laser Technology, 46, 53-60.

    Args:
        neighbours: K neighbours of the given point using kNN
        normal_vectors: The normal vectors approximated by PCA
    Returns:
        The coefficient of second order curve
    '''
    linear_response = neighbours[:, -1]
    # ax^2+by^2+cxy+dx+ey -> P^Tx
    linear_part = np.array((neighbours[:, 0] ** 2,
                            neighbours[:, 1] ** 2,
                            neighbours[:, 0] * neighbours[:, 1],
                            neighbours[:, 0],
                            neighbours[:, 1]))
    if normal_vectors is None:
        LS_fit = LinearRegression(fit_intercept=False).fit(
            linear_part.T, linear_response)
    else:
        # dr(x,y)/dx = 2ax+cy+d
        # dr(x,y)/dy = 2by+cx+e
        # Using the (Goldfeather & Interrante, 2004) method to expand the variables, approximate the normal vectors by principle vectors
        differential_part = differential_reconstruction(normal_vectors)
        differential_response_x = differential_part[:, 0]
        differential_response_y = differential_part[:, 1]
        differential_part_x = np.array((2 * neighbours[1:, 0],
                                       [0] * len(neighbours[1:, 1]),
                                       neighbours[1:, 1],
                                       [1] * len(neighbours[1:, 1]),
                                       [0] * len(neighbours[1:, 1])))
        differential_part_y = np.array(([0] * len(neighbours[1:, 1]),
                                       2 * neighbours[1:, 1],
                                       neighbours[1:, 0],
                                       [0] * len(neighbours[1:, 1]),
                                       [1] * len(neighbours[1:, 1])))
        response = np.concatenate(
            (linear_response, differential_response_x, differential_response_y), axis=0)
        variable = np.concatenate(
            (linear_part, differential_part_x, differential_part_y), axis=1)
        LS_fit = LinearRegression(
            fit_intercept=False).fit(variable.T, response)
    return LS_fit.coef_


def third_order_curve(neighbours, normal_vectors=None):
    '''
    Fit z = r(x,y) = ax^2+bxy+cy^2+dx^3+ex^2y+fxy^2+gy^3 (Goldfeather & Interrante, 2004) as the local surface
    Args:
        neighbours: K neighbours of the given point using kNN
        normal_vectors: The normal vectors approximated by PCA
    Returns:
        The coefficient of second order curve
    '''
    linear_response = neighbours[:, -1]
    # ax^2+bxy+cy^2+dx^3+ex+fy+gy^3 -> P^Tx
    linear_part = np.array((neighbours[:, 0] ** 2,
                            neighbours[:, 0] * neighbours[:, 1],
                            neighbours[:, 1] ** 2,
                            neighbours[:, 0] ** 3,
                            neighbours[:, 0],
                            neighbours[:, 1],
                            neighbours[:, 1] ** 3))

    if normal_vectors is None:
        LS_fit = LinearRegression(fit_intercept=False).fit(
            linear_part.T, linear_response)
    else:
        # dr(x,y)/dx = 2ax+by+3dx^2+e
        # dr(x,y)/dy = bx+2cy+f+3gy^2
        # Using the (Goldfeather & Interrante, 2004) method to expand the variables, approximate the normal vectors by principle vectors
        # Under construction
        differential_part = differential_reconstruction(normal_vectors)
        differential_response_x = differential_part[:, 0]
        differential_response_y = differential_part[:, 1]
        differential_part_x = np.array((2 * neighbours[1:, 0],
                                       neighbours[1:, 1],
                                       [0] * len(neighbours[1:, 1]),
                                       3 * neighbours[1:, 0] ** 2,
                                       [1] * len(neighbours[1:, 1]),
                                       [0] * len(neighbours[1:, 1]),
                                       [0] * len(neighbours[1:, 1])))
        differential_part_y = np.array(([0] * len(neighbours[1:, 1]),
                                       neighbours[1:, 0],
                                       2 * neighbours[1:, 1],
                                       [0] * len(neighbours[1:, 1]),
                                       [0] * len(neighbours[1:, 1]),
                                       [1] * len(neighbours[1:, 1]),
                                       3 * neighbours[1:, 1] ** 2))
        response = np.concatenate(
            (linear_response, differential_response_x, differential_response_y), axis=0)
        variable = np.concatenate(
            (linear_part, differential_part_x, differential_part_y), axis=1)
        LS_fit = LinearRegression(
            fit_intercept=False).fit(variable.T, response)
    return LS_fit.coef_


def gaussian_curvature(r_x, r_y, r_xx, r_xy, r_yy, normal_vector):
    """
    Finding the Gaussian curvature.
    Args: -> vectors as 1d np array
        r_x: partial derivative of the surface function respect to x
        r_y: partial derivative of the surface function respect to y
        r_xx: 2nd order partial derivative of the surface function respect to x
        r_xy: 2nd order partial derivative of the surface function respect to x and y
        r_yy: 2nd order partial derivative of the surface function respect to y
        normal_vector: The normal vector at the point of interest to the fitted surface

    Returns: -> float
        Gaussian curvature as the determent of the Weingarten map
    """
    E = np.dot(r_x, r_x)
    F = np.dot(r_x, r_y)
    G = np.dot(r_y, r_y)
    L = np.dot(normal_vector, r_xx)
    M = np.dot(normal_vector, r_xy)
    N = np.dot(normal_vector, r_yy)
    assert (E * G - F ** 2) != 0
    return (L * N - M ** 2) / (E * G - F ** 2)


def get_key_nums_second(coefs, point=(0, 0, 0)):
    # Get E,F,G,L,M,N to calculate Gaussian curvature
    assert len(coefs) == 5 and len(point) == 3
    x, y, z = point
    a, b, c, d, e = coefs
    # parietal derivatives
    r_x = np.array((1, 0, 2 * a * x + c * y + d))
    r_y = np.array((0, 1, 2 * b * y + c * x + e))
    r_xy = np.array((0, 0, c))
    r_xx = np.array((0, 0, 2 * a))
    r_yy = np.array((0, 0, 2 * b))
    # normal vector to the surface at p0
    normal_vector = np.array(
        (2 * a * x + c * y + d, 2 * b * y + c * x + e, -1))
    gs_curvature = gaussian_curvature(
        r_x, r_y, r_xx, r_xy, r_yy, normal_vector)
    return gs_curvature


def get_key_nums_third(coefs, point=(0, 0, 0)):
    # Get E,F,G,L,M,N to calculate Gaussian curvature
    assert len(coefs) == 7 and len(point) == 3
    x, y, z = point
    a, b, c, d, e, f, g = coefs
    r_x = np.array((1, 0, 2 * a * x + b * y + 3 * d * x ** 2 + e))
    r_y = np.array((0, 1, b * x + 2 * c * y + f + 3 * g * y ** 2))
    r_xy = np.array((0, 0, b))
    r_xx = np.array((0, 0, 2 * a + 6 * d * x))
    r_yy = np.array((0, 0, 2 * c + 6 * g * y))
    normal_vector = np.array(
        (2 * a * x + b * y + 3 * d * x ** 2 + e, b * x + 2 * c * y + f + 3 * g * y ** 2, -1))
    gs_curvature = gaussian_curvature(
        r_x, r_y, r_xx, r_xy, r_yy, normal_vector)
    return gs_curvature


def fit_surface(neighbours, method='second', expand_variables=False):
    '''
    The function is fitting the local surface of each point.
    Formula of the surface are:
    z = r(x,y) = ax^2+by^2+cxy+dx+ey (He et al., 2013)
    z = r(x,y) = ax^2+bxy+cy^2+dx^3+ex+fy+gy^3 (Goldfeather & Interrante, 2004)

    Using the fact that:
    1. The neighbour vectors should be lie on the fitted surface
        -> minimize the distance between the neighbour and the fitted surface
    2. The normal vector of the neighbour vector is not on the surface
        -> using the relationship between the normal vector and surface to expand the formula
    Returns: corresponding tuple of parameters for estimation
    '''
    if expand_variables is True:
        normal_vectors = approx_normal_vector(neighbours)[:-1]
    else:
        normal_vectors = None
    if method == 'third':
        coefs = third_order_curve(neighbours, normal_vectors)
        local_curvature = get_key_nums_third(coefs)
    else:
        coefs = second_order_curve(neighbours, normal_vectors)
        local_curvature = get_key_nums_second(coefs)
    return local_curvature


def compute_curvature_mp(point_cloud, indices, method='second', expand_variable=False, offset=1):
    # create a list contains 1 to shape
    processes = mp.cpu_count() - offset
    all_neighbours = []
    for i in range(len(indices)):
        centroid = point_cloud[i]
        neighbour_cord = np.array([point_cloud[n] for n in indices[i]])
        neighbours = compute_relative_cord(centroid, neighbour_cord)
        all_neighbours.append(neighbours)
    curvatures = []
    with mp.Pool(processes=processes) as pool:
        chunk_curvatures = pool.map(partial(
            fit_surface, method=method, expand_variables=expand_variable), all_neighbours)
        curvatures.append(chunk_curvatures)

    # get curvatures from the tuple (curvatures, seq) regrrange by seq
    curvatures = np.array(curvatures)
    # Compute the root mean squared curvature
    return curvatures


def compute_curvature(point_cloud, query_k=8, method='second', expand_variable=False):
    # Build a KDTree to find the nearest neighbors of each point
    tree = KDTree(point_cloud)

    # Find the nearest neighbors of each point
    indices = tree.query(point_cloud, k=query_k, return_distance=False)
    curvatures = compute_curvature_mp(
        point_cloud, indices, method, expand_variable)
    # debug script using single thread
    # curvatures = []
    # for i in range(len(indices)):
    #     centroid = point_cloud[i]
    #     neighbour_cord = np.array([point_cloud[n] for n in indices[i]])
    #     neighbours = compute_relative_cord(centroid, neighbour_cord)
    #     local_curvature = fit_surface(neighbours, method, expand_variable)
    #     curvatures.append(local_curvature)
    # curvatures=np.array(curvatures)
    log_curvatures = np.log(abs(curvatures)+1)
    curvature = np.mean(np.abs(curvatures))
    return curvature, log_curvatures


def plot_curvature(point_cloud, curvature, log_curvatures):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cur_plot = ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2],
                          c=log_curvatures, cmap='tab20b', s=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Mean Absolute Gaussian curvature:' +
                 '' + str(round(curvature, 4)))
    plt.colorbar(cur_plot, ax=ax, shrink=0.8,
                 label='Log Gaussian curvature', pad=0.1)
    # plt.show()
    plt.savefig('curvature.png', dpi=600, transparent=True)


def compute_roughness(point_cloud):
    # Fit a plane to the point cloud using SVD (Singular Value Decomposition)
    mean = np.mean(point_cloud, axis=0)
    centered_cloud = (point_cloud - mean)
    U, S, V = np.linalg.svd(centered_cloud, full_matrices=False)
    # First Eigen vector
    plane = V[2, :]
    # Mean point must be on the plane
    d = -np.dot(plane, mean)

    # Calculate the height differences between the points and the fitted plane
    # Plane formula: ax+by+cz+d =0; d is the central difference, a,b,c are from the principle vector
    # difference is explained as the non-zero part of the abs(ax+by+cz+d) = height_diff
    height_differences = np.abs(point_cloud[:, 2] * plane[2] +
                                plane[0] * point_cloud[:, 0] + plane[1] * point_cloud[:, 1] + d)

    # Calculate the structural roughness as the RMS of the height differences to the fitted plane
    roughness = np.sqrt(np.mean(height_differences ** 2))
    return plane, d, height_differences, roughness


def plot_roughness(point_cloud, plane, d, height_differences, roughness):
    fig, axs = plt.subplot_mosaic([['a', 'b'], ['c', 'd']], figsize=(15, 14), constrained_layout=True)

    for label, ax in axs.items():
        trans = transforms.ScaledTranslation(-20 /
                                             72, 7 / 72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
                fontsize='22', va='bottom', fontfamily='serif', fontweight='bold', color='black')
        ax.set_axis_off()

    # First subplot
    ax = fig.add_subplot(2, 2, 1)
    im = ax.scatter(point_cloud[:, 0], point_cloud[:, 1],
                    c=height_differences, cmap='cividis', s=0.9)
    cbhd = fig.colorbar(im, ax=ax, shrink=0.8, label='Height difference', fraction=0.046, pad=0.1)
    cbhd.set_label('Height difference',  fontsize='x-large')
    ax.set_title('Point cloud top view', fontsize='x-large')
    ax.set_xlabel('X', fontsize='14')
    ax.set_ylabel('Y', fontsize='14')
    ax.tick_params(axis='both', which='major', labelsize='11')

    # Second subplot
    ax = fig.add_subplot(2, 2, 2, projection='3d')
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1],
               point_cloud[:, 2], c=height_differences, cmap='cividis', s=0.6)
    ax.set_xlabel('X', fontsize='14')
    ax.set_ylabel('Y', fontsize='14')
    ax.set_zlabel('Z', fontsize='14')
    ax.tick_params(axis='both', which='major', labelsize='12', pad=0.3)
    x, y = np.meshgrid(np.linspace(np.min(point_cloud[:, 0]), np.max(point_cloud[:, 0]), 10),
                       np.linspace(np.min(point_cloud[:, 1]), np.max(point_cloud[:, 1]), 10))
    z = (-plane[0] * x - plane[1] * y - d) / plane[2]
    ax.plot_surface(x, y, z, alpha=0.5, color='#8c000f')
    ax.set_title('Structural roughness: ' +
                 str(round(roughness, 4)), fontsize='x-large')
    ax.view_init(elev=30, azim=-90, roll=0)
    
    # Third subplot
    ax = fig.add_subplot(2, 2, 3, projection='3d')
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1],
                point_cloud[:, 2], c=height_differences, cmap='cividis', s=0.6)
    ax.set_xlabel('X', fontsize='14')
    ax.set_ylabel('Y', fontsize='14')
    ax.set_zlabel('Z', fontsize='14')
    ax.tick_params(axis='both', which='major', labelsize='11', pad=0.3)
    x, y = np.meshgrid(np.linspace(np.min(point_cloud[:, 0]), np.max(point_cloud[:, 0]), 10),
                        np.linspace(np.min(point_cloud[:, 1]), np.max(point_cloud[:, 1]), 10))
    z = (-plane[0] * x - plane[1] * y - d) / plane[2]
    ax.plot_surface(x, y, z, alpha=0.5, color='#8c000f')


    # Fourth subplot
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1],
                point_cloud[:, 2], c=height_differences, cmap='cividis', s=0.6)
    ax.set_xlabel('X', fontsize='14')
    ax.set_ylabel('Y', fontsize='14')
    ax.set_zlabel('Z', fontsize='14')
    ax.tick_params(axis='both', which='major', labelsize='11', pad=0.3)
    x, y = np.meshgrid(np.linspace(np.min(point_cloud[:, 0]), np.max(point_cloud[:, 0]), 10),
                        np.linspace(np.min(point_cloud[:, 1]), np.max(point_cloud[:, 1]), 10))
    z = (-plane[0] * x - plane[1] * y - d) / plane[2]
    ax.plot_surface(x, y, z, alpha=0.5, color='#8c000f')
    # ax.set_title('Structural roughness: ' +
    #                 str(round(roughness, 4)))
    ax.view_init(elev=30, azim=0, roll=0)
    plt.savefig('roughness.png', dpi=600,
                transparent=True, bbox_inches='tight')


def compute_entropy(point_cloud, bins=300, base=2):
    # Calculate probability of occurrence for each point in the cloud
    point_density, _ = np.histogramdd(point_cloud, bins=bins)

    # Normalize probability values to sum up to 1
    norm_point_density = point_density / np.sum(point_density)

    # Use scipy.stats.entropy function to measure Shannon entropy value
    entropy_val = scipy.stats.entropy(np.ravel(norm_point_density), base=base)
    return entropy_val, norm_point_density


def plot_entropy(entropy_val, norm_point_density):
    im = plt.imshow(np.sum(norm_point_density, axis=2),
                    cmap='gray', origin='lower')
    plt.title("Shannon entropy ({entropy_val:.4f} bits)".format(
        entropy_val=entropy_val))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar(im, shrink=0.8, label='Entropy (bits)', pad=0.1)
    # plt.show()
    plt.savefig('entropy.png', dpi=600, transparent=True)


def compute_convex_hull(point_cloud):
    return ConvexHull(point_cloud)


def plot_convex_hull(point_cloud, convex_hull):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1],
               point_cloud[:, 2], color='dimgrey', s=0.1, alpha=0.1)
    for s in convex_hull.simplices:
        vertices = point_cloud[s, :]
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1],
                        vertices[:, 2], color='saddlebrown', alpha=1, linewidth=0.2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Convex hull volume: ' + str(round(convex_hull.volume, 4)))
    plt.savefig('convex_hull.png', dpi=600, transparent=True)


def compute_alpha_shape(point_cloud, alpha=1.1):
    return alphashape.alphashape(point_cloud, alpha)


def plot_alpha_shape(alpha_shape):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(*zip(*alpha_shape.vertices),
                    triangles=alpha_shape.faces, color='saddlebrown', alpha=0.7)
    ax.set_title('Alpha shape volume: ' + str(round(alpha_shape.volume, 4)))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig('alpha_shape.png', dpi=600, transparent=True)


def plot_volume_all(point_cloud, convex_hull, alpha_shape):
    fig, axs = plt.subplot_mosaic([['a', 'b']], figsize=(13, 5))

    for label, ax in axs.items():
        trans = transforms.ScaledTranslation(-20 /
                                             72, 7 / 72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
                fontsize='x-large', va='bottom', fontfamily='serif', fontweight='bold', color='black')
        ax.set_axis_off()

    # First subplot
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1],
               point_cloud[:, 2], color='dimgrey', s=0.1, alpha=0.1)
    for s in convex_hull.simplices:
        vertices = point_cloud[s, :]
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1],
                        vertices[:, 2], color='saddlebrown', alpha=1, linewidth=0.2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Convex hull volume: ' + str(round(convex_hull.volume, 4)))

    # Second subplot
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot_trisurf(*zip(*alpha_shape.vertices),
                    triangles=alpha_shape.faces, color='saddlebrown', alpha=0.7)
    ax.set_title('Alpha shape volume: ' + str(round(alpha_shape.volume, 4)))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig('volume_all.png', dpi=600,
                transparent=True, bbox_inches='tight')


def compute_gaussian_mixture(point_cloud, n_components=3, covraiance_type='full'):
    # Fit a Gaussian mixture model to the point cloud
    gmm = GaussianMixture(n_components=n_components,
                          covariance_type=covraiance_type)
    labels = gmm.fit_predict(point_cloud)
    covariance = gmm.covariances_
    weights = gmm.weights_
    weighted_average_covariance = np.average(
        covariance, axis=0, weights=weights)
    dispersion = np.linalg.det(weighted_average_covariance)
    iteration = gmm.n_iter_
    return labels, dispersion, iteration


def read_pts(file_name):
    with open(file_name, 'r') as f:
        # read lines, get the line number of 'end_header'
        lines = f.readlines()
        stop_line = 0
        for i in range(len(lines)):
            if lines[i] == 'end_header\n':
                stop_line = i
                break
        # read the coordinates of the points
        lines = lines[stop_line+1:]
        lines = [line.strip().split(' ') for line in lines]
        lines = [[float(line[0]), float(line[1]), float(line[2])]
                 for line in lines]
        # make lines np array
        points = np.array(lines)
        return points


def plot_gaussian_mixture(point_cloud, labels, dispersion, iteration):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1],
               point_cloud[:, 2], c=labels, s=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('GMM point dispersion: ' + str(round(dispersion, 4)
                                                ) + '\n' + 'Number of iterations: ' + str(iteration))
    plt.savefig('gaussian_mixture.png', dpi=600, transparent=True)


class roughness:
    def __init__(self, point_cloud) -> None:
        self.point_cloud = point_cloud
        self.plane, self.d, self.height_differences, self.roughness = compute_roughness(
            self.point_cloud)
        self.plot = plot_roughness(
            self.point_cloud, self.plane, self.d, self.height_differences, self.roughness)
        pass


class curvature:
    def __init__(self, point_cloud, query_k=14, method='third', expand_variable=True) -> None:
        self.point_cloud = point_cloud
        self.query_k = query_k
        self.method = method
        self.expand_variable = expand_variable
        self.mean_curvature, self.curvature_data = compute_curvature(
            self.point_cloud, self.query_k, self.method, self.expand_variable)
        self.plot = plot_curvature(
            self.point_cloud, self.mean_curvature, self.curvature_data)
        pass

    def re_compute(self, new_query_k):
        self.query_k = new_query_k
        self.mean_curvature, self.curvature_data = compute_curvature(
            self.point_cloud, self.query_k, self.method, self.expand_variable)


class convex_hull:
    def __init__(self, point_cloud) -> None:
        self.point_cloud = point_cloud
        self.convex_hull = compute_convex_hull(self.point_cloud)
        self.points = self.convex_hull.points
        self.vertices = self.convex_hull.vertices
        self.neighbors = self.convex_hull.neighbors
        self.volume = self.convex_hull.volume
        self.area = self.convex_hull.area
        self.simplices = self.convex_hull.simplices
        self.plot = plot_convex_hull(self.point_cloud, self.convex_hull)
        pass


class alpha_shape:
    def __init__(self, point_cloud) -> None:
        self.point_cloud = point_cloud
        self.alpha_shape = compute_alpha_shape(self.point_cloud)
        self.vertices = self.alpha_shape.vertices
        self.faces = self.alpha_shape.faces
        self.volume = self.alpha_shape.volume
        self.plot = plot_alpha_shape(self.alpha_shape)
        pass


class entropy:
    def __init__(self, point_cloud, bins=300, base=2) -> None:
        self.point_cloud = point_cloud
        self.bins = bins
        self.base = base
        self.entropy_val, self.norm_point_density = compute_entropy(
            self.point_cloud, self.bins, self.base)
        self.plot = plot_entropy(self.entropy_val,
                                 self.norm_point_density)
        pass

    def re_compute(self, new_bins, new_base):
        self.bins = new_bins
        self.base = new_base
        self.entropy_val, self.norm_point_density = compute_entropy(
            self.point_cloud, self.bins, self.base)


class gmm:
    def __init__(self, point_cloud, n_components=3, covariance_type='full') -> None:
        self.point_cloud = point_cloud
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.labels, self.dispersion, self.iteration = compute_gaussian_mixture(
            self.point_cloud, self.n_components, self.covariance_type)
        self.plot = plot_gaussian_mixture(
            self.point_cloud, self.labels, self.dispersion, self.iteration)
        pass

    def re_compute(self, new_n_components, new_covariance_type):
        self.n_components = new_n_components
        self.covariance_type = new_covariance_type
        self.gmm, self.labels, self.dispersion = compute_gaussian_mixture(
            self.point_cloud, self.n_components, self.covariance_type)
        pass


class volume_all:
    def __init__(self, point_cloud, convex_hull, alpha_shape) -> None:
        self.point_cloud = point_cloud
        self.convex_hull = convex_hull
        self.alpha_shape = alpha_shape
        self.plot = plot_volume_all(
            self.point_cloud, self.convex_hull, self.alpha_shape)
        pass


class HSC3D:

    def __init__(self, plot_path) -> None:
        self.point_cloud = read_pts(plot_path)
        print('Loadded the point cloud from:', plot_path,
              '. Shape of the point cloud is:', self.point_cloud.shape)
        print('Computing Shannon entropy ...')
        self.entropy = entropy(self.point_cloud)
        print('Entropy computed as:', self.entropy.entropy_val)
        print('Computing structural roughness ...')
        self.roughness = roughness(self.point_cloud)
        print('Structural roughness computed as:',
              self.roughness.roughness)
        print('Computing curvature ...')
        self.curvature = curvature(self.point_cloud)
        print('Curvature computed as:', self.curvature.mean_curvature)
        print('Computing convex hull ...')
        self.convex_hull = convex_hull(self.point_cloud)
        print('Convex hull volume computed as:', self.convex_hull.volume)
        print('Computing alpha shape...')
        self.alpha_shape = alpha_shape(self.point_cloud)
        print('Alpha shape volume computed as:', self.alpha_shape.volume)
        print('Plotting volume ...')
        self.volume_all = volume_all(
            self.point_cloud, self.convex_hull, self.alpha_shape)
        print('Computing Gaussian mixture model...')
        self.gmm = gmm(self.point_cloud)
        print('Gaussian mixture model dispersion computed as:',
              self.gmm.dispersion)
        pass


if __name__ == '__main__':
    '''
    Test only
    '''

    point_path = "ttq2af.ply"
    point_cloud = read_pts(point_path)
    a = HSC3D(point_path)
