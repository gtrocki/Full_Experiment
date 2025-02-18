import matplotlib.path
import numpy as np
import torch
import pyvista as pv
from svgpath2mpl import parse_path
import matplotlib as mpl  # for path transforms
from matplotlib.patches import PathPatch
import matplotlib.pyplot as plt
from matplotlib import colormaps
# import open3d as o3d
import config
import os


# =============================================================================
class ForceFinderModel(torch.nn.Module):
    """
    Defines the pytorch model that will serve as our neural network.
    """
    def __init__(self, hidden_layer_lengths):
        """
        Sets up the model, which has a variable number of layers with variable number of nodes each.
        :param hidden_layer_lengths: An array containing the length of each inner layer.
        """
        super().__init__()
        self.hidden_layer_lengths = hidden_layer_lengths
        self.model_layers = [torch.nn.Linear(2, self.hidden_layer_lengths[0]), torch.nn.Tanh()]
        for i in range(len(hidden_layer_lengths) - 1):
            self.model_layers.append(torch.nn.Linear(self.hidden_layer_lengths[i], self.hidden_layer_lengths[i + 1]))
            self.model_layers.append(torch.nn.Tanh())
        self.model_layers.append(torch.nn.Linear(self.hidden_layer_lengths[-1], 3))
        self.model = torch.nn.Sequential(*self.model_layers)

    def forward(self, points: torch.Tensor):

        return self.model(points)
# =============================================================================

class MinSurface():
    """
    A class defining the type of object that will be used to fit the point cloud by a model
    that minimizes the elastic energy of a surface while forcing the relevant points in the surface to
    agree with those in the point cloud. It essentially interpolates to fill the space in between
    points with the locations that minimize the elastic energy of the configuration.
    """
    def __init__(self, point_cloud_2d: np.ndarray, point_cloud_3d: np.ndarray, path, model, num_points=1000):
        self.point_cloud_2d = point_cloud_2d
        self.point_cloud_3d = point_cloud_3d
        self.fitting_points = torch.tensor(point_cloud_2d, dtype=torch.float32, requires_grad=True)
        self.target_points = torch.tensor(point_cloud_3d, dtype=torch.float32, requires_grad=True)
        self.path = path
        self.model = model
        self.num_points = num_points
# -----------------------------------------------------------------------------

    def e_b_terms_from_vgh(self,  g_x, g_y, g_z, h_x, h_y, h_z):
        """
        Find the terms that are going to be used in the elastic energy function.
        We need to keep in mind that when we train the model, the vales for the gradients
        and hessians, don't get updated in the surface object, unless we call the grad_and_hess
        method again.
        :return:
        Return the two terms of the energy.
        """

        # first, the jacobian, of shape batch*2*3
        jac = torch.stack((g_x, g_y, g_z), dim=2) # stack the gradients along the third direction to form a deformation "tensor".
        # Now, the strain, which is (J^T J - I)/2
        # strain = (torch.swapaxes(jac, 1, 2)@jac - torch.eye(3)[None, :, :])/2  # J^T J is 3x3, I is 3x3
        strain_2 = (jac@torch.swapaxes(jac, 1, 2) - torch.eye(2))/2  # J J^T is 2x2, I is 2x2. The @ operator stands for matrix
        # matrix multiplication where if the arguments are n dimensional it treats them as stacks of matrices residing in the
        # last two indexes and broadcast accordingly.

        # Compute the normal vectors by expressing the cross product in components.
        # first the normal, n = r_X x r_Y / norm(...)
        # so, nx = r_Xy r_Yz - r_Xz r_Yy = g_y[0]*g_z[1] - g_z[0]*g_y[1]
        nx = g_y[:, 0]*g_z[:, 1] - g_z[:, 0]*g_y[:, 1]
        # ny = r_Xz r_Yx - r_Xx r_Yz = g_z[0]*g_x[1] - g_x[0]*g_z[1]
        ny = g_z[:, 0]*g_x[:, 1] - g_x[:, 0]*g_z[:, 1]
        # nz = r_Xx r_Yy - r_Xy r_Yx = g_x[0]*g_y[1] - g_y[0]*g_x[1]
        nz = g_x[:, 0]*g_y[:, 1] - g_y[:, 0]*g_x[:, 1]
        # todo make sure these are all right

        # Normalize the length of the normal vector.
        inv_norm = 1/torch.norm(torch.stack((nx, ny, nz), dim=1), dim=1)
        nx = nx*inv_norm
        ny = ny*inv_norm
        nz = nz*inv_norm

        # Compute the 2nd fundamental form.
        # now, b_ij = n_k d_i d_j r_k
        # b = torch.zeros((vals.shape[0], 2, 2), dtype=vals.dtype)
        # for i, j in itertools.product(range(2), range(2)):
        #     b[:, i, j] = nx*h_x[:, i, j] +\
        #                  ny*h_y[:, i, j] +\
        #                  nz*h_z[:, i, j]
        b = nx[:, None, None]*h_x +\
            ny[:, None, None]*h_y +\
            nz[:, None, None]*h_z


        # Compute the E_term and b_term. NEED TO CHECK THESE.
        # E_term = (strain[:, 0, 0] + strain[:, 1, 1] + strain[:, 2, 2])**2 + torch.sum(strain**2, dim=(1, 2))
        E_term = (strain_2[:, 0, 0] + strain_2[:, 1, 1])**2 + torch.sum(strain_2**2, dim=(1, 2))
        b_term = (b[:, 0, 0] + b[:, 0, 0])**2 + torch.sum(b**2, dim=(1, 2))

        return E_term, b_term


    def grad_and_hess(self, val, points):
        # returns the gradient and hessian of val (shape (batch)) with respect to points (shape (batch, 2))
        grad = torch.autograd.grad(val, points, grad_outputs=torch.ones_like(val), create_graph=True)[0]
        # The function first calculates the gradient grad of the function val with respect to the input points.
        # It uses PyTorch's torch.autograd.grad function.

        g_X = grad[:, 0]  # derivative of val with respect to X
        g_Y = grad[:, 1]  # derivative of val with respect to Y
        grad_X = torch.autograd.grad(g_X, points, grad_outputs=torch.ones_like(g_X), create_graph=True)[0]
        grad_Y = torch.autograd.grad(g_Y, points, grad_outputs=torch.ones_like(g_Y), create_graph=True)[0]

        # assert torch.allclose(grad_X[:, 1], grad_Y[:, 0]), 'mixed derivatives should be equal'

        hess_x = torch.stack((grad_X, grad_Y), dim=2)  # shape is (batch, 2, 2)
        # assert torch.allclose(hess_x[:, 0, 1], hess_x[:, 1, 0]), 'mixed derivatives should be equal'  # sanity check

        # print(f'{grad_x.shape=}')
        # print(f'{hess_x.shape=}')
        # 1 / 0

        return grad, hess_x

    # Function for calculating the loss function (elastic energy) for the 1000 points together.
    def bulk_loss(self):
        factor = config.FACTOR
        # take points, calculate vals, gradients, and hessians, and calculate the energy by
        # plugging this into e_b_terms_from_vgh
        points = torch.tensor(self.points, dtype=torch.float32, requires_grad=True)  # we need the gradients and hessians
        # The function begins by converting the input points (2D coordinates) into a PyTorch tensor.
        # requires_grad=True is set to ensure that gradients can be calculated with respect to these points.
        vals = self.model(points)  # The neural network model is applied to the input points, resulting in a tensor vals containing the model's predictions.
        # The values are split into their X, Y, and Z components.
        x = vals[:, 0]
        y = vals[:, 1]
        z = vals[:, 2]

        # The gradient and Hessian matrices are calculated for each of the X, Y, and Z components using the grad_and_hess function.
        # This involves finding the first and second-order derivatives of each component with respect to the input points.
        g_x, h_x = self.grad_and_hess(x, points)
        g_y, h_y = self.grad_and_hess(y, points)
        g_z, h_z = self.grad_and_hess(z, points)

        E_term, b_term = self.e_b_terms_from_vgh(g_x, g_y, g_z, h_x, h_y, h_z)

        # The elastic energy is calculated as the mean of the energy term E_term plus a scaled mean of the b-term b_term.
        elastic_energy = torch.mean(E_term) + factor * torch.mean(b_term)

        tot_energy = elastic_energy

        return tot_energy

    def fitting_loss(self):
        fitting_points = self.fitting_points
        target_points = self.target_points
        vals = self.model(fitting_points)  # The neural network model is applied to the boundary_points, resulting in vals,
        # which contains the model's predictions for these points.
        fitting_cost = torch.sum((vals - target_points) ** 2)  # sum to make this very important
        return fitting_cost

    # A closure function closure is defined. It is used to compute the loss and its gradients, and also to update curr_loss.
    def closure(self):
        # The gradients of the model parameters are cleared at the beginning of each iteration.
        self.opt.zero_grad()

        # The bulk_loss function is called to compute the bulk loss, which measures the deviation of predicted values
        # from target values. The result is stored in l.
        l = self.bulk_loss()
        l_fit = self.fitting_loss()

        tot_l = l + config.FITTING_TO_ENERGY_RATIO * l_fit  # + 50*l_b + l_f + 0.05 * l_w  # boundary loss is much more important

        # The gradients of the total loss with respect to the model parameters are computed.
        tot_l.backward()
        # The current value of the bulk loss is stored in the curr_loss variable.
        self.curr_loss = l.item()
        self.tot_l_global = tot_l.item()
        return tot_l

    def fit(self):
        # opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        #  An instance of the LBFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno) optimization algorithm is created.
        #  It will be used to optimize the parameters of the model (neural network). Various settings, such as learning rate,
        #  maximum iterations, and line search method, are specified.
        self.opt = torch.optim.LBFGS(self.model.parameters(), lr=1e-2, max_iter=100, max_eval=100, history_size=100,
                                line_search_fn='strong_wolfe')
        # A variable curr_loss is initialized with positive infinity. This variable will be used to keep track of
        # the current loss during optimization.
        self.curr_loss = torch.inf
        self.tot_l_global = torch.inf

        closure = self.closure
        # A loop iterates over a range of values (30 in this case). This loop represents the optimization process.
        for i in range(30):  # ~30 works for lbfgs with the current settings, ~3000 for adam
            self.points = sample_low_disc_seq(self.num_points, self.path)  # or locally (better for convergence but bad for lbfgs)

            # The opt.step(closure) statement performs a single optimization step using the LBFGS optimizer.
            # It calls the closure function, which computes the loss and gradients.
            self.opt.step(closure)
            # if i % 100 == 0:
            print(str(i).zfill(5), self.curr_loss)  # print only the bulk loss, i.e. the elastic energy
            print(str(i).zfill(5), self.tot_l_global)

        return self.model
# =============================================================================

def flip_and_normalize(point_cloud, width=0.5, original_scaling_factor=None, original_min_val_x=None,
                       original_min_val_y=None):
    # Transpose the point cloud to flip it counterclockwise by 90 degrees
    flipped_cloud = np.array([point_cloud[:, 1], -point_cloud[:, 0]]).T
    flipped_cloud_x = flipped_cloud[:, 0]
    flipped_cloud_y = flipped_cloud[:, 1]

    # Normalize the longer dimension to fall between 0 and 1
    max_val_y = np.max(flipped_cloud_y)
    min_val_y = np.min(flipped_cloud_y)
    min_val_x = np.min(flipped_cloud_x)
    max_val_x = np.max(flipped_cloud_x)
    point_cloud_width = (max_val_x - min_val_x)

    # Setting the scaling factor so it is calculated for the first point cloud,
    # but we keep using the same value for the latter ones, since all point clouds were fitted to the same base.
    if original_scaling_factor is None:
        scaling_factor = width / point_cloud_width
    else:
        scaling_factor = original_scaling_factor
    if original_min_val_x is not None:
        min_val_x = original_min_val_x
        min_val_y = original_min_val_y
    normalized_cloud_x = (flipped_cloud_x - min_val_x) * scaling_factor
    normalized_cloud_y = (flipped_cloud_y - min_val_y) * scaling_factor + 0.09
    normalized_cloud = np.stack([normalized_cloud_x, normalized_cloud_y], axis=1)

    print(f"SCALING FACTOR = {scaling_factor}")
    print(f"width of blueprint = {(max_val_x - min_val_x)}")

    return normalized_cloud, scaling_factor, min_val_x, min_val_y

def sample_low_disc_seq(n, path):
    """
    Sample a low discrepancy sequence of points within a certain enveloping shape.
    :param n: The number of points to be sampled
    :param path: The svg path of the enveloping shape (in our case probably a leaf).
    :return:
    Return the sequence of points as a numpy ndarray.
    """
    bbox = path.get_extents()  # Obtain the bounding box of the path, an imaginary rectangular box that completely encloses a geometric shape or a set of points.
    # An affine transformation (norm_trans) is created to translate the path such that its top-left corner is at the origin (0, 0) and scale it so that the larger dimension is normalized to 1.
    norm_trans = mpl.transforms.Affine2D().translate(-bbox.x0, -bbox.y0).scale(1 / max(bbox.width, bbox.height))
    # The transformation is applied to the path using transform_path, and the bounding box is recalculated. The width and height of the bounding box are stored in the tuple wh.
    path = norm_trans.transform_path(path)
    bbox = path.get_extents()
    wh = (bbox.width, bbox.height)
    sampling_j = int(np.argmin(wh))  # index of the shortest side of the bounding box
    sampling_bbox_len = wh[sampling_j]  # length of the shortest side of the bounding box
    # sample from the leaf using a low discrepancy sequence
    inverse_plastic_ratio = 1 / 1.324717957244746025960908854
    gen = np.random.default_rng(42)  # just to set the first point differently each time
    init = gen.random(2)  # Get two random values between 0 an 1.
    x = np.arange(n) * inverse_plastic_ratio + init[0]  # Get the values of x for the points by getting a string from 0 to n-1, multiplying by plastic ratio and adding the first random num.
    y = np.arange(n) * inverse_plastic_ratio ** 2 + init[1]
    x = x % 1  # Make the x or y values stay inside tha unit square by using modulo.
    y = y % 1

    points = np.stack((x, y), axis=1)

    points[:, sampling_j] *= sampling_bbox_len  # scale the shortest side of the bounding box (maintains uniformity)

    points = points[path.contains_points(points)]  # filter out points outside the leaf

    return points

def sample_square_lattice(shape_num_points: tuple, path: matplotlib.patches.Path):
    """
    Sample a square lattice of points within a certain enveloping shape.
    :param shape_num_points: A tuple containing the number of points on each side.
    :param path: The svg path of the enveloping shape (in our case probably a leaf).
    :return:
    Return the sequence of points as a numpy array.
    """
    bbox = path.get_extents()  # Obtain the bounding box of the path, an imaginary rectangular box that completely encloses a geometric shape or a set of points.
    # An affine transformation (norm_trans) is created to translate the path such that its top-left corner is at the origin (0, 0) and scale it so that the larger dimension is normalized to 1.
    norm_trans = mpl.transforms.Affine2D().translate(-bbox.x0, -bbox.y0).scale(1 / max(bbox.width, bbox.height))
    # The transformation is applied to the path using transform_path, and the bounding box is recalculated. The width and height of the bounding box are stored in the tuple wh.
    path = norm_trans.transform_path(path)
    bbox = path.get_extents()
    # Calculate the spacing between points along each dimension
    num_points_width, num_points_height = shape_num_points
    total_width, total_height = bbox.width, bbox.height
    spacing_width = total_width / (num_points_width - 1)
    spacing_height = total_height / (num_points_height - 1)

    # Create an array of indices for the points along each dimension
    indices_width = np.arange(num_points_width)
    indices_height = np.arange(num_points_height)

    # Generate grid coordinates using meshgrid
    xv, yv = np.meshgrid(indices_width * spacing_width, indices_height * spacing_height)

    # Reshape the coordinates into a list of (x, y) points
    points = np.vstack((xv.flatten(), yv.flatten())).T

    points = points[path.contains_points(points)]

    return points
# -----------------------------------------------------------------------------

# Function for plotting the result of the transformation of the model being applied to num_points points sampled
# from a low discrepancy sequence (a model surface).
def plot_model(path, model,save_orbit_animation=False, num_points=3000):
    # plot num_points leaf points using pyvista (and save an orbit animation if requested)
    # we also plot the points used for the B.C (they look like a solid line along the leaf base)
    points = sample_low_disc_seq(num_points, path) # Sample the points from a low discrepancy series.

    # Put the points in the model (neural network) to get the output.
    vals = model(torch.tensor(points, dtype=torch.float32)).detach().numpy()

    # Set up a PyVista plotter and configure the settings.
    pv.set_plot_theme('dark')
    plotter = pv.Plotter()
    plotter.enable_terrain_style(mouse_wheel_zooms=True)
    plotter.enable_anti_aliasing()

    # Create a mesh that represents the data to be visualized, choose the colors and add it to the plotter.
    mesh = pv.PolyData(vals)
    points -= np.min(points, axis=0); points /= np.max(points, axis=0)  # normalize points to between 0 and 1.
    rgb = np.stack((points[:, 0], points[:, 1], np.ones(points.shape[0])*.8), axis=1) # Assign points to be colors.
    mesh['color'] = rgb
    # pv.plot(mesh, show_bounds=True, eye_dome_lighting=True)
    plotter.add_mesh(mesh, point_size=20,
                     render_points_as_spheres=True,
                     scalars='color', rgb=True,
                     )

    # enable eye_dome_lighting
    plotter.enable_eye_dome_lighting()

    # enable axes, with a large font size
    plotter.show_grid()
    plotter.show_bounds(all_edges=True, font_size=16, color='white', location='outer')

    if not save_orbit_animation:
        plotter.show(auto_close=False)
    else:
        path = plotter.generate_orbital_path(n_points=36, shift=mesh.length)
        # plotter.open_gif("orbit.gif")
        plotter.open_movie("orbit.mp4")
        plotter.orbit_on_path(path, write_frames=True)
        plotter.close()


# -----------------------------------------------------------------------------

# Function for plotting the result of the transformation of the model being applied to num_points points sampled
# from a low discrepancy sequence (a model surface).
def plot_model_plus_target(path, model, file_path_2d_param: str, file_path_3d_param: str, save_orbit_animation=False, num_points=3000,
                           scaling_factor = None, min_val_x = None, min_val_y = None):
    # This defines a Scalable Vector Graphics image which corresponds to the (i think flat) leaf.
    # You can just get it by parsing the image from Michal.

    # Files containing the point clouds to be fitted.
    file_path_2d = file_path_2d_param
    file_path_3d = file_path_3d_param

    point_cloud_2d1 = np.loadtxt(file_path_2d, delimiter=',', skiprows=1)
    point_cloud_3d1 = np.loadtxt(file_path_3d, delimiter=',', skiprows=1)

    point_cloud_2d = point_cloud_2d1
    # pcd = o3d.io.read_point_cloud(file_path_3d)
    point_cloud_3d = point_cloud_3d1  # np.asarray(pcd.points) * scaling_factor

    print(f"point cloud 2d = {point_cloud_2d}")
    print(f"point cloud 3d = {point_cloud_3d}")

    # sample points from a low discrepancy sequence
    # -----------------------------------------------------------------------------
    bbox = path.get_extents()  # Obtain the bounding box of the path, an imaginary rectangular box that completely encloses a geometric shape or a set of points.
    # An affine transformation (norm_trans) is created to translate the path such that its top-left corner is at the origin (0, 0) and scale it so that the larger dimension is normalized to 1.
    norm_trans = mpl.transforms.Affine2D().translate(-bbox.x0, -bbox.y0).scale(1 / max(bbox.width, bbox.height))
    # The transformation is applied to the path using transform_path, and the bounding box is recalculated. The width and height of the bounding box are stored in the tuple wh.
    path = norm_trans.transform_path(path)
    bbox = path.get_extents()
    wh = (bbox.width, bbox.height)
    sampling_j = int(np.argmin(wh))  # index of the shortest side of the bounding box
    sampling_bbox_len = wh[sampling_j]  # length of the shortest side of the bounding box

    point_cloud_2d, scaling_factor, min_val_x, min_val_y = flip_and_normalize(point_cloud_2d, wh[0], original_scaling_factor=scaling_factor,
                                                        original_min_val_x=min_val_x, original_min_val_y=min_val_y)
    point_cloud_3d *= scaling_factor
    point_cloud_2d_x = point_cloud_2d[:, 0]
    point_cloud_2d_y = point_cloud_2d[:, 1]

    # # ----------------- PLOT FOR TEST -----------------
    # if config.SHOW_PLOTS == True:
    #     # Create a figure and axis
    #     fig, ax = plt.subplots()
    #     # for plotting
    #     patch = PathPatch(path, facecolor="none", lw=2)
    #     ax.add_patch(patch)
    #     # Plot the point where we will apply the force in blue.
    #     # ax.scatter(wh[0]/2, leaf_force_height, color="blue", label="Blue Point")
    #     # ax.scatter(wh[0]/2, leaf_force_height2, color="blue", label="Blue Point")
    #     # ax.scatter(wh[0]/2, leaf_weight_height, color="blue", label="Blue Point")
    #     # ax.scatter(wh[0]/2, leaf_weight_height2, color="blue", label="Blue Point")
    #     # Plot the point cloud.
    #     ax.scatter(point_cloud_2d_x, point_cloud_2d_y, color="blue", label="Blue Point")
    #     # Set axis limits and labels
    #     # ax.set_xlim(0, wh[0])
    #     # ax.set_ylim(0, wh[1])
    #     ax.set_aspect('equal')
    #     ax.set_xlabel("X")
    #     ax.set_ylabel("Y")
    #     # Add a legend
    #     ax.legend("plot for test")
    #     # Show the plot
    #     plt.show()
    # # -------------- END PLOT FOR TEST -----------------

    # now, the model of r(X, Y):
    # Essentially, it is a model with 2 inputs, 3 outputs and two inner layers with 30 neurons. The activation values
    # are obtained at every step by using tanh of the linear combinations of the previous activations with the weights
    # and the bias.
    # build a ForceFinderModel object that takes one force and has two hidden layers with 50 nodes each.

    surface = MinSurface(point_cloud_2d, point_cloud_3d, path, model)

    plot_mean_curvature(path, surface, point_cloud_3d, point_cloud_2d, model)

    # # plot num_points leaf points using pyvista (and save an orbit animation if requested)
    # # we also plot the points of the 3d point cloud used for fitting the model
    # points = sample_low_disc_seq(num_points, path) # Sample the points from a low discrepancy series.
    #
    # point_cloud_3d1 = np.loadtxt(target_point_cloud_3d_path, delimiter=',', skiprows=1)
    #
    #
    # # Put the points in the model (neural network) to get the output.
    # vals = model(torch.tensor(points, dtype=torch.float32)).detach().numpy()
    #
    # # Set up a PyVista plotter and configure the settings.
    # pv.set_plot_theme('dark')
    # plotter = pv.Plotter()
    # plotter.enable_terrain_style(mouse_wheel_zooms=True)
    # plotter.enable_anti_aliasing()
    #
    # rgb1 = np.stack((np.ones(point_cloud_3d1.shape[0]), np.ones(point_cloud_3d1.shape[0]), np.ones(point_cloud_3d1.shape[0])), axis=1)
    # mesh1 = pv.PolyData(point_cloud_3d1)
    # mesh1['color'] = rgb1
    #
    # # Create a mesh that represents the data to be visualized, choose the colors and add it to the plotter.
    # mesh = pv.PolyData(vals)
    # points -= np.min(points, axis=0); points /= np.max(points, axis=0)  # normalize points to between 0 and 1.
    # rgb = np.stack((points[:, 0], points[:, 1], np.ones(points.shape[0])*.8), axis=1) # Assign points to be colors.
    # mesh['color'] = rgb
    # # pv.plot(mesh, show_bounds=True, eye_dome_lighting=True)
    # plotter.add_mesh(mesh, point_size=20,
    #                  render_points_as_spheres=True,
    #                  scalars='color', rgb=True,
    #                  )
    #
    # plotter.add_mesh(mesh1, point_size=20,
    #                  render_points_as_spheres=True,
    #                  scalars='color', rgb=True,
    #                  )
    #
    # # enable eye_dome_lighting
    # plotter.enable_eye_dome_lighting()
    #
    # # enable axes, with a large font size
    # plotter.show_grid()
    # plotter.show_bounds(all_edges=True, font_size=16, color='white', location='outer')
    #
    # if not save_orbit_animation:
    #     plotter.show(auto_close=False)
    # else:
    #     path = plotter.generate_orbital_path(n_points=36, shift=mesh.length)
    #     # plotter.open_gif("orbit.gif")
    #     plotter.open_movie("orbit.mp4")
    #     plotter.orbit_on_path(path, write_frames=True)
    #     plotter.close()

    return scaling_factor, min_val_x, min_val_y


# ------------------------------ FUNCTIONS FOR CURVATURE -----------------------------------
class Calculator():
    """
    Defines a class that will perform the relevant calculations on a surface.
    """
    def __init__(self, surface: MinSurface, model: ForceFinderModel = None):
        self.surface = surface
        if model is not None:
            self.model = model
        else:
            self.model = surface.model


# Calculate the mean curvature for the surface.
    def first_fund_form(self, points: np.ndarray):
        # Get the values that the model spits for the given points.
        points = torch.tensor(points, dtype=torch.float32, requires_grad=True)
        vals = self.model(points)

        # The values are split into their X, Y, and Z components.
        x = vals[:, 0]
        y = vals[:, 1]
        z = vals[:, 2]

        # The gradient and Hessian matrices are calculated for each of the X, Y, and Z components using the grad_and_hess function.
        # This involves finding the first and second-order derivatives of each component with respect to the input points.
        g_x, h_x = self.surface.grad_and_hess(x, points)
        g_y, h_y = self.surface.grad_and_hess(y, points)
        g_z, h_z = self.surface.grad_and_hess(z, points)

        g_x_1 = g_x[:, 0]
        g_x_2 = g_x[:, 1]
        g_y_1 = g_y[:, 0]
        g_y_2 = g_y[:, 1]
        g_z_1 = g_z[:, 0]
        g_z_2 = g_z[:, 1]

        a11 = g_x_1 * g_x_1 + g_y_1 * g_y_1 + g_z_1 * g_z_1
        a12 = g_x_1 * g_x_2 + g_y_1 * g_y_2 + g_z_1 * g_z_2
        a22 = g_x_2 * g_x_2 + g_y_2 * g_y_2 + g_z_2 * g_z_2

        a1 = torch.stack((a11, a12), dim=-1)
        a2 = torch.stack((a12, a22), dim=-1)
        a = torch.stack((a1, a2), dim=-1)
        return a

    def second_fund_form(self, points: np.ndarray):
        # Get the values that the model spits for the given points.
        points = torch.tensor(points, dtype=torch.float32, requires_grad=True)
        vals = self.model(points)

        # The values are split into their X, Y, and Z components.
        x = vals[:, 0]
        y = vals[:, 1]
        z = vals[:, 2]

        # The gradient and Hessian matrices are calculated for each of the X, Y, and Z components using the grad_and_hess function.
        # This involves finding the first and second-order derivatives of each component with respect to the input points.
        g_x, h_x = self.surface.grad_and_hess(x, points)
        g_y, h_y = self.surface.grad_and_hess(y, points)
        g_z, h_z = self.surface.grad_and_hess(z, points)

        # Compute the normal vectors by expressing the cross product in components.
        # first the normal, n = r_X x r_Y / norm(...)
        # so, nx = r_Xy r_Yz - r_Xz r_Yy = g_y[0]*g_z[1] - g_z[0]*g_y[1]
        nx = g_y[:, 0] * g_z[:, 1] - g_z[:, 0] * g_y[:, 1]
        # ny = r_Xz r_Yx - r_Xx r_Yz = g_z[0]*g_x[1] - g_x[0]*g_z[1]
        ny = g_z[:, 0] * g_x[:, 1] - g_x[:, 0] * g_z[:, 1]
        # nz = r_Xx r_Yy - r_Xy r_Yx = g_x[0]*g_y[1] - g_y[0]*g_x[1]
        nz = g_x[:, 0] * g_y[:, 1] - g_y[:, 0] * g_x[:, 1]
        # todo make sure these are all right

        # Normalize the length of the normal vector.
        inv_norm = 1 / torch.norm(torch.stack((nx, ny, nz), dim=1), dim=1)
        nx = nx * inv_norm
        ny = ny * inv_norm
        nz = nz * inv_norm

        # Compute the 2nd fundamental form.
        # now, b_ij = n_k d_i d_j r_k
        # b = torch.zeros((vals.shape[0], 2, 2), dtype=vals.dtype)
        # for i, j in itertools.product(range(2), range(2)):
        #     b[:, i, j] = nx*h_x[:, i, j] +\
        #                  ny*h_y[:, i, j] +\
        #                  nz*h_z[:, i, j]
        b = nx[:, None, None] * h_x + \
            ny[:, None, None] * h_y + \
            nz[:, None, None] * h_z

        return b

    def shape_operator(self, points: np.ndarray):
        a = self.first_fund_form(points)
        b = self.second_fund_form(points)
        a_inv = torch.inverse(a)
        shape_operator = torch.matmul(b, a_inv)
        return shape_operator

    def mean_curvature(self, points: np.ndarray):
        shape_op = self.shape_operator(points)
        mean_curvature = shape_op.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) / 2
        return mean_curvature
# -----------------------------------------------------------------------------
def plot_mean_curvature(path, surface, point_cloud_3d, point_cloud_2d, model, num_points=3000):
    # plot num_points leaf points using pyvista (and save an orbit animation if requested)
    # we also plot the points used for the B.C (they look like a solid line along the leaf base)
    points = sample_low_disc_seq(num_points, path)  # Sample the points from a low discrepancy series.

    calculator = Calculator(surface)
    # --------------
    # for plotting the actual point clouds
    point_cloud_3d1 = point_cloud_3d
    point_cloud_2d2 = point_cloud_2d
    point_cloud_2d2 = np.hstack((point_cloud_2d2, np.zeros((point_cloud_2d2.shape[0], 1))))

    rgb1 = np.stack((np.ones(point_cloud_3d1.shape[0]), np.ones(point_cloud_3d1.shape[0]), np.ones(point_cloud_3d1.shape[0])), axis=1)
    rgb2 = np.stack((np.ones(point_cloud_2d2.shape[0]), np.ones(point_cloud_2d2.shape[0]), np.ones(point_cloud_2d2.shape[0])), axis=1)
    mesh1 = pv.PolyData(point_cloud_3d1)
    mesh2 = pv.PolyData(point_cloud_2d2)
    mesh1['color'] = rgb1
    mesh2['color'] = rgb2
    # --------------

    # Put the points in the model (neural network) to get the output.
    vals = model(torch.tensor(points, dtype=torch.float32)).detach().numpy()

    mean_curv = calculator.mean_curvature(points).detach().numpy()

    # Set up a PyVista plotter and configure the settings.
    pv.set_plot_theme('dark')
    plotter = pv.Plotter()
    plotter.enable_terrain_style(mouse_wheel_zooms=True)
    plotter.enable_anti_aliasing()

    # Create a mesh that represents the data to be visualized, choose the colors and add it to the plotter.
    mesh = pv.PolyData(vals)
    points -= np.min(points, axis=0)
    points /= np.max(points, axis=0)  # normalize points to between 0 and 1.

    # Create a colormap instance
    colormap = colormaps['coolwarm']

    # Normalize the 'mean_curv' values to the range [0, 1]
    normalized_mean_curv = (mean_curv - mean_curv.min()) / (mean_curv.max() - mean_curv.min())

    # Map normalized 'mean_curv' values to RGBA colors using the colormap
    rgb = colormap(normalized_mean_curv)

    # rgb = np.stack((mean_curv, np.ones(points.shape[0]) * .3, np.ones(points.shape[0]) * .8), axis=1)  # Assign points to be colors.
    mesh['color'] = rgb
    # pv.plot(mesh, show_bounds=True, eye_dome_lighting=True)
    plotter.add_mesh(mesh, point_size=20,
                     render_points_as_spheres=True,
                     scalars='color', rgb=True,
                     )


    # --------------------
    # for plotting the actual point clouds
    plotter.add_mesh(mesh1, point_size=20,
                     render_points_as_spheres=True,
                     scalars='color', rgb=True,
                     )
    plotter.add_mesh(mesh2, point_size=20,
                     render_points_as_spheres=True,
                     scalars='color', rgb=True,
                     )
    # --------------------

    # enable eye_dome_lighting
    plotter.enable_eye_dome_lighting()

    # enable axes, with a large font size
    plotter.show_grid()
    plotter.show_bounds(all_edges=True, font_size=16, color='white', location='outer')


    plotter.show(auto_close=False)



# Define the sigmoid function using torch
def sigmoid(x, range_=6):
    return 1 / (1 + torch.exp(-range_ * x))

# Normalize the values using the sigmoid function
def normalize_values_torch(values, range):
    values_tensor = torch.tensor(values, dtype=torch.float32)
    # Adjust the range based on your desired mapping
    sigmoid_range = range  # Adjust as needed
    normalized_values = sigmoid((values_tensor - torch.mean(values_tensor)) / torch.std(values_tensor), range_=sigmoid_range)
    return normalized_values.numpy()

def plot_and_save_mean_curvature_2d(path, surface, save_plot=0, num_points=3000):
    points = sample_low_disc_seq(num_points, path)  # Sample the points from a low discrepancy series.
    # points1 = torch.tensor(points, dtype=torch.float32)

    calculator = Calculator(surface)

    # Put the points in the model (neural network) to get the output.
    # vals = model(torch.tensor(points, dtype=torch.float32)).detach().numpy()

    mean_curv = calculator.mean_curvature(points).detach().numpy()

    # Create a mesh that represents the data to be visualized, choose the colors and add it to the plotter.
    point_cloud_2d = np.hstack((points, np.zeros((points.shape[0], 1))))

    # points -= np.min(points, axis=0)
    # points /= np.max(points, axis=0)  # normalize points to between 0 and 1.

    # Create a colormap instance
    colormap = colormaps['coolwarm']

    # Normalize the 'mean_curv' values to the range [0, 1]
    # normalized_mean_curv = (mean_curv - mean_curv.min()) / (mean_curv.max() - mean_curv.min())
    range = mean_curv.max() - mean_curv.min()
    range = 1
    normalized_mean_curv = normalize_values_torch(mean_curv, range)

    # Create a scatter plot of the points with colors mapped to values
    scatter = plt.scatter(points[:, 0], points[:, 1], c=normalized_mean_curv, cmap='coolwarm', s=50, vmin=0, vmax=1)

    # Add a colorbar
    plt.colorbar(scatter, label='Mean curvature')

    # Set plot title and labels (optional)
    plt.title('2D Point Cloud with Colorbar')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.axis('equal')
    if save_plot != 0:
        plt.savefig(save_plot)
    if config.SHOW_PLOTS == True:
        # Show the plot
        plt.show()
    else:
        plt.clf()


def load_model(model_path: str):
    """
    Function for loading a model that has been pre-trained for a specific leaf.
    :param model_path: The path of the file where the model is saved
    :return:
    Return the model.
    """
    model = ForceFinderModel((30, 30))
    model.load_state_dict(torch.load(model_path))
    return model

def average_values_along_y(f_values, coordinates):
    """
    Compute the average values of 'f' corresponding to the same y-coordinate.

    Args:
    - f_values (numpy.ndarray): Array of values corresponding to the field on the surface.
    - coordinates (numpy.ndarray): Array of coordinates (x, y) where 'f' values occur.

    Returns:
    - unique_y (numpy.ndarray): Unique y-coordinates.
    - averaged_values (numpy.ndarray): Averaged 'f' values corresponding to unique y-coordinates.
    """
    # Get y-coordinates from the coordinates array
    y_coords = coordinates[:, 1]

    # Find unique y-coordinates and their indices
    unique_y, indices = np.unique(y_coords, return_inverse=True)

    # Use np.bincount to accumulate 'f' values based on y-coordinate indices
    accumulated_values = np.bincount(indices, weights=f_values)

    # Compute the count of 'f' values corresponding to each unique y-coordinate
    count_values = np.bincount(indices)

    # Compute the averaged 'f' values for each unique y-coordinate
    averaged_values = accumulated_values / count_values

    return unique_y, averaged_values


def load_models_and_plot_average_curvature(base_path: str, number_of_frames: int, path: matplotlib.patches.Path, offset: int):
    """
    Function for plotting the horizontal average of the curvature for one leaf over the time of the experiment. The
    function assumes that the leaves have been previously fitted by models, which are saved in their respective folders.
    :param base_path: The base path for the folder where the models are.
    :param number_of_frames: The number of frames for which to plot the curvature.
    :return:
    Return 0 for success.
    """
    points = sample_square_lattice((50, 100), path)
    averaged_values = []

    for i in range(number_of_frames):
        j = i + offset
        model_path = os.path.join(base_path, "Session_" + str(j), "fitting_model_red.pt")
        file_path_2d_current = os.path.join(base_path, "Session_" + str(j), "pcd_2d.csv")
        file_path_3d_current = os.path.join(base_path, "Session_" + str(j), "pcd_3d.csv")
        point_cloud_2d = np.loadtxt(file_path_2d_current, delimiter=',', skiprows=1)
        point_cloud_3d = np.loadtxt(file_path_3d_current, delimiter=',', skiprows=1)

        model = load_model(model_path)
        surface = MinSurface(point_cloud_2d, point_cloud_3d, path, model)

        calculator = Calculator(surface)

        mean_curv = calculator.mean_curvature(points).detach().numpy()

        unique_y, mean_curv_x_average = average_values_along_y(mean_curv, points)

        averaged_values.append(mean_curv_x_average)

    # Find the minimum and maximum values among all arrays
    min_value = min(np.min(arr) for arr in averaged_values)
    max_value = max(np.max(arr) for arr in averaged_values)

    # Subtract the minimum value and normalize the average values
    normalized_values = [(arr - min_value) / (max_value - min_value) for arr in averaged_values]

    # Create meshgrid for plotting
    # X, Y = np.meshgrid(np.arange(1, len(normalized_values) + 1), unique_y)
    X, Y = np.meshgrid(np.arange(offset, len(normalized_values) + offset), unique_y)

    # Flatten the meshgrid coordinates
    x_coords = X.flatten()
    y_coords = Y.flatten()
    # colors = np.array([val for arr in normalized_values for val in arr])  # Flatten and concatenate normalized values
    colors = np.column_stack(normalized_values)

    # Plotting
    plt.scatter(x_coords, y_coords, c=colors, cmap='coolwarm', edgecolor='none')
    plt.colorbar(label='Normalized Average Value')

    # Add text annotations to the colorbar indicating min and max values
    plt.text(1.05, 0, f'{min_value:.2f}', transform=plt.gca().transAxes, va='center')
    plt.text(1.05, 1, f'{max_value:.2f}', transform=plt.gca().transAxes, va='center')

    plt.xlabel('Frame number')
    plt.ylabel('Y Values')
    plt.title('2D Scatter Plot with Normalized Average Values (Overall Min-Max)')
    plt.show()

    return 0

def fit_and_plot_curvature(file_path_2d_param: str, file_path_3d_param: str, save_plot: int=0, save_model=0,
                           scaling_factor = None, min_val_x = None, min_val_y = None):

    # This defines a Scalable Vector Graphics image which corresponds to the (i think flat) leaf.
    # You can just get it by parsing the image from Michal.
    leaf_svg_path = config.LEAF_SVG_PATH

    # Files containing the point clouds to be fitted.
    file_path_2d = file_path_2d_param
    file_path_3d = file_path_3d_param

    point_cloud_2d1 = np.loadtxt(file_path_2d, delimiter=',', skiprows=1)
    point_cloud_3d1 = np.loadtxt(file_path_3d, delimiter=',', skiprows=1)


    point_cloud_2d = point_cloud_2d1
    # pcd = o3d.io.read_point_cloud(file_path_3d)
    point_cloud_3d = point_cloud_3d1  # np.asarray(pcd.points) * scaling_factor

    print(f"point cloud 2d = {point_cloud_2d}")
    print(f"point cloud 3d = {point_cloud_3d}")

    # sample points from a low discrepancy sequence
    # -----------------------------------------------------------------------------
    path = parse_path(leaf_svg_path)  # Parse the SVG path string and create a path object (path) that can be manipulated.
    bbox = path.get_extents()  # Obtain the bounding box of the path, an imaginary rectangular box that completely encloses a geometric shape or a set of points.
    # An affine transformation (norm_trans) is created to translate the path such that its top-left corner is at the origin (0, 0) and scale it so that the larger dimension is normalized to 1.
    norm_trans = mpl.transforms.Affine2D().translate(-bbox.x0, -bbox.y0).scale(1 / max(bbox.width, bbox.height))
    # The transformation is applied to the path using transform_path, and the bounding box is recalculated. The width and height of the bounding box are stored in the tuple wh.
    path = norm_trans.transform_path(path)
    bbox = path.get_extents()
    wh = (bbox.width, bbox.height)
    sampling_j = int(np.argmin(wh))  # index of the shortest side of the bounding box
    sampling_bbox_len = wh[sampling_j]  # length of the shortest side of the bounding box

    point_cloud_2d, scaling_factor, min_val_x, min_val_y = flip_and_normalize(point_cloud_2d, wh[0], original_scaling_factor=scaling_factor, original_min_val_x=min_val_x, original_min_val_y=min_val_y)
    point_cloud_3d *= scaling_factor
    point_cloud_2d_x = point_cloud_2d[:, 0]
    point_cloud_2d_y = point_cloud_2d[:, 1]

    # ----------------- PLOT FOR TEST -----------------
    if config.SHOW_PLOTS == True:
        # Create a figure and axis
        fig, ax = plt.subplots()
        # for plotting
        patch = PathPatch(path, facecolor="none", lw=2)
        ax.add_patch(patch)
        # Plot the point where we will apply the force in blue.
        # ax.scatter(wh[0]/2, leaf_force_height, color="blue", label="Blue Point")
        # ax.scatter(wh[0]/2, leaf_force_height2, color="blue", label="Blue Point")
        # ax.scatter(wh[0]/2, leaf_weight_height, color="blue", label="Blue Point")
        # ax.scatter(wh[0]/2, leaf_weight_height2, color="blue", label="Blue Point")
        # Plot the point cloud.
        ax.scatter(point_cloud_2d_x, point_cloud_2d_y, color="blue", label="Blue Point")
        # Set axis limits and labels
        # ax.set_xlim(0, wh[0])
        # ax.set_ylim(0, wh[1])
        ax.set_aspect('equal')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        # Add a legend
        ax.legend("plot for test")
        # Show the plot
        plt.show()
    # -------------- END PLOT FOR TEST -----------------

    # now, the model of r(X, Y):
    # Essentially, it is a model with 2 inputs, 3 outputs and two inner layers with 30 neurons. The activation values
    # are obtained at every step by using tanh of the linear combinations of the previous activations with the weights
    # and the bias.
    # build a ForceFinderModel object that takes one force and has two hidden layers with 50 nodes each.
    model = ForceFinderModel((30, 30))

    surface = MinSurface(point_cloud_2d, point_cloud_3d, path, model)
    model = surface.fit()

    if save_model != 0:
        torch.save(model.state_dict(), save_model)


    # calculator = Calculator(surface)


    # plot_model(True)
    if config.SHOW_PLOTS == True:
        plot_model(path, model)
        plot_mean_curvature(path, surface, point_cloud_3d, point_cloud_2d, model)
    plot_and_save_mean_curvature_2d(path, surface, save_plot=save_plot)

    return scaling_factor, min_val_x, min_val_y

def calculate_center_angle(surface: MinSurface, num_points: int, path: matplotlib.patches.Path):
    """
    Function for calculating the angle at the center of the leaf, relative to the floor.
    :return:
    """
    # Sample a set of points on which we are going to calculate the normals, and get from the surface the points
    # of the original 2d point cloud on which we based the surface.
    points = sample_low_disc_seq(num_points, path)
    point_cloud_2d = surface.point_cloud_2d


    min_y = np.min(point_cloud_2d[:, 1])

    print(f"point_cloud_2d = {point_cloud_2d}")
    print(f"min_y = {min_y}")

    # # Get the values that the model spits for the given points.
    # points = torch.tensor(points, dtype=torch.float32, requires_grad=True)
    # vals = surface.model(points)
    #
    # # The values are split into their X, Y, and Z components.
    # x = vals[:, 0]
    # y = vals[:, 1]
    # z = vals[:, 2]
    #
    # # The gradient and Hessian matrices are calculated for each of the X, Y, and Z components using the grad_and_hess function.
    # # This involves finding the first and second-order derivatives of each component with respect to the input points.
    # g_x, h_x = surface.grad_and_hess(x, points)
    # g_y, h_y = surface.grad_and_hess(y, points)
    # g_z, h_z = surface.grad_and_hess(z, points)
    #
    # # Compute the normal vectors by expressing the cross product in components.
    # # first the normal, n = r_X x r_Y / norm(...)
    # # so, nx = r_Xy r_Yz - r_Xz r_Yy = g_y[0]*g_z[1] - g_z[0]*g_y[1]
    # nx = g_y[:, 0] * g_z[:, 1] - g_z[:, 0] * g_y[:, 1]
    # # ny = r_Xz r_Yx - r_Xx r_Yz = g_z[0]*g_x[1] - g_x[0]*g_z[1]
    # ny = g_z[:, 0] * g_x[:, 1] - g_x[:, 0] * g_z[:, 1]
    # # nz = r_Xx r_Yy - r_Xy r_Yx = g_x[0]*g_y[1] - g_y[0]*g_x[1]
    # nz = g_x[:, 0] * g_y[:, 1] - g_y[:, 0] * g_x[:, 1]

    return


if __name__ == "__main__":

    base_path = config.BASE_PATH
    number_of_frames = 531

    leaf_svg_path = config.LEAF_SVG_PATH
    path = parse_path(leaf_svg_path)
    sample_square_lattice((100, 100), path)
    offset = 203

    # if config.SHOW_PLOTS == True:
    #     for i in range(number_of_frames):
    #         j = i + offset
    #         print(f'session {j}')
    #         file_path_2d = os.path.join(base_path, "Session_" + str(j),"pcd_2d_blue.csv")
    #         file_path_3d = os.path.join(base_path, "Session_" + str(j),"pcd_3d_blue.csv")
    #         # -------------
    #         # reconstruction = base_path + "Session_" + str(j) + "/fused.ply"
    #         # pcd = o3d.io.read_point_cloud(reconstruction)  # Read the point cloud
    #         # o3d.visualization.draw_geometries([pcd])
    #         # -------------
    #         point_cloud_2d1 = np.loadtxt(file_path_2d, delimiter=',', skiprows=1)
    #         point_cloud_3d1 = np.loadtxt(file_path_3d, delimiter=',', skiprows=1)
    #         point_cloud_2d2 = np.hstack((point_cloud_2d1, np.zeros((point_cloud_2d1.shape[0], 1))))
    #
    #         rgb1 = np.stack((np.ones(point_cloud_3d1.shape[0]), np.ones(point_cloud_3d1.shape[0]), np.ones(point_cloud_3d1.shape[0])), axis=1)
    #         rgb2 = np.stack((np.ones(point_cloud_2d2.shape[0]), np.ones(point_cloud_2d2.shape[0]), np.ones(point_cloud_2d2.shape[0])), axis=1)
    #         mesh1 = pv.PolyData(point_cloud_3d1)
    #         mesh2 = pv.PolyData(point_cloud_2d2)
    #         mesh1['color'] = rgb1
    #         mesh2['color'] = rgb2
    #
    #         # Set up a PyVista plotter and configure the settings.
    #         pv.set_plot_theme('dark')
    #         plotter = pv.Plotter()
    #         plotter.enable_terrain_style(mouse_wheel_zooms=True)
    #         plotter.enable_anti_aliasing()
    #
    #         # --------------------
    #         # for plotting the actual point clouds
    #         plotter.add_mesh(mesh1, point_size=20,
    #                          render_points_as_spheres=True,
    #                          scalars='color', rgb=True,
    #                          )
    #         plotter.add_mesh(mesh2, point_size=20,
    #                          render_points_as_spheres=True,
    #                          scalars='color', rgb=True,
    #                          )
    #         # --------------------
    #
    #         # enable eye_dome_lighting
    #         plotter.enable_eye_dome_lighting()
    #
    #         # enable axes, with a large font size
    #         plotter.show_grid()
    #         plotter.show_bounds(all_edges=True, font_size=16, color='white', location='outer')
    #
    #         plotter.show(title=str(j), auto_close=False)
    #
    #
    # scaling_factor = None
    # min_val_x = None
    # min_val_y = None
    # for i in range(number_of_frames):
    #     j = i + offset
    #
    #     print(f'session number {j}')
    #     # =========> make sure the format of the path is the correct one <============
    #     file_path_2d_current = os.path.join(base_path, "Session_" + str(j), "pcd_2d_red.csv")
    #     file_path_3d_current = os.path.join(base_path, "Session_" + str(j), "pcd_3d_red.csv")
    #     model_save_path = os.path.join(base_path, "Session_" + str(j), "fitting_model_red.pt")
    #     plot_save_path = os.path.join(base_path, "Session_" + str(j), "mean_curvature_red.png")
    #     scaling_factor, min_val_x, min_val_y = fit_and_plot_curvature(file_path_2d_current, file_path_3d_current,
    #                                                                   save_plot=plot_save_path, save_model=model_save_path, scaling_factor=scaling_factor, min_val_x=min_val_x, min_val_y=min_val_y)


    scaling_factor = None
    min_val_x = None
    min_val_y = None
    for i in range(number_of_frames):
        j = i + offset
        model_save_path = os.path.join(base_path, "Session_" + str(j), "fitting_model_red.pt")
        file_path_2d = os.path.join(base_path, "Session_" + str(j), "pcd_2d_red.csv")
        file_path_3d = os.path.join(base_path, "Session_" + str(j), "pcd_3d_red.csv")
        model = load_model(model_save_path)
        print(f"frame number = {j}")
        scaling_factor, min_val_x, min_val_y = plot_model_plus_target(path, model, file_path_2d, file_path_3d,
                                                                      scaling_factor=scaling_factor, min_val_x=min_val_x,
                                                                      min_val_y=min_val_y)



    #
    # load_models_and_plot_average_curvature(base_path, number_of_frames, path, offset)

# ==================
# torch.save(model.state_dict(), "Model with forces.pt")
#
# # to load:
# if False:
#     from force_finder import ForceFinderModel
#     model = ForceFinderModel(1, (50, 50))
#     model.load_state_dict(torch.load("Model with forces.pt"))