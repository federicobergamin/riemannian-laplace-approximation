import numpy as np
import sys

import torch
from scipy.integrate import solve_ivp
import scipy.integrate as integrate
import matplotlib.pyplot as plt
# import sklearn.neighbors.graph as knn_graph
from scipy.interpolate import CubicSpline
# from csaps import csaps


# Draws an elipsoid that correspond to the metric
def plot_metric(x, cov, color='r', inverse_metric=False, linewidth=1):
    eigvals, eigvecs = np.linalg.eig(cov)
    N = 100
    theta = np.linspace(0, 2 * np.pi, N)
    theta = theta.reshape(N, 1)
    points = np.concatenate((np.cos(theta), np.sin(theta)), axis=1)
    points = points * np.sqrt(eigvals)
    points = np.matmul(eigvecs, points.transpose()).transpose()
    points = points + x.flatten()
    plt.plot(points[:, 0], points[:, 1], c=color, linewidth=linewidth, label='Metric')


# This function evaluates the differential equation c'' = f(c, c')
def geodesic_system(manifold, c, dc):
    # Input: c, dc ( D x N )

    D, N = c.shape
    if (dc.shape[0] != D) | (dc.shape[1] != N):
        print('geodesic_system: second and third input arguments must have same dimensionality\n')
        sys.exit(1)

    # Evaluate the metric and the derivative
    M, dM = manifold.metric_tensor(c, nargout=2)

    # Prepare the output (D x N)
    ddc = np.zeros((D, N))

    # Diagonal Metric Case, M (N x D), dMdc_d (N x D x d=1,...,D) d-th column derivative with respect to c_d
    if manifold.is_diagonal():
        for n in range(N):
            dMn = np.squeeze(dM[n, :, :])
            ddc[:, n] = -0.5 * (2 * np.matmul(dMn * dc[:, n].reshape(-1, 1), dc[:, n])
                                - np.matmul(dMn.T, (dc[:, n] ** 2))) / M[n, :]

    # Non-Diagonal Metric Case, M ( N x D x D ), dMdc_d (N x D x D x d=1,...,D)
    else:
        M_inv = np.linalg.inv(M)  # N x D x D
        Term1 = dM.reshape(N, D, D * D, order='F')  # N x D x D^2
        Term2 = dM.reshape(N, D * D, D, order='F')  # N x D^2 x D

        for n in range(N):
            # Mn = np.squeeze(M[n, :, :])
            # if np.linalg.cond(Mn) < 1e-15:
            #     print('Ill-condition metric!\n')
            #     sys.exit(1)

            # dvecMdcn = dM[n, :, :, :].reshape(D * D, D, order='F')
            # blck = np.kron(np.eye(D), dc[:, n])

            ddc[:, n] = -0.5 * M_inv[n, :, :] @ ((2 * Term1[n, :, :] - Term2[n, :, :].T) @ np.kron(dc[:, n], dc[:, n]))
    return ddc


# This function changes the 2nd order ODE to two 1st order ODEs takes c, dc and returns dc, ddc.
def second2first_order(manifold, state, subset_of_weights):
    # Input: state [c; dc] (2D x N), y=[dc; ddc]: (2D x N)
    D = int(state.shape[0] / 2)

    # TODO: Something better for this?
    if state.ndim == 1:
        state = state.reshape(-1, 1)  # (2D,) -> (2D, 1)

    c = state[:D, :]  # D x N
    cm = state[D:, :]  # D x N
    if subset_of_weights == 'last_layer':
        # in the last layer case we can use the old implementation
        cmm = geodesic_system(manifold, c, cm)  # D x N
    else:
        # if we want full network we use the hvp implementation
        cmm = manifold.geodesic_system(c, cm)
    y = np.concatenate((cm, cmm), axis=0)
    return y


# If the solver failed provide the linear distance as the solution
def evaluate_failed_solution(p0, p1, t):
    # Input: p0, p1 (D x 1), t (T x 0)
    c = (1 - t) * p0 + t * p1  # D x T
    dc = np.repeat(p1 - p0, np.size(t), 1)  # D x T
    return c, dc


# If the solver_bvp() succeeded provide the solution.
def evaluate_solution(solution, t, t_scale):
    # Input: t (Tx0), t_scale is used from the Expmap to scale the curve in order to have correct length,
    #        solution is an object that solver_bvp() returns
    c_dc = solution.sol(t * t_scale)
    D = int(c_dc.shape[0] / 2)

    # TODO: Why the t_scale is used ONLY for the derivative component?
    if np.size(t) == 1:
        c = c_dc[:D].reshape(D, 1)
        dc = c_dc[D:].reshape(D, 1) * t_scale
    else:
        c = c_dc[:D, :]  # D x T
        dc = c_dc[D:, :] * t_scale  # D x T
    return c, dc


def evaluate_spline_solution(curve, dcurve, t):
    # Input: t (Tx0), t_scale is used from the Expmap to scale the curve in order to have correct length,
    #        solution is an object that solver_bvp() returns
    c = curve(t)
    dc = dcurve(t)
    D = int(c.shape[0])

    # TODO: Why the t_scale is used ONLY for the derivative component?
    if np.size(t) == 1:
        c = c.reshape(D, 1)
        dc = dc.reshape(D, 1)
    else:
        c = c.T  # Because the c([0,..,1]) -> N x D
        dc = dc.T
    return c, dc


# This function computes the infinitesimal small length on a curve
def local_length(manifold, curve, t):
    # Input: curve function of t returns (D X T), t (T x 0)
    c, dc = curve(t)  # [D x T, D x T]
    D = c.shape[0]
    M = manifold.metric_tensor(c, nargout=1)
    if manifold.is_diagonal():
        dist = np.sqrt(np.sum(M.transpose() * (dc ** 2), axis=0))  # T x 1, c'(t) M(c(t)) c'(t)
    else:
        dc = dc.T  # D x N -> N x D
        dc_rep = np.repeat(dc[:, :, np.newaxis], D, axis=2)  # N x D -> N x D x D
        Mdc = np.sum(M * dc_rep, axis=1)  # N x D
        dist = np.sqrt(np.sum(Mdc * dc, axis=1))  # N x 1
    return dist


# This function computes the length of the geodesic curve
# The smaller the approximation error (tol) the slower the computation.
def curve_length(manifold, curve, a=0, b=1, tol=1e-5, limit=50):
    # Input: curve a function of t returns (D x ?), [a,b] integration interval, tol error of the integration
    if callable(curve):
        # function returns: curve_length_eval = (integral_value, some_error)
        curve_length_eval = integrate.quad(lambda t: local_length(manifold, curve, t), a, b, epsabs=tol, limit=limit)  # , number of subintervals
    else:
        print("TODO: Not implemented yet integration for discrete curve!\n")
        sys.exit(1)

    return curve_length_eval[0]


# This function plots a curve that is given as a parametric function, curve: t -> (D x len(t)).
def plot_curve(curve, **kwargs):
    N = 1000
    T = np.linspace(0, 1, N)
    curve_eval = curve(T)[0]

    D = curve_eval.shape[0]  # Dimensionality of the curve

    if D == 2:
        plt.plot(curve_eval[0, :], curve_eval[1, :], **kwargs)
    elif D == 3:
        plt.plot(curve_eval[0, :], curve_eval[1, :], curve_eval[2, :], **kwargs)

# This function vectorizes an matrix by stacking the columns
def vec(x):
    # Input: x (NxD) -> (ND x 1)
    return x.flatten('F').reshape(-1, 1)



# This function implements the exponential map
def expmap(manifold, x, v, subset_of_weights='all'):
    assert subset_of_weights == 'all' or subset_of_weights == 'last_layer', 'subset_of_weights must be all or last_layer'

    # Input: v,x (Dx1)
    x = x.reshape(-1, 1)
    v = v.reshape(-1, 1)
    D = x.shape[0]

    ode_fun = lambda t, c_dc: second2first_order(manifold, c_dc, subset_of_weights).flatten()  # The solver needs this shape (D,)
    if np.linalg.norm(v) > 1e-5:
        # print('I think we should enter here')
        curve, failed = new_solve_expmap(manifold, x, v, ode_fun, subset_of_weights)
    else:
        curve = lambda t: (x.reshape(D, 1).repeat(np.size(t), axis=1),
                           v.reshape(D, 1).repeat(np.size(t), axis=1))  # Return tuple (2D x T)
        failed = True

    return curve, failed


# This function solves the initial value problem for the implementation of the expmap
def new_solve_expmap(manifold, x, v, ode_fun, subset_of_weights):
    D = x.shape[0]

    if isinstance(v, torch.Tensor):
        v = v.cpu().numpy()
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
        
    init = np.concatenate((x, v), axis=0).flatten()  # 2D x 1 -> (2D, ), the solver needs this shape

    failed = False

    prev_t = 0
    t = 1

    solution = solve_ivp(ode_fun, [prev_t, t], init, dense_output=True, atol = 1e-3, rtol= 1e-6)  # First solution of the IVP problem
    curve = lambda tt: evaluate_solution(solution, tt, 1)  # with length(c(t)) != ||v||_c
    
    return curve, failed

