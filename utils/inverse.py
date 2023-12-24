from torch.distributions.multivariate_normal import _precision_to_scale_tril


def get_inverse(Hessian):
    """
    I am trying to copy the way in which Laplace is computing the posterior
    covariance. Because they are able to get symmetric inverse, while I am
    failing in it.
    """
    posterior_scale = _precision_to_scale_tril(Hessian)
    posterior_cov = posterior_scale @ posterior_scale.T

    return posterior_cov
