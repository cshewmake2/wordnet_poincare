import torch


def d(u,v,eps = 1e-5):
    """
    Compute the distance between two points in the Poincare Ball.
    
    Parameters
    ----------
    u : torch.Tensor, shape = (1,n_features)
    v : torch.Tensor, shape = (n_samples,n_features)
    
    Returns
    -------
    dist : torch.Tensor, shape = (n_samples,)
    """
    if (len(u.shape) != 2) or (u.shape[0] != 1):
        raise ValueError('Distance function only supports comparison of one u to any number of vs. Got {}'.format(u.shape))
    norm_u = torch.linalg.norm(u,axis = 1)
    norm_v = torch.linalg.norm(v,axis = 1)
    if (norm_u >= 1.0) or (norm_v >= 1.0).any():
        raise ValueError('Norm of u or v cannot be greater than 1.\nGot u:{}, v:{}'.format(norm_u,norm_v))
    numerator = torch.linalg.norm(u - v,axis = 1)**2
    denominator = (1 - norm_u**2) * (1 - norm_v**2)
    acosh_arg = torch.clamp(1 + 2*(numerator / (denominator+eps)),1.0 + eps,float('inf'))
    return torch.arccosh(acosh_arg)