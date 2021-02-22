import torch
import geometry as geom


def L(x, D, Nu):
    """Compute the loss associated with the current embedding of the dataset.
    
    Parameters
    ----------
    x : np.array, shape={num_points, dimension}
        Embedding coordinates for each point in the data. 
        
    D : list of lists, shape={num_us, 1, num_v_per_u}
        Container relating a point (word) u to its connected points (words) vs.
    """
    loss = 0
    n_features = x.shape[-1]
    for (u_ind, v_inds) in D:
        u = x[u_ind,:].reshape(1,n_features)
        vs = x[v_inds,:].reshape(len(v_inds),n_features)
        # Numerator: Positive Pairs
        numerators = torch.log(torch.exp(-geom.d(u,vs)))

        # Denominator: Negative Pairs
        vp_inds_lists = Nu[u_ind]
        vp_x = x[vp_inds_lists,:]
        denominators = torch.stack([torch.logsumexp(-geom.d(u,vi),dim = 0) for vi in vp_x])
#         denominators = torch.stack([torch.sum(exp_d(u,vi)) for vi in vp_x])
        # Loss for this u: set of (u,[v1,v2,...,vn])
#         print(torch.isinf(denominators).any())
        L_u = torch.sum(numerators / denominators)
        
        loss += L_u
    return -loss 

# def exp_d(u,v):
#     return torch.exp(-geom.d(u,v))




#def L(x, D, Nu):
#     """Compute the loss associated with the current embedding of the dataset.
    
#     Parameters
#     ----------
#     x : np.array, shape={dimension, num_points}
#         Embedding coordinates for each point in the data. 
        
#     D : list of lists, shape={num_points, num_connections}
#         Container relating a point (word) u to its connected points (words) vs.
#     """
#     loss = 0
#     for (u_ind, v_inds) in D:
#         # Numerator: Positive Pairs
#         u = x[:, u_ind].reshape(-1,1)
#         vs = x[:, v_inds].reshape(-1,len(v_inds))
#         numerators = np.exp(-d(u,vs))

#         # Denominator: Negative Pairs
#         vp_inds_lists = Nu[u_ind]
#         compute_denominator = lambda v_inds: np.sum(np.exp(-d(u, x[:,v_inds])))
#         denominators = np.array([compute_denominator(vp_inds) for vp_inds in vp_inds_lists]).reshape(1,-1)
#         loss += np.sum(np.log(numerators / denominators))
#     return loss  


# def dp(u,v):
#     """Compute the gradient of d wrt u."""
#     a = 1 - np.linalg.norm(u, axis = 0)**2
#     b = 1 - np.linalg.norm(v, axis = 0)**2
#     gm = 1 + 2*np.linalg.norm(u-v, axis = 0)**2 / (a*b)
#     du = 4*((u*(np.linalg.norm(v, axis = 0)**2 - 2*np.dot(u.T,v) + 1) / a**2) - (v/a)) / (b*np.sqrt(gm**2 - 1))
#     return du


# def grad_u(x, D, Nu):
#     """Compute the gradient for all u points."""
#     ### TODO
#     return 0.1*(1 - 2*np.random.rand(*x.shape))
# #     x_u_grad = np.zeros_like(x)
# #     dim = x.shape[0]

# #     for (u_ind, v_inds) in D:
# #         u = x[:, u_ind].reshape(-1,1)
# #         vs = x[:, v_inds].reshape(-1,len(v_inds))

# #         dpsum = np.sum(-dp(u,vs), axis = 1, keepdims = True)
# #         vp_inds_lists = Nu[u_ind]
# #         compute_num = lambda u, vp: np.sum( dp(u, vp.reshape(dim, -1)) * np.exp(-d(u, vp.reshape(dim, -1))), axis = 1,  keepdims = True)
# #         compute_denom = lambda u, vp: np.sum( np.exp(-d(u, vp.reshape(dim,-1))) )
 
# #         neg_terms = [compute_num(u, x[:,vps]) / compute_denom(u, x[:,vps]) for vps in vp_inds_lists]
# #         dp_Nu = np.sum(np.concatenate(neg_terms, axis = 1), axis = 1, keepdims = True)
# #         x_u_grad[:, u_ind] = dp_Nu.T
# #     return x_u_grad
        
    
# def grad_v(x, D, Nu):
#     """Compute the gradient for all v coordinates."""
#     ### TODO
#     return np.zeros_like(x)


# def grad_vp(x, D, Nu):
#     """Compute the gradient for all vp (vprime) coordinates."""
#     ### TODO
#     return np.zeros_like(x)


# def grad_E(x, D, Nu):
#     """Compute the Euclidean gradient using the three coordinate group gradients."""
#     ### TODO
#     # Update u
#     x_grad_u = grad_u(x, D, Nu)
    
#     # Update v
#     x_grad_v = grad_v(x, D, Nu)

#     # Update v'
#     x_grad_vp = grad_vp(x, D, Nu)
    
#     grad = x_grad_u + x_grad_v + x_grad_vp
#     return grad


# def proj(x, epsilon = 1E-4):
#     """Rescale the point to ensure that it lies within the Poincare Ball."""
#     norm_x = np.linalg.norm(x, axis = 0, keepdims = True)
#     return np.where(norm_x >= 1, x/norm_x - epsilon, x)


# def rgd(x0, D, num_iter, nu = 10, c = 1):
#     """Perform Riemannian Stochastic Gradient Descent."""
#     losses = []
#     xts = []
#     xt = x0
#     nt = nu/c
#     xts.append(x0)
#     for i in range(0,num_iter):
#         Nu = generate_Nu(D)
#         g_pb_inv = ((1 - np.linalg.norm(xt, axis = 0)**2)**2)/4
#         xt = proj(xt - nt*g_pb_inv*grad_E(xt, D, Nu))
#         loss = L(xt, D, Nu)
#         losses.append(loss)
#         print('Iteration: ' + str(i))
#         print('Loss: ' + str(loss))
#         xts.append(xt)
#     return xts, losses