from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch.optim as optim
from .lbfgs_ls import LBFGS as LBFGSLs


def create_optimizer(parameters, optim_type='lbfgs',
                     lr=1e-3,
                     momentum=0.9,
                     use_nesterov=True,
                     beta1=0.9,
                     beta2=0.999,
                     epsilon=1e-8,
                     use_locking=False,
                     weight_decay=0.0,
                     centered=False,
                     rmsprop_alpha=0.99,
                     maxiters=20,
                     gtol=1e-6,
                     ftol=1e-9,
                     **kwargs):
    ''' Creates the optimizer
    '''
    if optim_type == 'adam':
        return (optim.Adam(parameters, lr=lr, betas=(beta1, beta2),
                           weight_decay=weight_decay),
                False)
    elif optim_type == 'lbfgs':
        return (optim.LBFGS(parameters, lr=lr, max_iter=maxiters), False)
    elif optim_type == 'lbfgsls':
        return LBFGSLs(parameters, lr=lr, max_iter=maxiters,
                       line_search_fn='strong_Wolfe'), False
    elif optim_type == 'rmsprop':
        return (optim.RMSprop(parameters, lr=lr, epsilon=epsilon,
                              alpha=rmsprop_alpha,
                              weight_decay=weight_decay,
                              momentum=momentum, centered=centered),
                False)
    elif optim_type == 'sgd':
        return (optim.SGD(parameters, lr=lr, momentum=momentum,
                          weight_decay=weight_decay,
                          nesterov=use_nesterov),
                False)
    else:
        raise ValueError('Optimizer {} not supported!'.format(optim_type))
