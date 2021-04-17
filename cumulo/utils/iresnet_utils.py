"""
All functions in here were just copied from https://github.com/FrontierDevelopmentLab/CUMULO.
The CUMULO paper authors adapted the functions from https://openreview.net/pdf?id=HJsjkMb0Z
and https://arxiv.org/abs/1804.04368.
"""


from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torch.nn as nn
import torch
from torch.nn.functional import normalize, conv_transpose2d, conv2d
from torch.nn.parameter import Parameter


def bits_per_dim(logpx, inputs):
    return -logpx / float(np.log(2.) * np.prod(inputs.shape[1:])) + 8.


def split(x):
    n = int(x.size(1) / 2)
    x1 = x[:, :n, :, :].contiguous()
    x2 = x[:, n:, :, :].contiguous()
    return x1, x2


class InjectivePad(nn.Module):
    def __init__(self, pad_size):
        super(InjectivePad, self).__init__()
        self.pad_size = pad_size
        self.pad = nn.ZeroPad2d((0, 0, 0, pad_size))

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        x = self.pad(x)
        return x.permute(0, 2, 1, 3)

    def inverse(self, x):
        return x[:, :x.size(1) - self.pad_size, :, :]


class Split(nn.Module):
    def __init__(self):
        super(Split, self).__init__()

    def forward(self, x):
        n = int(x.size(1) / 2)
        x1 = x[:, :n, :, :].contiguous()
        x2 = x[:, n:, :, :].contiguous()
        return x1, x2

    def inverse(self, x1, x2):
        return torch.cat((x1, x2), 1)


class squeeze(nn.Module):
    def __init__(self, block_size):
        super(squeeze, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def inverse(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / self.block_size_sq)
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)
        t_1 = output.contiguous().view(batch_size, d_height, d_width, self.block_size_sq, s_depth)
        spl = t_1.split(self.block_size, 3)
        stack = [t_t.contiguous().view(batch_size, d_height, s_width, s_depth) for t_t in spl]
        output = torch.stack(stack, 0).transpose(0, 1).permute(0, 2, 1, 3, 4).contiguous().view(batch_size, s_height,
                                                                                                s_width, s_depth)
        output = output.permute(0, 3, 1, 2)
        return output.contiguous()

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.contiguous().view(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        return output.contiguous()


class ActNorm2D(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super(ActNorm2D, self).__init__()
        self.eps = eps
        self.num_channels = num_channels
        self._log_scale = Parameter(torch.Tensor(num_channels))
        self._shift = Parameter(torch.Tensor(num_channels))
        self._init = False

    def log_scale(self):
        return self._log_scale[None, :, None, None]

    def shift(self):
        return self._shift[None, :, None, None]

    def forward(self, x):
        if not self._init:
            with torch.no_grad():
                # initialize params to input stats
                assert self.num_channels == x.size(1)
                mean = torch.transpose(x, 0, 1).contiguous().view(self.num_channels, -1).mean(dim=1)
                zero_mean = x - mean[None, :, None, None]
                var = torch.transpose(zero_mean ** 2, 0, 1).contiguous().view(self.num_channels, -1).mean(dim=1)
                std = (var + self.eps) ** .5
                log_scale = torch.log(1. / std)
                self._shift.data = - mean * torch.exp(log_scale)
                self._log_scale.data = log_scale
                self._init = True
        log_scale = self.log_scale()
        logdet = log_scale.sum() * x.size(2) * x.size(3)
        return x * torch.exp(log_scale) + self.shift(), logdet

    def inverse(self, x):
        return (x - self.shift()) * torch.exp(-self.log_scale())


class MaxMinGroup(nn.Module):
    def __init__(self, group_size, axis=-1):
        super(MaxMinGroup, self).__init__()
        self.group_size = group_size
        self.axis = axis

    def forward(self, x):
        maxes = maxout_by_group(x, self.group_size, self.axis)
        mins = minout_by_group(x, self.group_size, self.axis)
        maxmin = torch.cat((maxes, mins), dim=1)
        return maxmin

    def extra_repr(self):
        return 'group_size: {}'.format(self.group_size)


def process_maxmin_groupsize(x, group_size, axis=-1):
    size = list(x.size())
    num_channels = size[axis]

    if num_channels % group_size:
        raise ValueError('number of features({}) is not a '
                         'multiple of group_size({})'.format(num_channels, num_channels))
    size[axis] = -1
    if axis == -1:
        size += [group_size]
    else:
        size.insert(axis + 1, group_size)
    return size


def maxout_by_group(x, group_size, axis=-1):
    size = process_maxmin_groupsize(x, group_size, axis)
    sort_dim = axis if axis == -1 else axis + 1
    return torch.max(x.view(*size), sort_dim)[0]


def minout_by_group(x, group_size, axis=-1):
    size = process_maxmin_groupsize(x, group_size, axis)
    sort_dim = axis if axis == -1 else axis + 1
    return torch.min(x.view(*size), sort_dim)[0]


def batch_class_weights(labels, nb_classes):
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes, labels)

    class_weights = np.zeros(nb_classes)

    for c, w in zip(classes, weights):
        class_weights[c] = w

    return class_weights


def power_series_matrix_logarithm_trace(Fx, x, k, n):
    """
    Fast-boi Tr(Ln(d(Fx)/dx)) using power-series approximation
    biased but fast
    :param Fx: output of f(x)
    :param x: input
    :param k: number of power-series terms  to use
    :param n: number of Hitchinson's estimator samples
    :return: Tr(Ln(I + df/dx))
    """
    # trace estimation including power series
    outSum = Fx.sum(dim=0)
    dim = list(outSum.shape)
    dim.insert(0, n)
    dim.insert(0, x.size(0))
    u = torch.randn(dim).to(x.device)
    trLn = 0
    for j in range(1, k + 1):
        if j == 1:
            vectors = u
        # compute vector-jacobian product
        vectors = [torch.autograd.grad(Fx, x, grad_outputs=vectors[:, i],
                                       retain_graph=True, create_graph=True)[0] for i in range(n)]
        # compute summand
        vectors = torch.stack(vectors, dim=1)
        vjp4D = vectors.view(x.size(0), n, 1, -1)
        u4D = u.view(x.size(0), n, -1, 1)
        summand = torch.matmul(vjp4D, u4D)
        # add summand to power series
        if (j + 1) % 2 == 0:
            trLn += summand / np.float(j)
        else:
            trLn -= summand / np.float(j)
    trace = trLn.mean(dim=1).squeeze()
    return trace


class SpectralNormConv(object):
    _version = 1

    def __init__(self, coeff, input_dim, name='weight', n_power_iterations=1, eps=1e-12):
        self.coeff = coeff
        self.input_dim = input_dim
        self.name = name
        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                             'got n_power_iterations={}'.format(n_power_iterations))
        self.n_power_iterations = n_power_iterations
        self.eps = eps

    def compute_weight(self, module, do_power_iteration):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        v = getattr(module, self.name + '_v')
        sigma_log = getattr(module, self.name + '_sigma')  # for logging

        # get settings from conv-module (for transposed convolution)
        stride = module.stride
        padding = module.padding

        if do_power_iteration:
            with torch.no_grad():
                for _ in range(self.n_power_iterations):
                    v_s = conv_transpose2d(u.view(self.out_shape), weight, stride=stride,
                                           padding=padding, output_padding=0)
                    # Note: out flag for in-place changes
                    v = normalize(v_s.view(-1), dim=0, eps=self.eps, out=v)

                    u_s = conv2d(v.view(self.input_dim), weight, stride=stride, padding=padding,
                                 bias=None)
                    u = normalize(u_s.view(-1), dim=0, eps=self.eps, out=u)
                if self.n_power_iterations > 0:
                    # See above on why we need to clone
                    u = u.clone()
                    v = v.clone()
        weight_v = conv2d(v.view(self.input_dim), weight, stride=stride, padding=padding,
                          bias=None)
        weight_v = weight_v.view(-1)
        sigma = torch.dot(u.view(-1), weight_v)
        # enforce spectral norm only as constraint
        factorReverse = torch.max(torch.ones(1).to(weight.device),
                                  sigma / self.coeff)
        # for logging
        weight_v_det = weight_v.detach()
        u_det = u.detach()
        torch.max(torch.dot(u_det.view(-1), weight_v_det),
                  torch.dot(u_det.view(-1), weight_v_det), out=sigma_log)

        # rescaling
        weight = weight / (factorReverse + 1e-5)  # for stability
        return weight

    def remove(self, module):
        with torch.no_grad():
            weight = self.compute_weight(module, do_power_iteration=False)
        delattr(module, self.name)
        delattr(module, self.name + '_u')
        delattr(module, self.name + '_orig')
        module.register_parameter(self.name, torch.nn.Parameter(weight.detach()))

    def __call__(self, module, inputs):
        setattr(module, self.name, self.compute_weight(module, do_power_iteration=module.training))

    @staticmethod
    def apply(module, coeff, input_dim, name, n_power_iterations, eps):
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, SpectralNormConv) and hook.name == name:
                raise RuntimeError("Cannot register two spectral_norm hooks on "
                                   "the same parameter {}".format(name))

        fn = SpectralNormConv(coeff, input_dim, name, n_power_iterations, eps)
        weight = module._parameters[name]

        with torch.no_grad():
            num_input_dim = input_dim[0] * input_dim[1] * input_dim[2] * input_dim[3]
            v = normalize(torch.randn(num_input_dim), dim=0, eps=fn.eps)

            # get settings from conv-module (for transposed convolution)
            stride = module.stride
            padding = module.padding
            # forward call to infer the shape
            u = conv2d(v.view(input_dim), weight, stride=stride, padding=padding,
                       bias=None)
            fn.out_shape = u.shape
            num_output_dim = fn.out_shape[0] * fn.out_shape[1] * fn.out_shape[2] * fn.out_shape[3]
            # overwrite u with random init
            u = normalize(torch.randn(num_output_dim), dim=0, eps=fn.eps)

        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        setattr(module, fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)
        module.register_buffer(fn.name + "_v", v)
        module.register_buffer(fn.name + "_sigma", torch.ones(1).to(weight.device))

        module.register_forward_pre_hook(fn)

        module._register_state_dict_hook(SpectralNormConvStateDictHook(fn))
        module._register_load_state_dict_pre_hook(SpectralNormConvLoadStateDictPreHook(fn))
        return fn


class SpectralNormConvLoadStateDictPreHook(object):
    # See docstring of SpectralNorm._version on the changes to spectral_norm.
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, state_dict, prefix, local_metadata, strict,
                 missing_keys, unexpected_keys, error_msgs):
        fn = self.fn
        version = local_metadata.get('spectral_norm_conv', {}).get(fn.name + '.version', None)
        if version is None or version < 1:
            with torch.no_grad():
                weight_orig = state_dict[prefix + fn.name + '_orig']
                weight = state_dict.pop(prefix + fn.name)
                sigma = (weight_orig / weight).mean()
                weight_mat = fn.reshape_weight_to_matrix(weight_orig)
                u = state_dict[prefix + fn.name + '_u']


class SpectralNormConvStateDictHook(object):
    # See docstring of SpectralNorm._version on the changes to spectral_norm.
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, module, state_dict, prefix, local_metadata):
        if 'spectral_norm_conv' not in local_metadata:
            local_metadata['spectral_norm_conv'] = {}
        key = self.fn.name + '.version'
        if key in local_metadata['spectral_norm_conv']:
            raise RuntimeError("Unexpected key in metadata['spectral_norm_conv']: {}".format(key))
        local_metadata['spectral_norm_conv'][key] = self.fn._version


def spectral_norm_conv(module, coeff, input_dim, name='weight', n_power_iterations=1, eps=1e-12):
    input_dim_4d = (1, input_dim[0], input_dim[1], input_dim[2])
    SpectralNormConv.apply(module, coeff, input_dim_4d, name, n_power_iterations, eps)
    return module


class SpectralNorm(object):
    _version = 1

    def __init__(self, coeff, name='weight', n_power_iterations=1, dim=0, eps=1e-12):
        self.coeff = coeff
        self.name = name
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                             'got n_power_iterations={}'.format(n_power_iterations))
        self.n_power_iterations = n_power_iterations
        self.eps = eps

    def reshape_weight_to_matrix(self, weight):
        weight_mat = weight
        if self.dim != 0:
            # permute dim to front
            weight_mat = weight_mat.permute(self.dim,
                                            *[d for d in range(weight_mat.dim()) if d != self.dim])
        height = weight_mat.size(0)
        return weight_mat.reshape(height, -1)

    def compute_weight(self, module, do_power_iteration):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        v = getattr(module, self.name + '_v')
        sigma_log = getattr(module, self.name + '_sigma')  # for logging
        weight_mat = self.reshape_weight_to_matrix(weight)

        if do_power_iteration:
            with torch.no_grad():
                for _ in range(self.n_power_iterations):
                    v = normalize(torch.mv(weight_mat.t(), u), dim=0, eps=self.eps, out=v)
                    u = normalize(torch.mv(weight_mat, v), dim=0, eps=self.eps, out=u)
                if self.n_power_iterations > 0:
                    # See above on why we need to clone
                    u = u.clone()
                    v = v.clone()

        sigma = torch.dot(u, torch.mv(weight_mat, v))
        # soft normalization: only when sigma larger than coeff
        factor = torch.max(torch.ones(1).to(weight.device), sigma / self.coeff)
        weight = weight / factor
        # for logging
        sigma_det = sigma.detach()
        torch.max(torch.ones(1).to(weight.device), sigma_det / self.coeff,
                  out=sigma_log)
        return weight

    def remove(self, module):
        with torch.no_grad():
            weight = self.compute_weight(module, do_power_iteration=False)
        delattr(module, self.name)
        delattr(module, self.name + '_u')
        delattr(module, self.name + '_v')
        delattr(module, self.name + '_orig')
        module.register_parameter(self.name, torch.nn.Parameter(weight.detach()))

    def __call__(self, module, inputs):
        setattr(module, self.name, self.compute_weight(module, do_power_iteration=module.training))

    def _solve_v_and_rescale(self, weight_mat, u, target_sigma):
        v = torch.chain_matmul(weight_mat.t().mm(weight_mat).pinverse(), weight_mat.t(), u.unsqueeze(1)).squeeze(1)
        return v.mul_(target_sigma / torch.dot(u, torch.mv(weight_mat, v)))

    @staticmethod
    def apply(module, name, coeff, n_power_iterations, dim, eps):
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, SpectralNorm) and hook.name == name:
                raise RuntimeError("Cannot register two spectral_norm hooks on "
                                   "the same parameter {}".format(name))

        fn = SpectralNorm(coeff, name, n_power_iterations, dim, eps)
        weight = module._parameters[name]

        with torch.no_grad():
            weight_mat = fn.reshape_weight_to_matrix(weight)

            h, w = weight_mat.size()
            # randomly initialize `u` and `v`
            u = normalize(weight.new_empty(h).normal_(0, 1), dim=0, eps=fn.eps)
            v = normalize(weight.new_empty(w).normal_(0, 1), dim=0, eps=fn.eps)

        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        # We still need to assign weight back as fn.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an nn.Parameter and
        # gets added as a parameter. Instead, we register weight.data as a plain
        # attribute.
        setattr(module, fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)
        module.register_buffer(fn.name + "_v", v)
        module.register_buffer(fn.name + "_sigma", torch.ones(1).to(weight.device))

        module.register_forward_pre_hook(fn)

        module._register_state_dict_hook(SpectralNormStateDictHook(fn))
        module._register_load_state_dict_pre_hook(SpectralNormLoadStateDictPreHook(fn))
        return fn


class SpectralNormLoadStateDictPreHook(object):
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, state_dict, prefix, local_metadata, strict,
                 missing_keys, unexpected_keys, error_msgs):
        fn = self.fn
        version = local_metadata.get('spectral_norm', {}).get(fn.name + '.version', None)
        if version is None or version < 1:
            with torch.no_grad():
                weight_orig = state_dict[prefix + fn.name + '_orig']
                weight = state_dict.pop(prefix + fn.name)
                sigma = (weight_orig / weight).mean()
                weight_mat = fn.reshape_weight_to_matrix(weight_orig)
                u = state_dict[prefix + fn.name + '_u']
                v = fn._solve_v_and_rescale(weight_mat, u, sigma)
                state_dict[prefix + fn.name + '_v'] = v


class SpectralNormStateDictHook(object):
    # See docstring of SpectralNorm._version on the changes to spectral_norm.
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, module, state_dict, prefix, local_metadata):
        if 'spectral_norm' not in local_metadata:
            local_metadata['spectral_norm'] = {}
        key = self.fn.name + '.version'
        if key in local_metadata['spectral_norm']:
            raise RuntimeError("Unexpected key in metadata['spectral_norm']: {}".format(key))
        local_metadata['spectral_norm'][key] = self.fn._version


def spectral_norm_fc(module, coeff, name='weight', n_power_iterations=1, eps=1e-12, dim=None):
    r"""Applies spectral normalization to a parameter in the given module.

    .. math::
         \mathbf{W} = \dfrac{\mathbf{W}}{\sigma(\mathbf{W})} \\
         \sigma(\mathbf{W}) = \max_{\mathbf{h}: \mathbf{h} \ne 0} \dfrac{\|\mathbf{W} \mathbf{h}\|_2}{\|\mathbf{h}\|_2}

    Spectral normalization stabilizes the training of discriminators (critics)
    in Generaive Adversarial Networks (GANs) by rescaling the weight tensor
    with spectral norm :math:`\sigma` of the weight matrix calculated using
    power iteration method. If the dimension of the weight tensor is greater
    than 2, it is reshaped to 2D in power iteration method to get spectral
    norm. This is implemented via a hook that calculates spectral norm and
    rescales weight before every :meth:`~Module.forward` call.

    See `Spectral Normalization for Generative Adversarial Networks`_ .

    .. _`Spectral Normalization for Generative Adversarial Networks`: https://arxiv.org/abs/1802.05957

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
        n_power_iterations (int, optional): number of power iterations to
            calculate spectal norm
        eps (float, optional): epsilon for numerical stability in
            calculating norms
        dim (int, optional): dimension corresponding to number of outputs,
            the default is 0, except for modules that are instances of
            ConvTranspose1/2/3d, when it is 1

    Returns:
        The original module with the spectal norm hook
    """
    if dim is None:
        if isinstance(module, (torch.nn.ConvTranspose1d,
                               torch.nn.ConvTranspose2d,
                               torch.nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0
    SpectralNorm.apply(module, name, coeff, n_power_iterations, dim, eps)
    return module
