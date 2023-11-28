import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common import DepthConv

from compressai.models.priors import MeanScaleHyperprior
from compressai.models.priors import ScaleHyperprior, CompressionModel, JointAutoregressiveHierarchicalPriors
from compressai.entropy_models import EntropyBottleneck, GaussianConditional, EMGConditional, ExGConditional, ExGConditional_fixedL
from compressai.layers import MaskedConv2d

from compressai.models.utils import conv, deconv, update_registered_buffers

class Multi_Compressor_LayerPrior(nn.Module):

    def __init__(self, M = 192, N = 128, N_layers=4, N_freeze=0, **kwargs):
        super().__init__()
        
        # If M or N are not list, assume m and n channels are same for all layers
        self.M = M
        self.N = N
        if not isinstance(M, list):
            self.M = [M]*N_layers
        if not isinstance(N, list):
            self.N = [N]*N_layers
            
        # TODO : First put Quantize_Hyperprior, then append Quantize_Hyperprior_AdditionalInput
        
        self.Comp = [Quantize_Hyperprior(self.M[0],self.N[0])]
        m_prev = self.M[0]
        for m,n in zip(self.M[1:], self.N[1:]):
            self.Comp.append(Quantize_Hyperprior_AdditionalInput(m, n, m_prev))
            m_prev = m
        self.Comp = nn.ModuleList(self.Comp)
        self.Dconv = nn.ModuleList([nn.Sequential(DepthConv(m,m,k=3),DepthConv(m,m,k=3,activation='None')) for m in self.M[1:]])
        
        for i_comp in range(N_freeze):
            for param in self.Comp[i_comp].parameters():
                param.requires_grad = False
        
    def forward(self, Y_list, out_y_only = True, quantize = True, active_scales=1):
        if isinstance(Y_list, torch.Tensor):
            Y_list = list(torch.split(Y_list, self.M[0:active_scales], dim=1))
        
        y_or_yhat = 'yhat' if quantize else 'y'
        out_dict = {'likelihoods':{}, y_or_yhat:[]}
        
        y_prev = None
        for i_comp, y in enumerate(Y_list):
            if out_y_only:
                return self.Comp[i_comp](y, out_y_only, quantize)
            else:
                if quantize:
                    if i_comp == 0:
                        comp_out = self.Comp[i_comp](y, out_y_only, quantize)
                    else:
                        comp_out = self.Comp[i_comp](y, y_prev, out_y_only, quantize)
                    likelihoods = comp_out['likelihoods']
                    yhat = comp_out['yhat']
                    out_dict['likelihoods']['y'+str(i_comp)] = likelihoods['y']
                    out_dict['likelihoods']['z'+str(i_comp)] = likelihoods['z']
                    out_dict['yhat'].append(yhat)
                    y_prev = yhat
                else:
                    if i_comp == 0:
                        comp_out = self.Comp[i_comp](y, out_y_only, quantize)
                    else:
                        comp_out = self.Comp[i_comp](y, y_prev, out_y_only, quantize)
                    likelihoods = comp_out['likelihoods']
                    y = comp_out['y']
                    out_dict['likelihoods']['y'+str(i_comp)] = torch.ones(1).to(y.device)
                    out_dict['likelihoods']['z'+str(i_comp)] = torch.ones(1).to(y.device)
                    out_dict['y'].append(y)
                    y_prev = y
        out_dict[y_or_yhat] = torch.cat(out_dict[y_or_yhat], dim = 1)
        return out_dict
        
    def clear_grad_scalable(self, clear_layers=0):
        for i_comp in range(clear_layers):
            self.Comp[i_comp].zero_grad()
            
class Multi_Compressor(nn.Module):

    def __init__(self, M = 192, N = 128, N_layers=4, N_freeze=0, **kwargs):
        super().__init__()
        self.M = M
        self.N = N
        if not isinstance(M, list):
            self.M = [M]*N_layers
        if not isinstance(N, list):
            self.N = [N]*N_layers
        self.Comp = nn.ModuleList([Quantize_Hyperprior(m, n) for m,n in zip(self.M,self.N)])
        
        for i_comp in range(N_freeze):
            for param in self.Comp[i_comp].parameters():
                param.requires_grad = False
        
    def forward(self, Y_list, out_y_only = True, quantize = True, active_scales=1):
        if isinstance(Y_list, torch.Tensor):
            Y_list = list(torch.split(Y_list, self.M[0:active_scales], dim=1))
        
        y_or_yhat = 'yhat' if quantize else 'y'
        out_dict = {'likelihoods':{}, y_or_yhat:[]}
        
        for i_comp, y in enumerate(Y_list):
            if out_y_only:
                return self.Comp[i_comp](y, out_y_only, quantize)
            else:
                if quantize:
                    comp_out = self.Comp[i_comp](y, out_y_only, quantize)
                    likelihoods = comp_out['likelihoods']
                    yhat = comp_out['yhat']
                    out_dict['likelihoods']['y'+str(i_comp)] = likelihoods['y']
                    out_dict['likelihoods']['z'+str(i_comp)] = likelihoods['z']
                    out_dict['yhat'].append(yhat)
                else:
                    comp_out = self.Comp[i_comp](y, out_y_only, quantize)
                    likelihoods = comp_out['likelihoods']
                    y = comp_out['y']
                    out_dict['likelihoods']['y'+str(i_comp)] = torch.ones(1).to(y.device)
                    out_dict['likelihoods']['z'+str(i_comp)] = torch.ones(1).to(y.device)
                    out_dict['y'].append(y)
        #out_dict[y_or_yhat] = torch.cat(out_dict[y_or_yhat], dim = 1)
        return out_dict
        
    def clear_grad_scalable(self, clear_layers=0):
        for i_comp in range(clear_layers):
            self.Comp[i_comp].zero_grad()
            

class Quantize_Hyperprior(ScaleHyperprior):
    r"""Scale Hyperprior with non zero-mean Gaussian conditionals from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, M = 192, N = 128, **kwargs):
        CompressionModel.__init__(self, entropy_bottleneck_channels = N, **kwargs)
        
        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, M),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )
        
        
        
        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

    def forward(self, y, out_y_only = True, quantize = True):
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)

        if out_y_only:
            if quantize:
                return y_hat
            else:
                return y
        else:
            if quantize:
                return {
                    "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
                    "yhat": y_hat,
                }
            else:
                return {
                    "likelihoods": {"y": torch.ones(1).to(y.device), "z": torch.ones(1).to(y.device)},
                    "y": y,
                }

    def compress(self, y):
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        return {"y_hat": y_hat}
        
class Quantize_Hyperprior_AdditionalInput(Quantize_Hyperprior):
    def __init__(self, M = 192, N = 128, M_2=192, **kwargs):
        super().__init__(M = M, N = N, **kwargs)
        
        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, M),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )
        
        self.entropy_parameters = nn.Sequential(nn.Conv2d(2*M+M_2, 2*M, 1), nn.LeakyReLU(), nn.Conv2d(2*M, 2*M, 1))
        
        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

    def forward(self, y, y_prev, out_y_only = True, quantize = True):
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        z_hat_2 = self.h_s(z_hat)
        gaussian_params = self.entropy_parameters(torch.cat([z_hat_2, y_prev], dim=1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)

        if out_y_only:
            if quantize:
                return y_hat
            else:
                return y
        else:
            if quantize:
                return {
                    "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
                    "yhat": y_hat,
                }
            else:
                return {
                    "likelihoods": {"y": torch.ones(1).to(y.device), "z": torch.ones(1).to(y.device)},
                    "y": y,
                }
                
class Quantize_Hyperprior_Layered(ScaleHyperprior):
    r"""Use lower layer intput as part of hyperprior prediction
        Assume the lower layer input is concatenated with the main tensor to be coded along the channel
        dimension. When doing forward, this will first separate y into main tensor and lower layer tensor
        along the channel dimension based on N_down hyperparameter.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
        N_down (int) : Number of channels from the lower layer to be used as Hyperprior
    """

    def __init__(self, M = 192, N = 128, N_down = None, **kwargs):
        CompressionModel.__init__(self, entropy_bottleneck_channels = N, **kwargs)
        assert N_down != None
        self.N_down = N_down
        
        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, M),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )
        
        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 2 + N_down, M * 2 + N_down, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 2 + N_down, M * 2, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 2, M * 2, 1),
        )
        
        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

    def forward(self, y_combined, out_y_only = True, quantize = True):
        # y = y_combined[:,0:y_combined.shape[1]-self.N_down,:,:]
        # z_lower_layer = y_combined[:,y_combined.shape[1]-self.N_down:,:,:]
        y = y_combined[0]
        z_lower_layer = y_combined[1]
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params_z = self.h_s(z_hat)
        gaussian_params = self.entropy_parameters(torch.cat([gaussian_params_z, z_lower_layer], dim=1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)

        if out_y_only:
            if quantize:
                return y_hat
            else:
                return y
        else:
            if quantize:
                return {
                    "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
                    "yhat": y_hat,
                }
            else:
                return {
                    "likelihoods": {"y": torch.ones(1).to(y.device), "z": torch.ones(1).to(y.device)},
                    "y": y,
                }

    def compress(self, y):
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        return {"y_hat": y_hat}

class Quantize_Hyperprior2(Quantize_Hyperprior):
    def __init__(self, M = 192, N = 128, **kwargs):
        super().__init__(M = M, N = N, **kwargs)
        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, M),
            nn.LeakyReLU(inplace=True),
            deconv(M, M),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )
        
class Quantize_JointAutoHyper(JointAutoregressiveHierarchicalPriors):
    r"""Scale Hyperprior with non zero-mean Gaussian conditionals from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, M = 192, N = 128, **kwargs):
        CompressionModel.__init__(self, entropy_bottleneck_channels = N, **kwargs)
        
        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, M),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )
        
        self.context_prediction = MaskedConv2d(
            M, 2 * M, kernel_size=5, padding=2, stride=1
        )
        
        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
        )
        
        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)
        
    def forward(self, y, out_y_only = True, quantize = True):
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)
        
        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)

        if out_y_only:
            if quantize:
                return y_hat
            else:
                return y
        else:
            if quantize:
                return {
                    "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
                    "yhat": y_hat,
                }
            else:
                return {
                    "likelihoods": {"y": torch.ones(1).to(y.device), "z": torch.ones(1).to(y.device)},
                    "y": y,
                }

    def compress(self, y):
        raise NotImplementedError
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        raise NotImplementedError
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        return {"y_hat": y_hat}
        
class GaussianConditional_cnoise(GaussianConditional):

    def quantize(self, inputs, mode, means=None, cnoise=1):
        # type: (Tensor, str, Optional[Tensor]) -> Tensor
        if mode not in ("noise", "dequantize", "symbols"):
            raise ValueError(f'Invalid quantization mode: "{mode}"')

        if mode == "noise":
            half = float(cnoise)/2
            noise = torch.empty_like(inputs).uniform_(-half, half)
            inputs = inputs + noise.to(inputs.device)
            return inputs

        outputs = inputs.clone()
        if means is not None:
            outputs -= means

        outputs = torch.round(outputs)

        if mode == "dequantize":
            if means is not None:
                outputs += means
            return outputs

        assert mode == "symbols", mode
        outputs = outputs.int()
        return outputs
        
    def forward(self, inputs, scales, means=None, cnoise=1):
        # type: (Tensor, Tensor, Optional[Tensor]) -> Tuple[Tensor, Tensor]
        outputs = self.quantize(
            inputs, "noise" if self.training else "dequantize", means, cnoise
        )
        likelihood = self._likelihood(outputs, scales, means)
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)
        return outputs, likelihood
        
        
class Quantize_Hyperprior_cnoise(ScaleHyperprior):
    r"""Scale Hyperprior with non zero-mean Gaussian conditionals from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """
    # Use a coefficient cnoise to control the amount of noise added during training

    def __init__(self, M = 192, N = 128, **kwargs):
        CompressionModel.__init__(self, entropy_bottleneck_channels = N, **kwargs)
        
        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, M),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )
        self.gaussian_conditional_cnoise = GaussianConditional_cnoise(None)
        self.N = int(N)
        self.M = int(M)

    def forward(self, y, out_y_only = True, quantize = True, cnoise=1):
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional_cnoise(y, scales_hat, means=means_hat, cnoise=cnoise)

        if out_y_only:
            if quantize:
                return y_hat
            else:
                return y
        else:
            if quantize:
                return {
                    "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
                    "yhat": y_hat,
                }
            else:
                return {
                    "likelihoods": {"y": torch.ones(1).to(y.device), "z": torch.ones(1).to(y.device)},
                    "y": y,
                }

    def compress(self, y):
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        return {"y_hat": y_hat}
        
        
        
class Quantize_Hyperprior_EMG(ScaleHyperprior):
    r"""Scale Hyperprior with non zero-mean Gaussian conditionals from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, M = 192, N = 128, **kwargs):
        CompressionModel.__init__(self, entropy_bottleneck_channels = N, **kwargs)
        
        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, M),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 3, stride=1, kernel_size=3),
        )
        self.emg_conditional = EMGConditional(None)
        self.N = int(N)
        self.M = int(M)

    def forward(self, y, out_y_only = True, quantize = True):
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        emg_params = self.h_s(z_hat)
        mu_hat, sigma_hat, K_hat = emg_params.chunk(3, 1)
        sigma_hat = torch.abs(sigma_hat)
        K_hat = torch.abs(K_hat)
        # sigma_hat = torch.exp(sigma_hat) + 0.01
        # K_hat = torch.exp(K_hat) + 0.1
        y_hat, y_likelihoods = self.emg_conditional(y, mu_hat, sigma_hat, K_hat)

        if out_y_only:
            if quantize:
                return y_hat
            else:
                return y
        else:
            if quantize:
                return {
                    "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
                    "yhat": y_hat,
                }
            else:
                return {
                    "likelihoods": {"y": torch.ones(1).to(y.device), "z": torch.ones(1).to(y.device)},
                    "y": y,
                }

    def compress(self, y):
        assert NotImplementedError
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert NotImplementedError
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        return {"y_hat": y_hat}
        
        
class Quantize_Hyperprior_ExG(ScaleHyperprior):
    r"""Scale Hyperprior with non zero-mean Gaussian conditionals from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, M = 192, N = 128, **kwargs):
        CompressionModel.__init__(self, entropy_bottleneck_channels = N, **kwargs)
        
        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, M),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 3, stride=1, kernel_size=3),
        )
        self.exg_conditional = ExGConditional(None)
        self.N = int(N)
        self.M = int(M)

    def forward(self, y, out_y_only = True, quantize = True):
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        exg_params = self.h_s(z_hat)
        mu_hat, sigma_hat, lmbda_hat = exg_params.chunk(3, 1)
        sigma_hat = torch.abs(sigma_hat)
        lmbda_hat = torch.abs(lmbda_hat)
        y_hat, y_likelihoods = self.exg_conditional(y, mu_hat, sigma_hat, lmbda_hat)

        if out_y_only:
            if quantize:
                return y_hat
            else:
                return y
        else:
            if quantize:
                return {
                    "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
                    "yhat": y_hat,
                }
            else:
                return {
                    "likelihoods": {"y": torch.ones(1).to(y.device), "z": torch.ones(1).to(y.device)},
                    "y": y,
                }

    def compress(self, y):
        assert NotImplementedError
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert NotImplementedError
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        return {"y_hat": y_hat}
        
        
class Quantize_Hyperprior_ExG_FixedL(ScaleHyperprior):
    r"""Scale Hyperprior with non zero-mean Gaussian conditionals from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, M = 192, N = 128, **kwargs):
        CompressionModel.__init__(self, entropy_bottleneck_channels = N, **kwargs)
        
        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, M),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )
        self.exg_conditional = ExGConditional_fixedL(None)
        self.N = int(N)
        self.M = int(M)

    def forward(self, y, out_y_only = True, quantize = True):
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        exg_params = self.h_s(z_hat)
        mu_hat, sigma_hat= exg_params.chunk(2, 1)
        sigma_hat = torch.abs(sigma_hat)
        y_hat, y_likelihoods = self.exg_conditional(y, mu_hat, sigma_hat)

        if out_y_only:
            if quantize:
                return y_hat
            else:
                return y
        else:
            if quantize:
                return {
                    "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
                    "yhat": y_hat,
                }
            else:
                return {
                    "likelihoods": {"y": torch.ones(1).to(y.device), "z": torch.ones(1).to(y.device)},
                    "y": y,
                }

    def compress(self, y):
        assert NotImplementedError
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert NotImplementedError
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        return {"y_hat": y_hat}