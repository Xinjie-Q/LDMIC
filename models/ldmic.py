import torch
import math
from torch import nn
from compressai.models.utils import update_registered_buffers, conv, deconv
from compressai.entropy_models import GaussianConditional
from compressai.models import CompressionModel, get_scale_table
from compressai.ops import ste_round
from compressai.layers import ResidualBlock, GDN, MaskedConv2d, conv3x3, ResidualBlockWithStride
from deepspeed.profiling.flops_profiler import get_model_profile
import torch.nn.functional as F
from entropy_model import Hyperprior, CheckMaskedConv2d
from torch.autograd import Variable
from math import exp
import os
from compressai.ans import BufferedRansEncoder, RansDecoder


class JointContextTransfer(nn.Module):
    def __init__(self, channels):
        super(JointContextTransfer, self).__init__()
        self.rb1 = ResidualBlock(channels, channels)
        self.rb2 = ResidualBlock(channels, channels)
        self.attn = EfficientAttention(key_in_channels=channels, query_in_channels=channels, key_channels=channels//8, 
            head_count=2, value_channels=channels//4)

        self.refine = nn.Sequential(
            ResidualBlock(channels*2, channels),
            ResidualBlock(channels, channels))

    def forward(self, x_left, x_right):
        B, C, H, W = x_left.size()
        identity_left, identity_right = x_left, x_right
        x_left, x_right = self.rb2(self.rb1(x_left)), self.rb2(self.rb1(x_right))
        A_right_to_left, A_left_to_right = self.attn(x_left, x_right), self.attn(x_right, x_left)
        compact_left = identity_left + self.refine(torch.cat((A_right_to_left, x_left), dim=1))
        compact_right = identity_right + self.refine(torch.cat((A_left_to_right, x_right), dim=1))
        return compact_left, compact_right


class Multi_JointContextTransfer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.rb = nn.Sequential(
            ResidualBlock(channels, channels),
            ResidualBlock(channels, channels),
        )
        self.aggeregate_module = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, None, None)),
        )
        self.attn = EfficientAttention(key_in_channels=channels, query_in_channels=channels, key_channels=channels//8, 
            head_count=2, value_channels=channels//4)

        self.refine = nn.Sequential(
            ResidualBlock(channels*2, channels),
            ResidualBlock(channels, channels))

    def forward(self, x, num_camera):
        identity_list = x.chunk(num_camera, 0)
        rb_x = self.rb(x)
        rb_x_list = rb_x.chunk(num_camera, 0)
        compact_list = []
        for idx, rb in enumerate(rb_x_list):
            other_rb = [r.unsqueeze(2) for i, r in enumerate(rb_x_list) if i!=idx]
            other_rb = torch.cat(other_rb, dim=2)
            aggeregate_rb = self.aggeregate_module(other_rb).squeeze(2)
            #print(rb.shape, aggeregate_rb.shape)
            A_other_camera_to_current = self.attn(rb, aggeregate_rb)
            compact = identity_list[idx] + self.refine(torch.cat([A_other_camera_to_current, rb], dim=1))
            compact_list.append(compact)
        
        return torch.cat(compact_list, dim=0)


class EfficientAttention(nn.Module):
    def __init__(self, key_in_channels, query_in_channels, key_channels=32, head_count=8, value_channels=64):
        super().__init__()
        self.in_channels = query_in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(key_in_channels, key_channels, 1)
        self.queries = nn.Conv2d(query_in_channels, key_channels, 1)
        self.values = nn.Conv2d(key_in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, query_in_channels, 1)

    def forward(self, target, input):
        n, _, h, w = input.size()
        keys = self.keys(input).reshape((n, self.key_channels, h * w))
        queries = self.queries(target).reshape(n, self.key_channels, h * w)
        values = self.values(input).reshape((n, self.value_channels, h * w))
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count
        
        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:, i * head_key_channels: (i + 1) * head_key_channels,:], dim=2)
            query = F.softmax(queries[:, i * head_key_channels: (i + 1) * head_key_channels,:], dim=1)
            value = values[:, i * head_value_channels: (i + 1) * head_value_channels, :]
            context = key @ value.transpose(1, 2)
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, h, w)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        reprojected_value = self.reprojection(aggregated_values)
        attention = reprojected_value #+ input_

        return attention

    def parallel_forward(self, target, input):
        n, _, h, w = input.size()
        keys = self.keys(input).reshape((n, self.key_channels, h * w))
        queries = self.queries(target).reshape(n, self.key_channels, h * w)
        values = self.values(input).reshape((n, self.value_channels, h * w))
        
        keys = keys.permute(0, 2, 1).reshape(n, h*w, self.head_count, -1).permute(0, 2, 3, 1).reshape(n*self.head_count, -1, h*w)
        queries = queries.permute(0, 2, 1).reshape(n, h*w, self.head_count, -1).permute(0, 2, 3, 1).reshape(n*self.head_count, -1, h*w)
        values = values.permute(0, 2, 1).reshape(n, h*w, self.head_count, -1).permute(0, 2, 3, 1).reshape(n*self.head_count, -1, h*w)

        key = F.softmax(keys, dim=2)
        queries = F.softmax(queries, dim=1)
        context = key @ value.transpose(1, 2)
        attended_values = (context.transpose(1, 2) @ query).reshape(n, -1, h, w)
        reprojected_value = self.reprojection(aggregated_values)
        attention = reprojected_value #+ target
        return attention


class LDMIC(nn.Module):
    def __init__(self, N = 128, M = 192, decode_atten=JointContextTransfer):
        super().__init__()
        self.encoder = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2)
        )
        self.hyperprior = Hyperprior(in_planes=M, mid_planes=N, out_planes=M*2)
        self.context_prediction = MaskedConv2d(
            M, M*2, kernel_size=5, padding=2, stride=1
        )
        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
        )
        self.gaussian_conditional = GaussianConditional(None)
        self.M = M
        self.atten_3 = decode_atten(M)
        self.decoder_1 = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),)
        self.atten_4 = decode_atten(N)
        self.decoder_2 = nn.Sequential(
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 3, kernel_size=5, stride=2)
        )

    def forward(self, x):
        x_left, x_right = x[0], x[1] #x.chunk(2, 1) 

        y_left, y_right = self.encoder(x_left), self.encoder(x_right)
        left_params, z_left_likelihoods, z_left_hat = self.hyperprior(y_left, out_z=True)
        right_params, z_right_likelihoods, z_right_hat = self.hyperprior(y_right, out_z=True)
        y_left_hat = self.gaussian_conditional.quantize(
            y_left, "noise" if self.training else "dequantize"
        )
        y_right_hat = self.gaussian_conditional.quantize(
            y_right, "noise" if self.training else "dequantize"
        )
        ctx_left_params = self.context_prediction(y_left_hat)
        ctx_right_params = self.context_prediction(y_right_hat)

        gaussian_left_params = self.entropy_parameters(torch.cat([left_params, ctx_left_params], 1))
        gaussian_right_params = self.entropy_parameters(torch.cat([right_params, ctx_right_params], 1))
        
        left_means_hat, left_scales_hat = gaussian_left_params.chunk(2, 1)
        right_means_hat, right_scales_hat = gaussian_right_params.chunk(2, 1)
 
        _, y_left_likelihoods = self.gaussian_conditional(y_left, left_scales_hat, means=left_means_hat)
        _, y_right_likelihoods = self.gaussian_conditional(y_right, right_scales_hat, means=right_means_hat)


        y_left_ste, y_right_ste = ste_round(y_left - left_means_hat) + left_means_hat, ste_round(y_right - right_means_hat) + right_means_hat
        y_left, y_right = self.atten_3(y_left_ste, y_right_ste)
        y_left, y_right = self.atten_4(self.decoder_1(y_left), self.decoder_1(y_right))
        x_left_hat, x_right_hat = self.decoder_2(y_left), self.decoder_2(y_right)
        
        return {
            "x_hat": [x_left_hat, x_right_hat],
            "likelihoods": [{"y": y_left_likelihoods, "z": z_left_likelihoods}, {"y":y_right_likelihoods, "z":z_right_likelihoods}],
            "feature": [y_left_ste, y_right_ste, z_left_hat, z_right_hat, left_means_hat, right_means_hat],
        }

    def aux_loss(self):
        """Return a list of the auxiliary entropy bottleneck over module(s)."""
        aux_loss_list = []
        for m in self.modules():
            if isinstance(m, CompressionModel):
                aux_loss_list.append(m.aux_loss())
        return aux_loss_list

    def fix_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.hyperprior.parameters():
            p.requires_grad = False
        for p in self.context_prediction.parameters():
            p.requires_grad = False
        for p in self.entropy_parameters.parameters():
            p.requires_grad = False
        for p in self.gaussian_conditional.parameters():
            p.requires_grad = False

    def load_encoder(self, current_state_dict, checkpoint):
        encoder_dict = {k.replace("g_a", "encoder"): v for k, v in checkpoint.items() if "g_a" in k}
        context_prediction_dict = {k: v for k, v in checkpoint.items() if "context_prediction" in k}
        entropy_parameters_dict = {k: v for k, v in checkpoint.items() if "entropy_parameters" in k}
        gaussian_conditional_dict = {k: v for k, v in checkpoint.items() if "gaussian_conditional" in k}
        hyperprior_dict = {}
        for k, v in checkpoint.items():
            if "h_a" in k:
                hyperprior_dict[k.replace("h_a", "hyperprior.hyper_encoder")] = v
            elif "h_s" in k:
                hyperprior_dict[k.replace("h_s", "hyperprior.hyper_decoder")] = v
            elif "entropy_bottleneck" in k:
                hyperprior_dict[k.replace("entropy_bottleneck", "hyperprior.entropy_bottleneck")] = v

        current_state_dict.update(encoder_dict)
        current_state_dict.update(hyperprior_dict)
        current_state_dict.update(context_prediction_dict)
        current_state_dict.update(entropy_parameters_dict)
        current_state_dict.update(gaussian_conditional_dict)
        #print(current_state_dict.keys())
        #input()
        return current_state_dict

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        update_registered_buffers(
            self.hyperprior.entropy_bottleneck,
            "hyperprior.entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= self.hyperprior.entropy_bottleneck.update(force=force)
        return updated

    def compress(self, x):
        x_left, x_right = x[0], x[1]
        left_dict = self.encode(x_left)
        right_dict = self.encode(x_right)
        return left_dict, right_dict

    def decompress(self, left_dict, right_dict):
        y_left_hat = self.decode(left_dict["strings"], left_dict["shape"])
        y_right_hat = self.decode(right_dict["strings"], right_dict["shape"])
        y_left, y_right = self.atten_3(y_left_hat, y_right_hat)
        y_left, y_right = self.atten_4(self.decoder_1(y_left), self.decoder_1(y_right))
        x_left_hat, x_right_hat = self.decoder_2(y_left).clamp_(0, 1), self.decoder_2(y_right).clamp_(0, 1)
        return {
            "x_hat": [x_left_hat, x_right_hat],
        }  

    def encode(self, x):
        y = self.encoder(x)
        params, z_hat, z_strings = self.hyperprior.compress(y)
        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))

        y_strings = []
        for i in range(y.size(0)):
            string = self._compress_ar(
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
            y_strings.append(string)

        return {"strings": [y_strings, z_strings], "shape": z_hat.size()[-2:]}

    def _compress_ar(self, y_hat, params, height, width, kernel_size, padding):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []

        # Warning, this is slow...
        # TODO: profile the calls to the bindings...
        masked_weight = self.context_prediction.weight * self.context_prediction.mask
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    masked_weight,
                    bias=self.context_prediction.bias,
                )

                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)
                means_hat, scales_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)

                y_crop = y_crop[:, :, padding, padding]
                y_q = self.gaussian_conditional.quantize(y_crop, "symbols", means_hat)
                y_hat[:, :, h + padding, w + padding] = y_q + means_hat

                symbols_list.extend(y_q.squeeze().tolist())
                indexes_list.extend(indexes.squeeze().tolist())

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )

        string = encoder.flush()
        return string

    def decode(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2

        # FIXME: we don't respect the default entropy coder and directly call the
        # range ANS decoder
        params, z_hat = self.hyperprior.decompress(strings[1], shape)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        y_hat = torch.zeros(
            (z_hat.size(0), self.M, y_height + 2 * padding, y_width + 2 * padding),
            device=z_hat.device,
        )

        for i, y_string in enumerate(strings[0]):
            self._decompress_ar(
                y_string,
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )

        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        #x_hat = self.g_s(y_hat).clamp_(0, 1)
        return y_hat

    def _decompress_ar(self, y_string, y_hat, params, height, width, kernel_size, padding):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        # Warning: this is slow due to the auto-regressive nature of the
        # decoding... See more recent publication where they use an
        # auto-regressive module on chunks of channels for faster decoding...
        for h in range(height):
            for w in range(width):
                # only perform the 5x5 convolution on a cropped tensor
                # centered in (h, w)
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    self.context_prediction.weight,
                    bias=self.context_prediction.bias,
                )
                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                means_hat, scales_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)
                rv = decoder.decode_stream(
                    indexes.squeeze().tolist(), cdf, cdf_lengths, offsets
                )
                rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
                rv = self.gaussian_conditional.dequantize(rv, means_hat)

                hp = h + padding
                wp = w + padding
                y_hat[:, :, hp : hp + 1, wp : wp + 1] = rv

class LDMIC_checkboard(nn.Module):
    def __init__(self, N = 128, M = 192, decode_atten=JointContextTransfer, training=False):
        super().__init__()
        self.encoder = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2)
        )
        self.atten_3 = decode_atten(M)
        self.decoder_1 = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),)
        self.atten_4 = decode_atten(N)
        self.decoder_2 = nn.Sequential(
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 3, kernel_size=5, stride=2)
        )
        
        self.hyperprior = Hyperprior(in_planes=M, mid_planes=N, out_planes=M*2)
        self.context_prediction = CheckMaskedConv2d(
            M, M*2, kernel_size=5, padding=2, stride=1
        )
        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
        )
        self.gaussian_conditional = GaussianConditional(None)
        self.M = M
        self.N = N
        if training:
            self.training_ctx_params_anchor = torch.zeros([8, self.M * 2, 16, 16]).cuda()

    def forward(self, x):
        x_left, x_right = x[0], x[1] #x.chunk(2, 1)
        y_left, y_right = self.encoder(x_left), self.encoder(x_right)
        

        y_left_ste, y_left_likelihoods = self.forward_entropy(y_left)
        y_right_ste, y_right_likelihoods = self.forward_entropy(y_right) 

        y_left, y_right = self.atten_3(y_left_ste, y_right_ste)
        y_left, y_right = self.atten_4(self.decoder_1(y_left), self.decoder_1(y_right))
        x_left_hat, x_right_hat = self.decoder_2(y_left), self.decoder_2(y_right)

        return {
            "x_hat": [x_left_hat, x_right_hat],
            "likelihoods": [y_left_likelihoods, y_right_likelihoods],
        }

    def aux_loss(self):
        """Return a list of the auxiliary entropy bottleneck over module(s)."""
        aux_loss_list = []
        for m in self.modules():
            if isinstance(m, CompressionModel):
                aux_loss_list.append(m.aux_loss())
        return aux_loss_list

    def forward_entropy(self, y):
        params, z_likelihoods = self.hyperprior(y)

        batch_size, _, y_height, y_width = y.size()
        # compress anchor
        if self.training:
            ctx_params_anchor = self.training_ctx_params_anchor[:batch_size]
        else:
            ctx_params_anchor = torch.zeros([batch_size, self.M * 2, y_height, y_width]).to(y.device)
        gaussian_params_anchor = self.entropy_parameters(
            torch.cat([ctx_params_anchor, params], dim=1)
        )
        scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        anchor = self.get_anchor(y_hat)
        ctx_params_non_anchor = self.context_prediction(anchor)
        gaussian_params_non_anchor = self.entropy_parameters(
            torch.cat([ctx_params_non_anchor, params], dim=1)
        )
        scales_non_anchor, means_non_anchor = gaussian_params_non_anchor.chunk(2, 1)
        scales_hat, means_hat = self.merge(scales_anchor, means_anchor, 
            scales_non_anchor, means_non_anchor)
        _, y_likelihoods = self.gaussian_conditional(y, scales=scales_hat, means=means_hat)
        y_ste = ste_round(y-means_hat) + means_hat

        return y_ste, {"y": y_likelihoods, "z": z_likelihoods}

    def merge(self, scales_anchor, means_anchor, scales_non_anchor, means_non_anchor, mask_type="A"):
        scales_hat = scales_anchor.clone()
        means_hat = means_anchor.clone()
        if mask_type == "A":
            scales_hat[:, :, 0::2, 0::2] = scales_non_anchor[:, :, 0::2, 0::2]
            scales_hat[:, :, 1::2, 1::2] = scales_non_anchor[:, :, 1::2, 1::2]
            means_hat[:, :, 0::2, 0::2] = means_non_anchor[:, :, 0::2, 0::2]
            means_hat[:, :, 1::2, 1::2] = means_non_anchor[:, :, 1::2, 1::2]
        else:
            scales_hat[:, :, 0::2, 1::2] = scales_non_anchor[:, :, 0::2, 1::2]
            scales_hat[:, :, 1::2, 0::2] = scales_non_anchor[:, :, 1::2, 0::2]
            means_hat[:, :, 0::2, 1::2] = means_non_anchor[:, :, 0::2, 1::2]
            means_hat[:, :, 1::2, 0::2] = means_non_anchor[:, :, 1::2, 0::2]
        return scales_hat, means_hat

    def get_anchor(self, y_hat, mask_type="A"):
        y_anchor = y_hat.clone()
        if mask_type == "A":
            y_anchor[:, :, 0::2, 0::2] = 0
            y_anchor[:, :, 1::2, 1::2] = 0
        else:
            y_anchor[:, :, 0::2, 1::2] = 0
            y_anchor[:, :, 1::2, 0::2] = 0
        return y_anchor

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        update_registered_buffers(
            self.hyperprior.entropy_bottleneck,
            "hyperprior.entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= self.hyperprior.entropy_bottleneck.update(force=force)
        return updated

    def compress(self, x):
        x_left, x_right = x[0], x[1]
        left_dict = self.encode(x_left)
        right_dict = self.encode(x_right)
        return left_dict, right_dict

    def decompress(self, left_dict, right_dict):
        y_left_hat = self.decode(left_dict["strings"], left_dict["shape"])
        y_right_hat = self.decode(right_dict["strings"], right_dict["shape"])
        #print(y_left_hat[0, 0, 0, 0:10], y_right_hat[0, 0, 0, 0:10])
        y_left, y_right = self.atten_3(y_left_hat, y_right_hat)
        #print(y_left[0, 0, 0, 0:10], y_right[0, 0, 0, 0:10])
        y_left, y_right = self.atten_4(self.decoder_1(y_left), self.decoder_1(y_right))
        #print(y_left[0, 0, 0, 0:10], y_right[0, 0, 0, 0:10])
        x_left_hat, x_right_hat = self.decoder_2(y_left), self.decoder_2(y_right) #.clamp_(0, 1), self.decoder_2(y_right).clamp_(0, 1)
        
        return {
            "x_hat": [x_left_hat, x_right_hat],
        }   

    def encode(self, x):
        """
        Slice y into four parts A, B, C and D. Encode and Decode them separately.
        Workflow is described in Figure 1 and Figure 2 in Supplementary Material.
        NA  A   ->  A B
        A  NA       C D
        After padding zero:
            anchor : 0 B     non-anchor: A 0
                     c 0                 0 D
        """
        batch_size, channel, x_height, x_width = x.shape

        y = self.encoder(x)

        y_a = y[:, :, 0::2, 0::2]
        y_d = y[:, :, 1::2, 1::2]
        y_b = y[:, :, 0::2, 1::2]
        y_c = y[:, :, 1::2, 0::2]

        params, z_hat, z_strings = self.hyperprior.compress(y)

        anchor = torch.zeros_like(y).to(x.device)
        anchor[:, :, 0::2, 1::2] = y[:, :, 0::2, 1::2]
        anchor[:, :, 1::2, 0::2] = y[:, :, 1::2, 0::2]
        ctx_params_anchor = torch.zeros([batch_size, self.M * 2, x_height // 16, x_width // 16]).to(x.device)
        gaussian_params_anchor = self.entropy_parameters(
            torch.cat([ctx_params_anchor, params], dim=1)
        )
        scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)

        scales_b = scales_anchor[:, :, 0::2, 1::2]
        means_b = means_anchor[:, :, 0::2, 1::2]
        indexes_b = self.gaussian_conditional.build_indexes(scales_b)
        y_b_strings = self.gaussian_conditional.compress(y_b, indexes_b, means_b)
        y_b_quantized = self.gaussian_conditional.decompress(y_b_strings, indexes_b, means=means_b)

        scales_c = scales_anchor[:, :, 1::2, 0::2]
        means_c = means_anchor[:, :, 1::2, 0::2]
        indexes_c = self.gaussian_conditional.build_indexes(scales_c)
        y_c_strings = self.gaussian_conditional.compress(y_c, indexes_c, means_c)
        y_c_quantized = self.gaussian_conditional.decompress(y_c_strings, indexes_c, means=means_c)

        anchor_quantized = torch.zeros_like(y).to(x.device)
        anchor_quantized[:, :, 0::2, 1::2] = y_b_quantized[:, :, :, :]
        anchor_quantized[:, :, 1::2, 0::2] = y_c_quantized[:, :, :, :]

        ctx_params_non_anchor = self.context_prediction(anchor_quantized)
        gaussian_params_non_anchor = self.entropy_parameters(
            torch.cat([ctx_params_non_anchor, params], dim=1)
        )
        scales_non_anchor, means_non_anchor = gaussian_params_non_anchor.chunk(2, 1)

        scales_a = scales_non_anchor[:, :, 0::2, 0::2]
        means_a = means_non_anchor[:, :, 0::2, 0::2]
        indexes_a = self.gaussian_conditional.build_indexes(scales_a)
        y_a_strings = self.gaussian_conditional.compress(y_a, indexes_a, means=means_a)

        scales_d = scales_non_anchor[:, :, 1::2, 1::2]
        means_d = means_non_anchor[:, :, 1::2, 1::2]
        indexes_d = self.gaussian_conditional.build_indexes(scales_d)
        y_d_strings = self.gaussian_conditional.compress(y_d, indexes_d, means=means_d)

        return {
            "strings": [y_a_strings, y_b_strings, y_c_strings, y_d_strings, z_strings],
            "shape": z_hat.size()[-2:]
        }

    def decode(self, strings, shape):
        """
        Slice y into four parts A, B, C and D. Encode and Decode them separately.
        Workflow is described in Figure 1 and Figure 2 in Supplementary Material.
        NA  A   ->  A B
        A  NA       C D
        After padding zero:
            anchor : 0 B     non-anchor: A 0
                     c 0                 0 D
        """
        params, z_hat = self.hyperprior.decompress(strings[4], shape)
        #z_hat = self.entropy_bottleneck.decompress(strings[4], shape)
        #params = self.h_s(z_hat)

        batch_size, channel, z_height, z_width = z_hat.shape
        ctx_params_anchor = torch.zeros([batch_size, self.M * 2, z_height * 4, z_width * 4]).to(z_hat.device)
        gaussian_params_anchor = self.entropy_parameters(
            torch.cat([ctx_params_anchor, params], dim=1)
        )
        scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)

        scales_b = scales_anchor[:, :, 0::2, 1::2]
        means_b = means_anchor[:, :, 0::2, 1::2]
        indexes_b = self.gaussian_conditional.build_indexes(scales_b)
        y_b_quantized = self.gaussian_conditional.decompress(strings[1], indexes_b, means=means_b)

        scales_c = scales_anchor[:, :, 1::2, 0::2]
        means_c = means_anchor[:, :, 1::2, 0::2]
        indexes_c = self.gaussian_conditional.build_indexes(scales_c)
        y_c_quantized = self.gaussian_conditional.decompress(strings[2], indexes_c, means=means_c)

        anchor_quantized = torch.zeros([batch_size, self.M, z_height * 4, z_width * 4]).to(z_hat.device)
        anchor_quantized[:, :, 0::2, 1::2] = y_b_quantized[:, :, :, :]
        anchor_quantized[:, :, 1::2, 0::2] = y_c_quantized[:, :, :, :]

        ctx_params_non_anchor = self.context_prediction(anchor_quantized)
        gaussian_params_non_anchor = self.entropy_parameters(
            torch.cat([ctx_params_non_anchor, params], dim=1)
        )
        scales_non_anchor, means_non_anchor = gaussian_params_non_anchor.chunk(2, 1)

        scales_a = scales_non_anchor[:, :, 0::2, 0::2]
        means_a = means_non_anchor[:, :, 0::2, 0::2]
        indexes_a = self.gaussian_conditional.build_indexes(scales_a)
        y_a_quantized = self.gaussian_conditional.decompress(strings[0], indexes_a, means=means_a)

        scales_d = scales_non_anchor[:, :, 1::2, 1::2]
        means_d = means_non_anchor[:, :, 1::2, 1::2]
        indexes_d = self.gaussian_conditional.build_indexes(scales_d)
        y_d_quantized = self.gaussian_conditional.decompress(strings[3], indexes_d, means=means_d)

        # Add non_anchor_quantized
        anchor_quantized[:, :, 0::2, 0::2] = y_a_quantized[:, :, :, :]
        anchor_quantized[:, :, 1::2, 1::2] = y_d_quantized[:, :, :, :]

        #print(anchor_quantized[0, 0, 0, :])
        return anchor_quantized 

class Multi_LDMIC(nn.Module):
    def __init__(self, N = 128, M = 192, decode_atten=Multi_JointContextTransfer):
        super().__init__()
        self.encoder = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2)
        )

        self.atten_3 = decode_atten(M)
        self.decoder_1 = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),)
        self.atten_4 = decode_atten(N)
        self.decoder_2 = nn.Sequential(
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 3, kernel_size=5, stride=2)
        )
        
        self.hyperprior = Hyperprior(in_planes=M, mid_planes=N, out_planes=M*2)
        self.context_prediction = MaskedConv2d(
            M, M*2, kernel_size=5, padding=2, stride=1
        )
        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
        )
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x):
        num_camera = len(x)

        x = torch.cat(x, dim=0)
        y = self.encoder(x)
        params, z_likelihoods = self.hyperprior(y)
        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(torch.cat([params, ctx_params], 1))
        means_hat, scales_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        y_ste = ste_round(y-means_hat)+means_hat

        y_ste = self.decoder_1(self.atten_3(y_ste, num_camera))
        x_hat = self.decoder_2(self.atten_4(y_ste, num_camera))
        
        x_hat_list = x_hat.chunk(num_camera, 0)
        z_likelihoods_list = z_likelihoods.chunk(num_camera, 0)
        y_likelihoods_list = y_likelihoods.chunk(num_camera, 0)
        likelihoods = [{"y": y_likelihood, "z": z_likelihood} for y_likelihood, z_likelihood in zip(y_likelihoods_list, z_likelihoods_list)]

        return {
            "x_hat": x_hat_list,
            "likelihoods": likelihoods,
        }

    def aux_loss(self):
        """Return a list of the auxiliary entropy bottleneck over module(s)."""
        aux_loss_list = []
        for m in self.modules():
            if isinstance(m, CompressionModel):
                aux_loss_list.append(m.aux_loss())
        return aux_loss_list

    def fix_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.hyperprior.parameters():
            p.requires_grad = False
        for p in self.context_prediction.parameters():
            p.requires_grad = False
        for p in self.entropy_parameters.parameters():
            p.requires_grad = False
        for p in self.gaussian_conditional.parameters():
            p.requires_grad = False

    def load_encoder(self, current_state_dict, checkpoint):
        encoder_dict = {k.replace("g_a", "encoder"): v for k, v in checkpoint.items() if "g_a" in k}
        context_prediction_dict = {k: v for k, v in checkpoint.items() if "context_prediction" in k}
        entropy_parameters_dict = {k: v for k, v in checkpoint.items() if "entropy_parameters" in k}
        gaussian_conditional_dict = {k: v for k, v in checkpoint.items() if "gaussian_conditional" in k}
        hyperprior_dict = {}
        for k, v in checkpoint.items():
            if "h_a" in k:
                hyperprior_dict[k.replace("h_a", "hyperprior.hyper_encoder")] = v
            elif "h_s" in k:
                hyperprior_dict[k.replace("h_s", "hyperprior.hyper_decoder")] = v
            elif "entropy_bottleneck" in k:
                hyperprior_dict[k.replace("entropy_bottleneck", "hyperprior.entropy_bottleneck")] = v

        current_state_dict.update(encoder_dict)
        current_state_dict.update(hyperprior_dict)
        current_state_dict.update(context_prediction_dict)
        current_state_dict.update(entropy_parameters_dict)
        current_state_dict.update(gaussian_conditional_dict)
        return current_state_dict

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        update_registered_buffers(
            self.hyperprior.entropy_bottleneck,
            "hyperprior.entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= self.hyperprior.entropy_bottleneck.update(force=force)
        return updated

class Multi_LDMIC_checkboard(nn.Module):
    def __init__(self, N = 128, M = 192, decode_atten=Multi_JointContextTransfer, training=False):
        super().__init__()
        self.encoder = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2)
        )
    
        self.atten_3 = decode_atten(M)
        self.decoder_1 = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),)
        self.atten_4 = decode_atten(N)
        self.decoder_2 = nn.Sequential(
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 3, kernel_size=5, stride=2)
        )
        
        self.hyperprior = Hyperprior(in_planes=M, mid_planes=N, out_planes=M*2)
        self.context_prediction = CheckMaskedConv2d(
            M, M*2, kernel_size=5, padding=2, stride=1
        )
        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
        )
        self.gaussian_conditional = GaussianConditional(None)
        self.M = M

        if training:
            self.training_ctx_params_anchor = torch.zeros([8*7, M * 2, 16, 16]).cuda()

    def forward(self, x):
        num_camera = len(x)
        x = torch.cat(x, dim=0)
        y = self.encoder(x)

        y_ste, y_likelihoods, z_likelihoods = self.forward_entropy(y)
        y_ste = self.decoder_1(self.atten_3(y_ste, num_camera))
        x_hat = self.decoder_2(self.atten_4(y_ste, num_camera))
        
        x_hat_list = x_hat.chunk(num_camera, 0)
        z_likelihoods_list = z_likelihoods.chunk(num_camera, 0)
        y_likelihoods_list = y_likelihoods.chunk(num_camera, 0)
        likelihoods = [{"y": y_likelihood, "z": z_likelihood} for y_likelihood, z_likelihood in zip(y_likelihoods_list, z_likelihoods_list)]

        return {
            "x_hat": x_hat_list,
            "likelihoods": likelihoods,
        }

    def aux_loss(self):
        """Return a list of the auxiliary entropy bottleneck over module(s)."""
        aux_loss_list = []
        for m in self.modules():
            if isinstance(m, CompressionModel):
                aux_loss_list.append(m.aux_loss())
        return aux_loss_list


    def forward_entropy(self, y):
        params, z_likelihoods = self.hyperprior(y)

        batch_size, _, y_height, y_width = y.size()
        # compress anchor
        if self.training:
            ctx_params_anchor = self.training_ctx_params_anchor[:batch_size]
        else:
            ctx_params_anchor = torch.zeros([batch_size, self.M * 2, y_height, y_width]).to(y.device)
        gaussian_params_anchor = self.entropy_parameters(
            torch.cat([ctx_params_anchor, params], dim=1)
        )
        scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        anchor = self.get_anchor(y_hat)
        ctx_params_non_anchor = self.context_prediction(anchor)
        gaussian_params_non_anchor = self.entropy_parameters(
            torch.cat([ctx_params_non_anchor, params], dim=1)
        )
        scales_non_anchor, means_non_anchor = gaussian_params_non_anchor.chunk(2, 1)
        scales_hat, means_hat = self.merge(scales_anchor, means_anchor, 
            scales_non_anchor, means_non_anchor)
        _, y_likelihoods = self.gaussian_conditional(y, scales=scales_hat, means=means_hat)
        y_ste = ste_round(y-means_hat)+means_hat

        return y_ste, y_likelihoods, z_likelihoods

    def merge(self, scales_anchor, means_anchor, scales_non_anchor, means_non_anchor, mask_type="A"):
        scales_hat = scales_anchor.clone()
        means_hat = means_anchor.clone()
        if mask_type == "A":
            scales_hat[:, :, 0::2, 0::2] = scales_non_anchor[:, :, 0::2, 0::2]
            scales_hat[:, :, 1::2, 1::2] = scales_non_anchor[:, :, 1::2, 1::2]
            means_hat[:, :, 0::2, 0::2] = means_non_anchor[:, :, 0::2, 0::2]
            means_hat[:, :, 1::2, 1::2] = means_non_anchor[:, :, 1::2, 1::2]
        else:
            scales_hat[:, :, 0::2, 1::2] = scales_non_anchor[:, :, 0::2, 1::2]
            scales_hat[:, :, 1::2, 0::2] = scales_non_anchor[:, :, 1::2, 0::2]
            means_hat[:, :, 0::2, 1::2] = means_non_anchor[:, :, 0::2, 1::2]
            means_hat[:, :, 1::2, 0::2] = means_non_anchor[:, :, 1::2, 0::2]
        return scales_hat, means_hat

    def get_anchor(self, y_hat, mask_type="A"):
        y_anchor = y_hat.clone()
        if mask_type == "A":
            y_anchor[:, :, 0::2, 0::2] = 0
            y_anchor[:, :, 1::2, 1::2] = 0
        else:
            y_anchor[:, :, 0::2, 1::2] = 0
            y_anchor[:, :, 1::2, 0::2] = 0
        return y_anchor

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        update_registered_buffers(
            self.hyperprior.entropy_bottleneck,
            "hyperprior.entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= self.hyperprior.entropy_bottleneck.update(force=force)
        return updated
 

class Multi_MSE_Loss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""
    def __init__(self, device):
        super().__init__()
        self.mse = nn.MSELoss().to(device)
        #self.num_camera = num_camera
        #self.lmbda = lmbda

    def forward(self, output, target, lmbda):
        #target1, target2 = target[0], target[1]
        num_camera = len(target)
        N, _, H, W = target[0].size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = 0
        out["mse_loss"] = 0
        out["psnr"] = 0

        # 计算误差
        for i in range(num_camera):
            out['bpp'+str(i)] = sum(
                (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                for likelihoods in output['likelihoods'][i].values())
            out["mse"+str(i)] = self.mse(output['x_hat'][i], target[i])
            out["bpp_loss"] += out['bpp'+str(i)]/num_camera
            out['mse_loss'] += lmbda * out["mse"+str(i)] /num_camera
            if out["mse"+str(i)] > 0:
                out["psnr"+str(i)] = 10 * (torch.log10(1 / out["mse"+str(i)])).mean()
            else:
                out["psnr"+str(i)] = 0
            out["psnr"] += out["psnr"+str(i)]/num_camera
        
        out['loss'] = out['mse_loss'] + out['bpp_loss']
        return out

class Multi_MS_SSIM_Loss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""
    def __init__(self, device, size_average=True, max_val=1):
        super().__init__()
        self.ms_ssim = MS_SSIM(size_average, max_val).to(device)
        #self.num_camera = num_camera
        #self.lmbda = lmbda

    def forward(self, output, target, lmbda):
        #target1, target2 = target[0], target[1]
        num_camera = len(target)
        N, _, H, W = target[0].size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = 0
        out["ms_ssim_loss"] = 0
        out["ms_db"] = 0

        # 计算误差
        for i in range(num_camera):
            out['bpp'+str(i)] = sum(
                (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                for likelihoods in output['likelihoods'][i].values())
            out["ms_ssim"+str(i)] = 1 - self.ms_ssim(output['x_hat'][i], target[i])
            out["bpp_loss"] += out['bpp'+str(i)]/num_camera
            out['ms_ssim_loss'] += lmbda * out["ms_ssim"+str(i)] /num_camera
            if out["ms_ssim"+str(i)] > 0:
                out["ms_db"+str(i)] = 10 * (torch.log10(1 / out["ms_ssim"+str(i)])).mean()
            else:
                out["ms_db"+str(i)] = 0
            out["ms_db"] += out["ms_db"+str(i)]/num_camera
        
        out['loss'] = out['ms_ssim_loss'] + out['bpp_loss']
        return out

class MSE_Loss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        #self.lmbda = lmbda

    def forward(self, output, target, lmbda):
        target1, target2 = target[0], target[1]
        N, _, H, W = target1.size()
        out = {}
        num_pixels = N * H * W

        # 计算误差
        out['bpp0'] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output['likelihoods'][0].values())
        out['bpp1'] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output['likelihoods'][1].values())        
        out["bpp_loss"] = (out['bpp0'] + out['bpp1'])/2
        out["mse0"] = self.mse(output['x_hat'][0], target1)
        out["mse1"] = self.mse(output['x_hat'][1], target2)
        
        if isinstance(lmbda, list):
            out['mse_loss'] = (lmbda[0] * out["mse0"] + lmbda[1] * out["mse1"])/2 
        else:
            out['mse_loss'] = lmbda * (out["mse0"] + out["mse1"])/2        #end to end
        out['loss'] = out['mse_loss'] + out['bpp_loss']

        return out

class MS_SSIM_Loss(nn.Module):
    def __init__(self, device, size_average=True, max_val=1):
        super().__init__()
        self.ms_ssim = MS_SSIM(size_average, max_val).to(device)
        #self.lmbda = lmbda

    def forward(self, output, target, lmbda):
        target1, target2 = target[0], target[1]
        N, _, H, W = target1.size()
        out = {}
        num_pixels = N * H * W

        # 计算误差
        out['bpp0'] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output['likelihoods'][0].values())
        out['bpp1'] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output['likelihoods'][1].values())        
        out["bpp_loss"] = (out['bpp0'] + out['bpp1'])/2

        out["ms_ssim0"] = 1 - self.ms_ssim(output['x_hat'][0], target1)
        out["ms_ssim1"] = 1- self.ms_ssim(output['x_hat'][1], target2)
 
        out['ms_ssim_loss'] = (out["ms_ssim0"] + out["ms_ssim1"])/2        #end to end
        out['loss'] = lmbda * out['ms_ssim_loss'] + out['bpp_loss']
        return out


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, sigma, channel):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(
        _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(
        channel, 1, window_size, window_size).contiguous())
    return window

class MS_SSIM(nn.Module):
    def __init__(self, size_average=True, max_val=255, device_id=0):
        super(MS_SSIM, self).__init__()
        self.size_average = size_average
        self.channel = 3
        self.max_val = max_val
        self.device_id = device_id

    def _ssim(self, img1, img2):

        _, c, w, h = img1.size()
        window_size = min(w, h, 11)
        sigma = 1.5 * window_size / 11

        window = create_window(window_size, sigma, self.channel)
        if self.device_id != None:
            window = window.cuda(self.device_id)

        mu1 = F.conv2d(img1, window, padding=window_size //
                       2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=window_size //
                       2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(
            img1*img1, window, padding=window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(
            img2*img2, window, padding=window_size//2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size //
                           2, groups=self.channel) - mu1_mu2

        C1 = (0.01*self.max_val)**2
        C2 = (0.03*self.max_val)**2
        V1 = 2.0 * sigma12 + C2
        V2 = sigma1_sq + sigma2_sq + C2
        ssim_map = ((2*mu1_mu2 + C1)*V1)/((mu1_sq + mu2_sq + C1)*V2)
        mcs_map = V1 / V2
        if self.size_average:
            return ssim_map.mean(), mcs_map.mean()

    def ms_ssim(self, img1, img2, levels=5):

        weight = Variable(torch.Tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]))
        msssim=Variable(torch.Tensor(levels,))
        mcs=Variable(torch.Tensor(levels,))
        # if self.device_id != None:
        #     weight = weight.cuda(self.device_id)
        #     weight = msssim.cuda(self.device_id)
        #     weight = mcs.cuda(self.device_id)
        #     print(weight.device)

        for i in range(levels):
            ssim_map, mcs_map=self._ssim(img1, img2)
            msssim[i]=ssim_map
            mcs[i]=mcs_map
            filtered_im1=F.avg_pool2d(img1, kernel_size=2, stride=2)
            filtered_im2=F.avg_pool2d(img2, kernel_size=2, stride=2)
            img1=filtered_im1
            img2=filtered_im2

        value=(torch.prod(mcs[0:levels-1]**weight[0:levels-1]) *
                                    (msssim[levels-1]**weight[levels-1]))
        return value


    def forward(self, img1, img2, levels=5):
        return self.ms_ssim(img1, img2, levels)



