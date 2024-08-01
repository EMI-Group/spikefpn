import torch
import torch.nn as nn
import torch.nn.functional as F
from .operations import *
from .decoding_formulas import network_layer_to_space

decay = 0.2


class MixedOp(nn.Module):
    def __init__(self):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()


    def forward(self, x, weights, left_or_right):
        opt_outs = []
        for i in range(3):
            opt_out = self._ops[i](x, left_or_right)
            opt_out = weights[i] * opt_out
            opt_outs.append(opt_out)
        return sum(opt_outs)  


class Cell(nn.Module):
    def __init__(self, steps, block_multiplier, prev_prev_fmultiplier,
                 prev_filter_multiplier, cell_arch, network_arch,
                 filter_multiplier, downup_sample, args=None):
        super(Cell, self).__init__()
        self.cell_arch = cell_arch

        self.C_in = block_multiplier * filter_multiplier
        self.C_out = filter_multiplier
        self.C_prev = int(block_multiplier * prev_filter_multiplier)
        self.C_prev_prev = int(block_multiplier * prev_prev_fmultiplier)
        self.downup_sample = downup_sample
        self._steps = steps
        self.block_multiplier = block_multiplier
        self._ops = nn.ModuleList()
        self.sparsity = []
        if downup_sample == -1:
            self.scale = 0.5
        elif downup_sample == 1:
            self.scale = 2
        self.cell_arch = torch.sort(self.cell_arch,dim=0)[0].to(torch.uint8)
        for x in self.cell_arch:
            primitive = PRIMITIVES[x[1]]
            if x[0] in [0, 2, 5]:
                op = OPS[primitive](self.C_prev_prev, self.C_out, stride=1, signal=1)
            elif x[0] in [1, 3, 6]:
                op = OPS[primitive](self.C_prev, self.C_out, stride=1, signal=1)
            else:
                op = OPS[primitive](self.C_out, self.C_out, stride=1, signal=1)

            self._ops.append(op)

        self.mem = None
        self.act_fun = ActFun_changeable().apply

    def scale_dimension(self, dim, scale):
        return (int((float(dim) - 1.0) * scale + 1.0) if dim % 2 == 1 else int((float(dim) * scale)))

    def forward(self, prev_prev_input, prev_input, param):
        s0 = prev_prev_input
        s1 = prev_input
        if self.downup_sample != 0:
            feature_size_h = self.scale_dimension(s1.shape[2], self.scale)
            feature_size_w = self.scale_dimension(s1.shape[3], self.scale)
            s1 = F.interpolate(s1, [feature_size_h, feature_size_w], mode="nearest")
        if (s0.shape[2] != s1.shape[2]) or (s0.shape[3] != s1.shape[3]):
            s0 = F.interpolate(s0, (s1.shape[2], s1.shape[3]), mode="nearest")

        device = prev_input.device

        states = [s0, s1]
        offset = 0
        ops_index = 0
        for i in range(self._steps):
            new_states = []
            for j, h in enumerate(states):
                branch_index = offset + j
                if branch_index in self.cell_arch[:, 0]:
                    if prev_prev_input is None and j == 0:
                        ops_index += 1
                        continue
                    if isinstance(self._ops[ops_index],Identity):
                        new_state = self._ops[ops_index](h)
                    else:
                        param["mixed_at_mem"] = True
                        new_state = self._ops[ops_index](h, param)
                        if self.mem == None:
                            self.mem = [torch.zeros_like(new_state,device=device)] * self._steps
                        param["mixed_at_mem"] = False

                    new_states.append(new_state)
                    ops_index += 1

            s = sum(new_states)
            # lif
            self.mem[i] = self.mem[i] + s
            spike = self.act_fun(self.mem[i],3)
            self.mem[i] = self.mem[i] * decay * (1. - spike)

            offset += len(states)
            states.append(spike)

        concat_feature = torch.cat(states[-self.block_multiplier:], dim=1) 
        
        return prev_input, concat_feature
    
    def clear_sparsity(self):
        self.sparsity = []


class newFeature(nn.Module):
    def __init__(self, frame_rate, network_path, cell_arch, cell=Cell, args=None):
        super(newFeature, self).__init__()
        self.args = args
        self.cells = nn.ModuleList()
        self.out_index = [-1, -1, -1]
        for i in range(len(network_path) - 1, -1, -1):
            if network_path[i] == 1 and self.out_index[0] == -1:
                self.out_index[0] = i
                continue
            if network_path[i] == 2 and self.out_index[1] == -1:
                self.out_index[1] = i
                continue
            if network_path[i] == 3 and self.out_index[2] == -1:
                self.out_index[2] = i
                continue
        self.stride = []
        if self.out_index[0] != -1:
            self.stride.append(8)
        if self.out_index[1] != -1:
            self.stride.append(16)
        if self.out_index[2] != -1:
            self.stride.append(32)
        network_arch = network_layer_to_space(network_path)
        self.network_arch = torch.from_numpy(network_arch)
        self.cell_arch = torch.from_numpy(cell_arch)
        self._step = 3
        self._num_layers = 10
        self._block_multiplier = 3
        self._filter_multiplier = 32

        f_initial = int(self._filter_multiplier)
        half_f_initial = int(f_initial/2)
        self._num_end = self._filter_multiplier*self._block_multiplier

        self.stem0 = SNN_2d_lsnn(frame_rate, half_f_initial * self._block_multiplier, kernel_size=5, stride=2, padding=2, b=3)
        self.stem1 = SNN_2d(half_f_initial * self._block_multiplier, f_initial * self._block_multiplier, kernel_size=3, stride=2, padding=1, b=3)
        self.left_vpre_feature =  [None] * 2
        self.right_vpre_feature = [None] * 2

        filter_param_dict = {0: 1, 1: 2, 2: 4, 3: 8}

        for i in range(self._num_layers):
            level_option = torch.sum(self.network_arch[i], dim=1)
            prev_level_option = torch.sum(self.network_arch[i - 1], dim=1)
            prev_prev_level_option = torch.sum(self.network_arch[i - 2], dim=1)
            level = torch.argmax(level_option).item()
            prev_level = torch.argmax(prev_level_option).item()
            prev_prev_level = torch.argmax(prev_prev_level_option).item()

            if i == 0:
                downup_sample = - torch.argmax(torch.sum(self.network_arch[0], dim=1))
                _cell = cell(
                    self._step, self._block_multiplier, self._filter_multiplier / 2,
                    self._filter_multiplier,
                    self.cell_arch, self.network_arch[i],
                    int(self._filter_multiplier * filter_param_dict[level]),
                    downup_sample, self.args
                )
            else:
                three_branch_options = torch.sum(self.network_arch[i], dim=0)
                downup_sample = torch.argmax(three_branch_options).item() - 1
                if i == 1:
                    _cell = cell(
                        self._step, self._block_multiplier,
                        self._filter_multiplier,
                        int(self._filter_multiplier * filter_param_dict[prev_level]),
                        self.cell_arch, self.network_arch[i],
                        int(self._filter_multiplier * filter_param_dict[level]),
                        downup_sample, self.args
                    )
                else:
                    _cell = cell(
                        self._step, self._block_multiplier,
                        int(self._filter_multiplier * filter_param_dict[prev_prev_level]),
                        int(self._filter_multiplier * filter_param_dict[prev_level]),
                        self.cell_arch, self.network_arch[i],
                        int(self._filter_multiplier * filter_param_dict[level]), downup_sample, self.args
                    )

            self.cells += [_cell]

        num_out = len(self.stride)
        if num_out == 1:
            self.connect_3 = SNN_2d(self._num_end * 2, self._num_end * 1, 1, 1, 0)
            self.smooth3 = SNN_2d(self._num_end * 1, self._num_end * 2, 3, 1, 1, dilation=1)
        elif num_out == 2:
            self.connect_2 = SNN_2d(self._num_end * 4, self._num_end * 2, 1, 1, 0)
            self.connect_3 = SNN_2d(self._num_end * 3, self._num_end * 1, 1, 1, 0)
            self.up_conv2 = SNN_2d(self._num_end * 2, self._num_end * 1, 1, 1, 0)

            self.smooth2 = SNN_2d(self._num_end * 2, self._num_end * 4, 3, 1, 1, dilation=1)
            self.smooth3 = SNN_2d(self._num_end * 1, self._num_end * 2, 3, 1, 1, dilation=1)
        elif num_out == 3:
            self.connect_1 = SNN_2d(self._num_end * 8, self._num_end * 4, 1, 1, 0)
            self.connect_2 = SNN_2d(self._num_end * 6, self._num_end * 2, 1, 1, 0)
            self.connect_3 = SNN_2d(self._num_end * 3, self._num_end * 1, 1, 1, 0)

            self.up_conv1 = SNN_2d(self._num_end * 4, self._num_end * 2, 1, 1, 0)
            self.up_conv2 = SNN_2d(self._num_end * 2, self._num_end * 1, 1, 1, 0)

            self.smooth1 = SNN_2d(self._num_end * 4, self._num_end * 8, 3, 1, 1, dilation=1)
            self.smooth2 = SNN_2d(self._num_end * 2, self._num_end * 4, 3, 1, 1, dilation=1)
            self.smooth3 = SNN_2d(self._num_end * 1, self._num_end * 2, 3, 1, 1, dilation=1)
        self.stem0_fmap = None


    def forward(self, x, param=None):
        if param == None:
            param = {"is_first":True}
        param["mixed_at_mem"] = False
        stem0 = self.stem0(x, param)
        self.stem0_fmap = stem0.detach().cpu().numpy()
        stem1 = self.stem1(stem0, param)
        out = (stem0, stem1)
        for i in range(self._num_layers):
            out = self.cells[i](out[0], out[1], param)
            if i == self.out_index[0]:
                last_output_3 = out
            elif i == self.out_index[1]:
                last_output_2 = out
            elif i == self.out_index[2]:
                last_output_1 = out
        upsample_1 = nn.Upsample(scale_factor=2, mode="nearest", align_corners=None)
        upsample_2 = nn.Upsample(scale_factor=2, mode="nearest", align_corners=None)
        num_out = len(self.stride)
        if num_out == 1:
            p3 = self.connect_3(last_output_3[-1], param)
            return self.smooth3(p3, param)
        elif num_out == 2:
            p2 = self.connect_2(last_output_2[-1], param)
            p2_up = upsample_2(self.up_conv2(p2, param))
            
            p3 = self.connect_3(torch.cat([last_output_3[-1], p2_up], dim=1), param)

            return self.smooth2(p2, param), self.smooth3(p3, param)
        else:
            p1 = self.connect_1(last_output_1[-1], param)
            p1_up = upsample_1(self.up_conv1(p1, param))

            p2 = self.connect_2(torch.cat([last_output_2[-1], p1_up], dim=1), param)
            p2_up = upsample_2(self.up_conv2(p2, param))
            
            p3 = self.connect_3(torch.cat([last_output_3[-1], p2_up], dim=1), param)

            return self.smooth1(p1, param), self.smooth2(p2, param), self.smooth3(p3, param)

    def get_params(self):
        bn_params = []
        non_bn_params = []
        for name, param in self.named_parameters():
            if "bn" in name or "downsample.1" in name:
                bn_params.append(param)
            else:
                bn_params.append(param)
        return bn_params, non_bn_params

