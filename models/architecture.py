from builtins import isinstance
import numpy as np
import torch
import torch.nn as nn

from .backbone import newFeature
from .operations import *


class ConvLTC(nn.Module):
    """
    A more general discrete form of LTC model
    """
    def __init__(self, in_channels, out_channels, tau_input=True, taum_ini=[0.5, 0.8], usetaum=True, stream_opt=True, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = self._make_layer(in_channels, out_channels, kernel_size, padding, stride)
        self.usetaum = usetaum
        self.stream_opt = stream_opt
        self.cm = nn.Parameter(0.1 * torch.randn(out_channels, 1, 1) + 1.0)
        self.vleak = nn.Parameter(0.1 * torch.randn(out_channels, 1, 1) + 1.0)
        if self.usetaum:
            self.tau_m = nn.Parameter((taum_ini[0]-taum_ini[1]) * torch.rand(out_channels, 1, 1) + taum_ini[1])
        else:
            self.gleak = nn.Parameter((taum_ini[0]-taum_ini[1]) * torch.rand(out_channels, 1, 1) + taum_ini[1])

        self.E_revin = nn.Parameter(0.1 * torch.randn(out_channels, 1, 1) + 1.0) # mean=1.0, std=0.1

        self._epsilon = 1e-8

        self.sigmoid = nn.Sigmoid()
        self.tau_input = tau_input
        self.tanh = nn.Tanh()
        self.debug = None

        nn.init.xavier_normal_(self.conv[0].weight.data)

    def _make_layer(self, in_channels, out_channels, kernel_size, padding, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels))

    def _clip(self, w):
        return torch.nn.ReLU()(w)

    def apply_weight_constraints(self):
        self.cm.data.clamp_(0,1000)
        if self.usetaum:
            self.tau_m.data.clamp_(0,2000)
        else:
            self.gleak.data.clamp_(0,1000)

    def forward(self, inputs):
        """
        :param inputs: (B, C_in, S, H, W)
        :param hidden_state: (hx: (B, C, S, H, W), cx: (B, C, S, H, W))
        :return: (B, C_out, H, W)
        """
        B, S, C, H, W = inputs.size()

        outputs = []
        cm_t = self.cm
        v_pre = torch.zeros(B, self.out_channels, H, W).to(inputs.device)
        for t in range(S):
            wih = self.conv(inputs[:, t, ...])

            if self.tau_input:
                if self.usetaum:
                    numerator = self.tau_m * v_pre / (self.vleak + self.cm * self.sigmoid(wih)) + wih*self.E_revin
                    denominator = 1
                else:
                    numerator = cm_t * v_pre + self.gleak * self.vleak + wih*self.E_revin
                    denominator = cm_t + self.gleak + wih

            else:
                if self.usetaum:
                    numerator = self.tau_m * v_pre + wih
                    denominator = 1

                else:
                    numerator = cm_t * v_pre + self.gleak * self.vleak + wih
                    denominator = cm_t + self.gleak

            v_pre = numerator / (denominator + self._epsilon)
            v_pre = self.sigmoid(v_pre)

            outputs.append(v_pre)
        self.debug = outputs[-1]
        if self.stream_opt:
            return torch.cat(outputs, 1).reshape(B, S, C, H, W)
        else:
            return outputs[-1]


class SpikeFPN_NCARS(nn.Module):
    def __init__(
            self, 
            device, 
            input_size = None, 
            num_classes = 20, 
            conf_thresh = 0.01, 
            nms_thresh = 0.5, 
            cfg = None, 
            init_channels = 5, 
            time_steps = 5, 
            args = None
        ):
        super(SpikeFPN_NCARS, self).__init__()
        self.device = device
        self.input_size = input_size
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh 
        self.time_steps = time_steps
        self.file = ""
        network_path_fea = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3]
        network_path_fea = np.array(network_path_fea)
        cell_arch_fea = np.array([
            [1, 1],
            [0, 1],
            [3, 2],
            [2, 1],
            [7, 1],
            [8, 1]
        ])
        self.encoder = ConvLTC(init_channels, init_channels)
        self.feature = newFeature(init_channels, network_path_fea, cell_arch_fea, args=args)
        self.stride = self.feature.stride
        num_out = len(self.stride)
        anchor_size = cfg[f"anchor_size_gen1_{num_out * 3}"]
        self.anchor_list = anchor_size
        self.anchor_size = torch.tensor(anchor_size).reshape(len(self.stride), len(anchor_size) // len(self.stride), 2).float()
        self.num_anchors = self.anchor_size.size(1)
        self.grid_xy, self.anchor_wh = self.create_grid(input_size)
        out_channel = 2 * 3 * 32

        # Prediction
        num_out = len(self.stride)
        if num_out == 1:
            self.head_det_3 = nn.Conv2d(out_channel, self.num_anchors * (1 + self.num_classes + 4), kernel_size=1)
        elif num_out == 2:
            self.head_det_2 = nn.Conv2d(out_channel * 2, self.num_anchors * (1 + self.num_classes + 4), kernel_size=1)
            self.head_det_3 = nn.Conv2d(out_channel, self.num_anchors * (1 + self.num_classes + 4), kernel_size=1)
        else:
            self.head_det_1 = nn.Conv2d(out_channel * 4, self.num_anchors * (1 + self.num_classes + 4), kernel_size=1)
            self.head_det_2 = nn.Conv2d(out_channel * 2, self.num_anchors * (1 + self.num_classes + 4), kernel_size=1)
            self.head_det_3 = nn.Conv2d(out_channel, self.num_anchors * (1 + self.num_classes + 4), kernel_size=1)

        self.lsnn = SNN_2d_lsnn_front(1, 1, kernel_size=3, stride=1, padding=1,b=3)

        # Classification merge
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(output_size=(2, 2))
        self.linear = nn.Linear(in_features=252, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def create_grid(self, input_size):
        total_grid_xy = []
        total_anchor_wh = []
        w, h = input_size, input_size
        for ind, s in enumerate(self.stride):
            # Generate grid cells
            fmp_w, fmp_h = w // s, h // s
            grid_y, grid_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
            # [H, W, 2] -> [HW, 2]
            grid_xy = torch.stack([grid_x, grid_y], dim=-1).float().view(-1, 2)
            # [HW, 2] -> [1, HW, 1, 2]   
            grid_xy = grid_xy[None, :, None, :].to(self.device)
            # [1, HW, 1, 2]
            anchor_wh = self.anchor_size[ind].repeat(fmp_h*fmp_w, 1, 1).unsqueeze(0).to(self.device)

            total_grid_xy.append(grid_xy)
            total_anchor_wh.append(anchor_wh)

        return total_grid_xy, total_anchor_wh

    def set_grid(self, input_size):
        self.input_size = input_size
        self.grid_xy, self.anchor_wh = self.create_grid(input_size)
    
    def forward(self, x):
        self.clear_mem()
        param = {"mixed_at_mem": True, "left_or_right": "left", "is_first": False}

        for t in range(self.time_steps):
            inputs = x[:, t, ...]
            if t == 0:
                param["is_first"] = True
            else:
                param["is_first"] = False
            y = self.feature(inputs, param)
        num_out = len(self.stride)
        if num_out == 1:
            y3 = y
            pred_s = self.head_det_3(y3)
            preds = [pred_s]
        elif num_out == 2:
            y2, y3 = y
            pred_m = self.head_det_2(y2)
            pred_s = self.head_det_3(y3)
            preds = [pred_s, pred_m]
        else:
            y1, y2, y3 = y
            pred_l = self.head_det_1(y1)
            pred_m = self.head_det_2(y2)
            pred_s = self.head_det_3(y3)
            preds = [pred_s, pred_m, pred_l]

        # Binary classification merge
        y = []
        for pred in preds:
            y.append(self.adaptive_avg_pool(pred))
        y = torch.concat(y, dim=1)
        y = y.view(y.shape[0], -1)
        y = self.linear(y)
        y = self.sigmoid(y)
        return y

    def set_mem_keys(self, mem_keys):
        self.mem_keys = mem_keys
    
    def clear_mem(self):
        for key in self.mem_keys:
            exec(f"self.{key:s}.mem=None")
        for m in self.modules():
            if isinstance(m, SNN_2d) or isinstance(m, SNN_2d_lsnn) or isinstance(m, SNN_2d_thresh) or isinstance(m, Mem_Relu):
                m.mem = None


class SpikeFPN_GAD(nn.Module):
    def __init__(
            self, 
            device, 
            input_size = None, 
            num_classes = 20, 
            conf_thresh = 0.01, 
            nms_thresh = 0.5, 
            cfg = None, 
            center_sample = False, 
            init_channels = 5, 
            time_steps = 5, 
            args = None
        ):
        super(SpikeFPN_GAD, self).__init__()
        self.device = device
        self.input_size = input_size
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh 
        self.center_sample = center_sample
        self.time_steps = time_steps
        self.file = ""
        network_path_fea = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3]
        network_path_fea = np.array(network_path_fea)
        cell_arch_fea = np.array([
            [1, 1],
            [0, 1],
            [3, 2],
            [2, 1],
            [7, 1],
            [8, 1]
        ])
        self.encoder = ConvLTC(init_channels, init_channels)
        self.feature = newFeature(init_channels, network_path_fea, cell_arch_fea, args=args)
        self.stride = self.feature.stride
        num_out = len(self.stride)
        anchor_size = cfg[f"anchor_size_gen1_{num_out * 3}"]
        self.anchor_list = anchor_size
        self.anchor_size = torch.tensor(anchor_size).reshape(len(self.stride), len(anchor_size) // len(self.stride), 2).float()
        self.num_anchors = self.anchor_size.size(1)
        self.grid_xy, self.anchor_wh = self.create_grid(input_size)
        out_channel = 2 * 3 * 32

        # Prediction
        num_out = len(self.stride)
        if num_out == 1:
            self.head_det_3 = nn.Conv2d(out_channel, self.num_anchors * (1 + self.num_classes + 4), kernel_size=1)
        elif num_out == 2:
            self.head_det_2 = nn.Conv2d(out_channel * 2, self.num_anchors * (1 + self.num_classes + 4), kernel_size=1)
            self.head_det_3 = nn.Conv2d(out_channel, self.num_anchors * (1 + self.num_classes + 4), kernel_size=1)
        else:
            self.head_det_1 = nn.Conv2d(out_channel * 4, self.num_anchors * (1 + self.num_classes + 4), kernel_size=1)
            self.head_det_2 = nn.Conv2d(out_channel * 2, self.num_anchors * (1 + self.num_classes + 4), kernel_size=1)
            self.head_det_3 = nn.Conv2d(out_channel, self.num_anchors * (1 + self.num_classes + 4), kernel_size=1)

        self.lsnn = SNN_2d_lsnn_front(1, 1, kernel_size=3, stride=1, padding=1,b=3)

    def create_grid(self, input_size):
        total_grid_xy = []
        total_anchor_wh = []
        w, h = input_size, input_size
        for ind, s in enumerate(self.stride):
            # Generate grid cells
            fmp_w, fmp_h = w // s, h // s
            grid_y, grid_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
            # [H, W, 2] -> [HW, 2]
            grid_xy = torch.stack([grid_x, grid_y], dim=-1).float().view(-1, 2)
            # [HW, 2] -> [1, HW, 1, 2]   
            grid_xy = grid_xy[None, :, None, :].to(self.device)
            # [1, HW, 1, 2]
            anchor_wh = self.anchor_size[ind].repeat(fmp_h*fmp_w, 1, 1).unsqueeze(0).to(self.device)

            total_grid_xy.append(grid_xy)
            total_anchor_wh.append(anchor_wh)

        return total_grid_xy, total_anchor_wh

    def set_grid(self, input_size):
        self.input_size = input_size
        self.grid_xy, self.anchor_wh = self.create_grid(input_size)

    def decode_xywh(self, txtytwth_pred):
        """
        Input:
            txtytwth_pred: [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
        Output:
            xywh_pred: [B, H*W*anchor_n, 4] containing [xmin, ymin, xmax, ymax]
        """
        B, HW, ab_n, _ = txtytwth_pred.size()
        xy_pred = torch.sigmoid(txtytwth_pred[:, :, :, :2]) + self.grid_xy
        wh_pred = torch.exp(txtytwth_pred[:, :, :, 2:]) * self.anchor_wh
        # [H * W, anchor_n, 4] -> [H * W * anchor_n, 4]
        xywh_pred = torch.cat([xy_pred, wh_pred], -1).view(B, HW * ab_n, 4) * self.stride

        return xywh_pred

    def decode_boxes(self, txtytwth_pred, requires_grad=False):
        """
        Input:
            txtytwth_pred: [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
        Output:
            x1y1x2y2_pred: [B, H*W*anchor_n, 4] containing [xmin, ymin, xmax, ymax]
        """
        # [H * W * anchor_n, 4]
        xywh_pred = self.decode_xywh(txtytwth_pred)

        # [center_x, center_y, w, h] -> [xmin, ymin, xmax, ymax]
        x1y1x2y2_pred = torch.zeros_like(xywh_pred, requires_grad=requires_grad)
        x1y1x2y2_pred[:, :, 0] = (xywh_pred[:, :, 0] - xywh_pred[:, :, 2] / 2)
        x1y1x2y2_pred[:, :, 1] = (xywh_pred[:, :, 1] - xywh_pred[:, :, 3] / 2)
        x1y1x2y2_pred[:, :, 2] = (xywh_pred[:, :, 0] + xywh_pred[:, :, 2] / 2)
        x1y1x2y2_pred[:, :, 3] = (xywh_pred[:, :, 1] + xywh_pred[:, :, 3] / 2)

        return x1y1x2y2_pred
    
    def decode_bbox(self, reg_pred, index):
        """
        reg_pred: [B, N, KA, 4]
        """
        B = reg_pred.size(0)
        # txty -> cxcy
        if self.center_sample:
            xy_pred = (reg_pred[..., :2].sigmoid() * 2.0 - 1.0 + self.grid_xy[index]) * self.stride[index]
        else:
            xy_pred = (reg_pred[..., :2].sigmoid() + self.grid_xy[index]) * self.stride[index]
        # twth -> wh
        wh_pred = reg_pred[..., 2:].exp() * self.anchor_wh[index]
        xywh_pred = torch.cat([xy_pred, wh_pred], dim=-1).view(B, -1, 4)
        # xywh -> x1y1x2y2
        x1y1_pred = xywh_pred[..., :2] - xywh_pred[..., 2:] / 2
        x2y2_pred = xywh_pred[..., :2] + xywh_pred[..., 2:] / 2
        box_pred = torch.cat([x1y1_pred, x2y2_pred], dim=-1)

        return box_pred

    def forward(self, x):
        self.clear_mem()
        C = self.num_classes
        B, T, c, H, W = x.shape
        param = {"mixed_at_mem": True, "left_or_right": "left", "is_first": False}

        for t in range(self.time_steps):
            inputs = x[:, t, ...]
            if t == 0:
                param["is_first"] = True
            else:
                param["is_first"] = False
            y = self.feature(inputs, param)
        num_out = len(self.stride)
        if num_out == 1:
            y3 = y
            pred_s = self.head_det_3(y3)
            preds = [pred_s]
        elif num_out == 2:
            y2, y3 = y
            pred_m = self.head_det_2(y2)
            pred_s = self.head_det_3(y3)
            preds = [pred_s, pred_m]
        else:
            y1, y2, y3 = y
            pred_l = self.head_det_1(y1)
            pred_m = self.head_det_2(y2)
            pred_s = self.head_det_3(y3)
            preds = [pred_s, pred_m, pred_l]

        obj_pred_list = []
        cls_pred_list = []
        reg_pred_list = []
        box_pred_list = []

        for i, pred in enumerate(preds):
            # [B, num_anchors*(1 + C + 4), H, W] -> [B, num_anchors, H, W] -> [B, H, W, num_anchors] ->  [B, HW*num_anchors, 1]
            obj_pred_i = pred[:, :self.num_anchors, :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, 1)
            # [B, num_anchors*(1 + C + 4), H, W] -> [B, num_anchors*C, H, W] -> [B, H, W, num_anchors*C] -> [B, H*W*num_anchors, C]
            cls_pred_i = pred[:, self.num_anchors:self.num_anchors*(1+C), :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, C)
            # [B, num_anchors*(1 + C + 4), H, W] -> [B, num_anchors*4, H, W] -> [B, H, W, num_anchors*4] -> [B, HW, num_anchors, 4]
            reg_pred_i = pred[:, self.num_anchors*(1+C):, :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_anchors, 4)
            box_pred_i = self.decode_bbox(reg_pred_i, i) / self.input_size

            obj_pred_list.append(obj_pred_i)
            cls_pred_list.append(cls_pred_i)
            reg_pred_list.append(reg_pred_i)
            box_pred_list.append(box_pred_i)
        
        obj_pred = torch.cat(obj_pred_list, dim=1)
        cls_pred = torch.cat(cls_pred_list, dim=1)
        reg_pred = torch.cat(reg_pred_list, dim=1)
        box_pred = torch.cat(box_pred_list, dim=1)
        
        return obj_pred, cls_pred, reg_pred, box_pred

    def set_mem_keys(self, mem_keys):
        self.mem_keys = mem_keys
    
    def clear_mem(self):
        for key in self.mem_keys:
            exec(f"self.{key:s}.mem=None")
        for m in self.modules():
            if isinstance(m, SNN_2d) or isinstance(m, SNN_2d_lsnn) or isinstance(m, SNN_2d_thresh) or isinstance(m, Mem_Relu):
                m.mem = None
