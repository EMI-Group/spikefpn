import torch
import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval


thresh = 0.3 # neuronal threshold
lens = 0.5   # hyper-parameters of approximate function
decay = 0.2  # decay constants
b_delta = 1


class ActFun_changeable(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, b):
        ctx.save_for_backward(input)
        ctx.b = b
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        original_bp = False
        if original_bp:
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            temp = abs(input - thresh) < lens
        else:
            input, = ctx.saved_tensors
            device = input.device
            grad_input = grad_output.clone()
            b = torch.tensor(ctx.b,device=device)
            temp = (1-torch.tanh(b*(input-thresh))**2)*b/2/(torch.tanh(b/2))
            temp[input<=0]=0
            temp[input>=1]=0
        return grad_input * temp.float(), None


class ActFun_lsnn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, b, v_th):
        ctx.save_for_backward(input)
        ctx.b = b
        ctx.v_th = v_th
        return input.gt(v_th).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        device = input.device
        grad_input = grad_output.clone()
        b = torch.tensor(ctx.b,device=device)
        v_th = ctx.v_th.clone().detach().requires_grad_(False)
        temp = (1-torch.tanh(b*(input-v_th))**2)*b/2/(torch.tanh(b/2))
        temp[input<=0]=0
        temp[input>=1]=0
        return grad_input * temp.float(), None,  - grad_input * temp.float()


class SNN_2d_lsnn_front(nn.Module):
    def __init__(self, input_c, output_c, kernel_size=3, stride=1, padding=1, b=3, dilation=1, act="spike"):
        super(SNN_2d_lsnn_front, self).__init__()
        
        self.conv1 = nn.Conv2d(input_c, output_c, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.act = act
        self.act_fun = ActFun_lsnn().apply
        self.b = b
        self.bn = nn.BatchNorm2d(output_c)
        self.relu = nn.LeakyReLU()
        
        self.beta = 0.07
        self.rho = nn.Parameter(0.0435*torch.ones(output_c,1,1)).requires_grad_(True)
        
    def forward(self, all_input):
        B, S, C, H, W = all_input.shape
        all_x = []

        for t in range(S):
            input = all_input[:,t]
            if not self.bn.training:
                conv_bn = fuse_conv_bn_eval(self.conv1,self.bn)
                mem_this = conv_bn(input)
            else:
                mem_this = self.bn(self.conv1(input))

            device = input.device
            if t == 0:
                self.mem = torch.zeros_like(mem_this, device=device)
                self.a = torch.zeros_like(self.mem, device=device)
                self.rho.data.clamp_(0.032,0.055)
            
            A = thresh + self.beta*self.a
            self.mem = self.mem + mem_this
            spike = self.act_fun(self.mem, self.b, A)
            self.mem = self.mem * decay * (1. - spike) 
            self.a = torch.exp(-0.05/self.rho)*self.a + spike

            all_x.append(spike.unsqueeze(1))
        all_x = torch.cat(all_x, 1)
        all_x = all_x.reshape(B, 2, 5, H, W)
        return all_x


class SNN_2d_lsnn(nn.Module):
    def __init__(self, input_c, output_c, kernel_size=3, stride=1, padding=1, b=3, dilation=1, act="spike"):
        super(SNN_2d_lsnn, self).__init__()

        self.conv1 = nn.Conv2d(input_c, output_c, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.act = act
        self.act_fun = ActFun_lsnn().apply
        self.b = b
        self.mem = None
        self.bn = nn.BatchNorm2d(output_c)
        self.relu = nn.LeakyReLU()
        
        self.a = None
        self.thresh = 0.3
        self.beta = 0.07
        self.rho = nn.Parameter(0.87 * torch.ones(output_c, 1, 1)).requires_grad_(True)
        self.sparsity = []
    
    def fuse_conv_bn(self):
        self.conv1 = fuse_conv_bn_eval(self.conv1,self.bn)
        
    def forward(self, input, param):
        if not self.bn.training:
            conv_bn = fuse_conv_bn_eval(self.conv1,self.bn)
            mem_this = conv_bn(input)
        else:
            mem_this = self.bn(self.conv1(input))

        if param["mixed_at_mem"]:
            return mem_this
        
        device = input.device
        if param["is_first"] or self.mem == None:
            self.mem = torch.zeros_like(mem_this, device=device)
            self.a = torch.zeros_like(self.mem, device=device)
            self.rho.data.clamp_(0.64, 1.1)
        A = self.thresh + self.beta * self.a # ALIF
        self.mem = self.mem + mem_this
        spike = self.act_fun(self.mem, self.b, A)
        self.mem = self.mem * decay * (1. - spike) 
        self.a = torch.exp(-1 / self.rho) * self.a - spike
        return spike

    def clear_sparsity(self):
        self.sparsity = []


class ActFun_thresh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, b, v_th):
        ctx.save_for_backward(input)
        ctx.b = b
        ctx.v_th = v_th
        return input.gt(v_th).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        device = input.device
        grad_input = grad_output.clone()
        b = torch.tensor(ctx.b, device=device)
        v_th = torch.tensor(ctx.v_th, device=device)
        temp = (1 - torch.tanh(b * (input-v_th))**2) * b / 2 / (torch.tanh(b / 2))
        temp[input <= 0] = 0
        temp[input >= 1] = 0
        return grad_input * temp.float(), None,  None


class SNN_2d_thresh(nn.Module):
    def __init__(self, input_c, output_c, kernel_size=3, stride=1, padding=1, b=3, dilation=1, act="spike", thresh=0.3):
        super(SNN_2d_thresh, self).__init__()
        
        self.conv1 = nn.Conv2d(input_c, output_c, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.act = act
        self.act_fun = ActFun_thresh().apply
        self.b = b
        self.mem = None
        self.bn = nn.BatchNorm2d(output_c)
        self.thresh = thresh
        self.sparsity = []
    
    def fuse_conv_bn(self):
        self.conv1 = fuse_conv_bn_eval(self.conv1,self.bn)

    def forward(self, input, param):
        if not self.bn.training:
            mem_this = self.conv1(input)
        else:
            mem_this = self.bn(self.conv1(input))
        if param["mixed_at_mem"]:
            if self.act == "spike":
                return mem_this
            else:
                return self.relu(mem_this)
        
        device = input.device
        if param["is_first"]:
            self.mem = torch.zeros_like(mem_this, device=device)
        self.mem = self.mem + mem_this
        spike = self.act_fun(self.mem, self.b, self.thresh)
        self.mem = self.mem * decay * (1. - spike)
        return spike
    
    def clear_sparsity(self):
        self.sparsity = []


class SNN_2d(nn.Module):
    def __init__(self, input_c, output_c, kernel_size=3, stride=1, padding=1, b=3, dilation=1, act="spike"):
        super(SNN_2d, self).__init__()
        
        self.conv1 = nn.Conv2d(input_c, output_c, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.act = act
        self.act_fun = ActFun_changeable().apply
        self.b = b
        self.mem = None
        self.bn = nn.BatchNorm2d(output_c)
        self.relu = nn.LeakyReLU()
        self.sparsity = []
    
    def fuse_conv_bn(self):
        self.conv1 = fuse_conv_bn_eval(self.conv1,self.bn)

    def forward(self, input, param):
        if not self.bn.training:
            conv_bn = fuse_conv_bn_eval(self.conv1,self.bn)
            mem_this = conv_bn(input)
        else:
            mem_this = self.bn(self.conv1(input))
        if param["mixed_at_mem"]:
            if self.act == "spike":
                return mem_this
            else:
                return self.relu(mem_this)
        
        device = input.device
        if param["is_first"] or self.mem == None:
            self.mem = torch.zeros_like(mem_this, device=device)
        self.mem = self.mem + mem_this
        spike = self.act_fun(self.mem, self.b)
        self.mem = self.mem * decay * (1. - spike)
        return spike
    
    def clear_sparsity(self):
        self.sparsity = []


class Mem_Relu(nn.Module):
    def __init__(self, input_c, output_c, kernel_size=3, stride=1, padding=1, b=3, dilation=1, act="spike"):
        super(Mem_Relu, self).__init__()
        
        self.conv1 = nn.Conv2d(input_c, output_c, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.act = act
        self.act_fun = ActFun_changeable().apply
        self.b = b
        self.mem = None
        self.bn = nn.BatchNorm2d(output_c)
        self.relu = nn.ReLU()
        self.decay = nn.Parameter(0.5 * torch.ones(output_c, 1, 1)).requires_grad_(True)
        self.sparsity = []
    
    def fuse_conv_bn(self):
        self.conv1 = fuse_conv_bn_eval(self.conv1,self.bn)

    def forward(self, input, param):
        if not self.bn.training:
            conv_bn = fuse_conv_bn_eval(self.conv1,self.bn)
            mem_this = conv_bn(input)
        else:
            mem_this = self.bn(self.conv1(input))
        device = input.device
        if param["is_first"] or self.mem == None:
            self.mem = torch.zeros_like(mem_this, device=device)
        self.decay.data.clamp_(0,1)
        self.mem = self.mem * self.decay + mem_this 
        return self.relu(self.mem)
    
    def clear_sparsity(self):
        self.sparsity = []
