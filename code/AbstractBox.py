import torch
import torch.nn as nn
from typing import Dict
import numpy as np
import torch.nn.functional as F

from skip_block import SkipBlock


def calc_out_shape(input_shape, stride, padding, kernel):
    in_channels, in_height, in_width = input_shape
    out_channels, _, kernel_h, kernel_w = kernel.shape

    # Calculate output dimensions
    out_height = ((in_height + 2 * padding - kernel_h) // stride) + 1
    out_width = ((in_width + 2 * padding - kernel_w) // stride) + 1

    return out_channels, out_height, out_width


def convmtx2(kernel, input_shape, stride=1, padding=0):
    output_shape = calc_out_shape(input_shape, stride, padding, kernel)
    identity = torch.eye(np.prod(input_shape).item()).reshape([-1] + list(input_shape))
    output = F.conv2d(identity, kernel, None, stride, padding)
    w = output.reshape(-1, np.prod(output_shape).item()).T
    return w, output_shape



class AbstractBox:
    def __init__(
            self,
            init_lb: torch.Tensor,
            init_ub: torch.Tensor,
            lb: torch.Tensor = None,
            ub: torch.Tensor = None,
            weight_lb: Dict[int, torch.Tensor] = None,
            weight_ub: Dict[int, torch.Tensor] = None,
            bias_lb: torch.Tensor = None,
            bias_ub: torch.Tensor = None,
            predecessor: 'AbstractBox' = None
    ):
        if lb is not None and ub is not None:
            assert lb.shape == ub.shape
            assert (lb > ub).sum() == 0
        self.lb = lb
        self.ub = ub
        self.init_lb = init_lb
        self.init_ub = init_ub
        self.weight_lb = weight_lb
        self.weight_ub = weight_ub
        self.bias_lb = bias_lb
        self.bias_ub = bias_ub
        self.predecessor = predecessor

    @staticmethod
    def construct_initial_box(x: torch.Tensor, eps: float) -> 'AbstractBox':
        init_lb = torch.clamp(x - eps, 0., 1.)
        init_ub = torch.clamp(x + eps, 0., 1.)

        return AbstractBox(
            lb=init_lb,
            ub=init_ub,
            init_lb=init_lb,
            init_ub=init_ub,
            weight_lb=None,
            weight_ub=None,
            bias_lb=None,
            bias_ub=None
        )

    def backsubstitute(self):
        current = self
        curr_weight_ub = current.weight_ub.copy()
        curr_weight_lb = current.weight_lb.copy()
        curr_bias_ub = current.bias_ub
        curr_bias_lb = current.bias_lb

        while current.predecessor is not None:
            predecessor = current.predecessor
            prev_weight_ub = predecessor.weight_ub
            prev_weight_lb = predecessor.weight_lb
            prev_bias_ub = predecessor.bias_ub
            prev_bias_lb = predecessor.bias_lb

            curr_deepest_ub = curr_weight_ub.pop(max(curr_weight_ub.keys()))
            curr_deepest_lb = curr_weight_lb.pop(max(curr_weight_lb.keys()))

            curr_bias_ub = torch.clamp(curr_deepest_ub, min=0.) @ prev_bias_ub + \
                           torch.clamp(curr_deepest_ub, max=0.) @ prev_bias_lb + \
                            + curr_bias_ub

            curr_bias_lb = torch.clamp(curr_deepest_lb, min=0.) @ prev_bias_lb + \
                           torch.clamp(curr_deepest_lb, max=0.) @ prev_bias_ub + \
                            + curr_bias_lb

            for layer in prev_weight_ub.keys():


                new_ub = torch.clamp(curr_deepest_ub, min=0) @ prev_weight_ub[layer] + \
                         torch.clamp(curr_deepest_ub, max=0) @ prev_weight_lb[layer]
                new_lb = torch.clamp(curr_deepest_lb, min=0) @ prev_weight_lb[layer] + \
                         torch.clamp(curr_deepest_lb, max=0) @ prev_weight_ub[layer]
                if layer in curr_weight_ub:
                    curr_weight_ub[layer] += new_ub
                else:
                    curr_weight_ub[layer] = new_ub

                if layer in curr_weight_lb:
                    curr_weight_lb[layer] += new_lb
                else:
                    curr_weight_lb[layer] = new_lb

            current = current.predecessor


        return curr_weight_ub[1], curr_bias_ub, curr_weight_lb[1], curr_bias_lb

    def propagate_linear(self, fc: nn.Linear) -> 'AbstractBox':
        if self.weight_ub is None:
            # Separate the positive and negative weights
            w_pos = torch.clamp(fc.weight.detach(), min=0)  # Positive weights
            w_neg = torch.clamp(fc.weight.detach(), max=0)  # Negative weights

            # Vectorized operations for the bounds
            res_lb = w_pos @ self.lb + w_neg @ self.ub
            res_ub = w_pos @ self.ub + w_neg @ self.lb

            if fc.bias is not None:
                res_lb += fc.bias.detach()
                res_ub += fc.bias.detach()

            return AbstractBox(
                lb=res_lb,
                ub=res_ub,
                init_lb=self.init_lb,
                init_ub=self.init_ub,
                weight_lb={1: fc.weight.detach()},
                weight_ub={1: fc.weight.detach()},
                bias_lb=fc.bias if fc.bias is not None else torch.zeros((fc.out_features, 1)),
                bias_ub=fc.bias if fc.bias is not None else torch.zeros((fc.out_features, 1)),
                predecessor=None
            )
        else:
            res = AbstractBox(
                self.init_lb,
                self.init_ub,
                lb=None,
                ub=None,
                weight_lb={max(self.weight_lb.keys()) + 1: fc.weight},
                weight_ub={max(self.weight_ub.keys()) + 1: fc.weight},
                bias_lb=fc.bias.detach() if fc.bias is not None else torch.zeros((fc.out_features,)),
                bias_ub=fc.bias.detach() if fc.bias is not None else torch.zeros((fc.out_features,)),
                predecessor=self
            )

            weight_ub, bias_ub, weight_lb, bias_lb = res.backsubstitute()

            w_pos_lb = torch.clamp(weight_lb, min=0)  # Positive weights
            w_pos_ub = torch.clamp(weight_ub, min=0)  # Positive weights
            w_neg_lb = torch.clamp(weight_lb, max=0)  # Negative weights
            w_neg_ub = torch.clamp(weight_ub, max=0)  # Negative weights

            res.lb = w_pos_lb@res.init_lb + w_neg_lb@res.init_ub + bias_lb
            res.ub = w_pos_ub@res.init_ub + w_neg_ub@res.init_lb + bias_ub

            assert (res.lb > res.ub).sum() == 0

            return res

    def propagate_conv(self, conv: nn.Conv2d) -> 'AbstractBox':
        w, output_shape = convmtx2(
            conv.weight.detach(),
            self.lb.shape,
            stride=conv.stride[0],
            padding=conv.padding[0]
        )
        out_channels, out_height, out_width = output_shape
        if self.weight_ub is None:
            w_pos = torch.clamp(w, min=0)
            w_neg = torch.clamp(w, max=0)
            res_ub = w_pos@self.ub.reshape(-1) + w_neg@self.lb.reshape(-1)
            res_lb = w_pos@self.lb.reshape(-1) + w_neg@self.ub.reshape(-1)

            if conv.bias is not None:
                bias = conv.bias.detach().view(-1, 1).repeat(1, out_height * out_width).flatten()
                res_lb += bias
                res_ub += bias
            else:
                bias = torch.zeros_like(res_lb)

            return AbstractBox(
                lb=res_lb.reshape(output_shape),
                ub=res_ub.reshape(output_shape),
                init_lb=self.lb.reshape(-1),
                init_ub=self.ub.reshape(-1),
                weight_lb={1: w},
                weight_ub={1: w},
                bias_lb=bias,
                bias_ub=bias,
                predecessor=None
            )
        else:
            res = AbstractBox(
                self.init_lb,
                self.init_ub,
                lb=None,
                ub=None,
                weight_lb={max(self.weight_lb.keys()) + 1: w},
                weight_ub={max(self.weight_ub.keys()) + 1: w},
                bias_lb=conv.bias.detach().view(-1, 1).repeat(1, out_height * out_width).flatten(),
                bias_ub=conv.bias.detach().view(-1, 1).repeat(1, out_height * out_width).flatten(),
                predecessor=self
            )

            weight_ub, bias_ub, weight_lb, bias_lb = res.backsubstitute()

            w_pos_lb = torch.clamp(weight_lb, min=0)  # Positive weights
            w_pos_ub = torch.clamp(weight_ub, min=0)  # Positive weights
            w_neg_lb = torch.clamp(weight_lb, max=0)  # Negative weights
            w_neg_ub = torch.clamp(weight_ub, max=0)  # Negative weights

            res.lb = w_pos_lb @ res.init_lb + w_neg_lb @ res.init_ub + bias_lb
            res.ub = w_pos_ub @ res.init_ub + w_neg_ub @ res.init_lb + bias_ub
            res.lb = res.lb.reshape(output_shape)
            res.ub = res.ub.reshape(output_shape)

            assert (res.lb > res.ub).sum() == 0

            return res



    def propagate_relu6(self, relu6: nn.ReLU6) -> 'AbstractBox':
        lb = self.lb.reshape(-1)
        ub = self.ub.reshape(-1)

        diag_ub = torch.zeros_like(lb)
        diag_lb = torch.zeros_like(lb)
        new_bias_ub = torch.zeros_like(lb)
        new_bias_lb = torch.zeros_like(lb)

        # l > 6
        new_bias_ub += (lb > 6) * 6
        new_bias_lb += (lb > 6) * 6

        # Case 0 < lb < 6 && 0 < ub < 6
        identity_lb = torch.logical_and(lb > 0, lb < 6)
        identity_ub = torch.logical_and(ub > 0, ub < 6)
        identity = torch.logical_and(identity_lb, identity_ub)
        diag_ub += identity * torch.ones_like(lb)
        diag_lb += identity * torch.ones_like(lb)

        # l < 0 && 0 < u < 6
        slope = torch.where((ub - lb) != 0, ub / (ub - lb), torch.tensor(0.0))
        like_relu = torch.logical_and(lb < 0, torch.logical_and(0 < ub, ub < 6))
        diag_ub += like_relu * slope
        new_bias_ub -= like_relu * slope * lb
        diag_lb += like_relu * torch.sigmoid(relu6.alpha1)

        # 0 < l < 6 && u > 6
        slope = torch.where((ub - lb) != 0, (6-lb) / (ub - lb), torch.tensor(0.0))
        like_flipped_relu = torch.logical_and(ub > 6, torch.logical_and(0 < lb, lb < 6))
        diag_lb += like_flipped_relu * slope
        new_bias_lb += like_flipped_relu * (6-slope*ub)
        diag_ub += like_flipped_relu * torch.sigmoid(relu6.alpha2)
        new_bias_ub += like_flipped_relu * 6 * (1-torch.sigmoid(relu6.alpha2))

        # l < 0 && u > 6
        alpha1_ub = 6 / ub
        alpha2_ub = 6 / (6-lb)
        crossing = torch.logical_and(lb < 0, ub > 6)
        diag_lb += crossing * torch.sigmoid(relu6.alpha1) * alpha1_ub
        diag_ub += crossing * torch.sigmoid(relu6.alpha2) * alpha2_ub
        new_bias_ub += crossing * 6 * (1-torch.sigmoid(relu6.alpha2) * alpha2_ub)


        new_lb = relu6(self.lb)
        new_ub = relu6(self.ub)

        # strictly negative case doesn't contribute anything
        new_weight_lb = torch.diag(diag_lb.view(-1))
        new_weight_ub = torch.diag(diag_ub.view(-1))

        assert (new_lb > new_ub).sum() == 0

        return AbstractBox(
            lb=new_lb,
            ub=new_ub,
            init_lb=self.init_lb,
            init_ub=self.init_ub,
            weight_lb={max(self.weight_lb.keys()) + 1: new_weight_lb},
            weight_ub={max(self.weight_ub.keys()) + 1: new_weight_ub},
            bias_lb=new_bias_lb,
            bias_ub=new_bias_ub,
            predecessor=self
        )


    def propagate_skip(self, skip: SkipBlock) -> 'AbstractBox':
        init_box = self
        box = self
        for idx, layer in enumerate(skip.path):
            if isinstance(layer, nn.Linear):
                box = box.propagate_linear(layer)
            elif isinstance(layer, nn.ReLU):
                box = box.propagate_relu(layer)
            elif isinstance(layer, nn.ReLU6):
                box = box.propagate_relu6(layer)
            else:
                raise NotImplementedError(f'Unsupported layer type: {type(layer)}')

        box.ub += init_box.ub
        box.lb += init_box.lb
        box.weight_ub[max(self.weight_ub.keys()) + 1] = torch.eye(len(self.ub))
        box.weight_lb[max(self.weight_lb.keys()) + 1] = torch.eye(len(self.lb))

        return box


    def propagate_relu(self, relu: nn.ReLU) -> 'AbstractBox':
        lb = self.lb.reshape(-1)
        ub = self.ub.reshape(-1)
        diag_ub = torch.zeros_like(lb)
        diag_lb = torch.zeros_like(lb)
        new_bias_ub = torch.zeros_like(lb)

        slope = torch.where((ub - lb) != 0, ub / (ub - lb), torch.tensor(0.0))

        # Case l > 0:
        diag_ub += (lb > 0)
        diag_lb += (lb > 0)

        # Case l < 0 && u > 0
        crossing = torch.logical_and(lb < 0, ub > 0)

        # Upper Bound
        diag_ub += crossing*slope
        new_bias_ub -= crossing*slope*lb

        # Lower Bound
        diag_lb += crossing * torch.sigmoid(relu.alpha)

        # strictly negative case doesn't contribute anything
        new_weight_lb = torch.diag(diag_lb.view(-1))
        new_weight_ub = torch.diag(diag_ub.view(-1))

        new_lb = relu(self.lb)
        new_ub = relu(self.ub)

        assert (new_lb > new_ub).sum() == 0

        return AbstractBox(
            lb=new_lb,
            ub=new_ub,
            init_lb=self.init_lb,
            init_ub=self.init_ub,
            weight_lb={max(self.weight_lb.keys()) + 1: new_weight_lb},
            weight_ub={max(self.weight_ub.keys()) + 1: new_weight_ub},
            bias_lb=torch.zeros_like(new_bias_ub),
            bias_ub=new_bias_ub,
            predecessor=self
        )
