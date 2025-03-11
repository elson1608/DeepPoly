import argparse
import torch
import torch.nn as nn
from torch import optim
import math
from AbstractBox import *
from networks import get_network
from utils.loading import parse_spec
from skip_block import SkipBlock

DEVICE = "cpu"

def initialize_net(input_shape, net):
    params = []
    for layer in net:
        if isinstance(layer, nn.Flatten):
            input_shape = (math.prod(input_shape),)
        elif isinstance(layer, nn.Conv2d):
            layer.weight.requires_grad = False
            layer.bias.requires_grad = False
            input_shape = calc_out_shape(input_shape, layer.stride[0], layer.padding[0], layer.weight)
        elif isinstance(layer, nn.Linear):
            layer.weight.requires_grad = False
            layer.bias.requires_grad = False
            input_shape = (layer.out_features,)
        elif isinstance(layer, nn.ReLU):
            layer.alpha = torch.nn.Parameter(
                data=torch.randn(math.prod(input_shape)),
                requires_grad=True
            )
            params.append(layer.alpha)
        elif isinstance(layer, nn.ReLU6):
            layer.alpha1 = torch.nn.Parameter(
                data=torch.randn(math.prod(input_shape)),
                requires_grad=True
            )
            layer.alpha2 = layer.alpha = torch.nn.Parameter(
                data=torch.randn(math.prod(input_shape)),
                requires_grad=True
            )
            layer.alpha3 = layer.alpha = torch.nn.Parameter(
                data=torch.randn(math.prod(input_shape)),
                requires_grad=True
            )
            layer.alpha4 = layer.alpha = torch.nn.Parameter(
                data=torch.randn(math.prod(input_shape)),
                requires_grad=True
            )
            params.append(layer.alpha1)
            params.append(layer.alpha2)
        elif isinstance(layer, SkipBlock):
            input_shape, skip_params = initialize_net(input_shape, layer.path)
            params += skip_params
    return input_shape, params

def analyze(net: torch.nn.Module, inputs: torch.Tensor, eps: float, true_label: int) -> bool:
    """
    Analyzes the given network with the given input and epsilon.
    :param net: Network to analyze.
    :param inputs: Input to analyze.
    :param eps: Epsilon to analyze.
    :param true_label: True label of the input.
    :return: True if the network is verified, False otherwise.
    """
    input_shape, params = initialize_net(inputs.shape, net)

    # Add final layer
    last_layer = nn.Linear(input_shape[0], input_shape[0] - 1, None)
    weight = torch.eye(input_shape[0])
    weight[:, true_label] = -1
    weight = np.delete(weight, true_label, axis=0)
    with torch.no_grad():
        last_layer.weight.copy_(weight)
    net.add_module("last layer", last_layer)

    # print(net)


    init_box = AbstractBox.construct_initial_box(inputs, eps)
    box = propagate(net, init_box,true_label)
    if not params:
        return (box.ub < 0).all()
    optimizer = optim.AdamW(params, lr=1.5)
    while True:
        optimizer.zero_grad()
        box = propagate(net, init_box,true_label)
        assert (box.lb > box.ub).sum() == 0
        loss = torch.clamp(box.ub, min=0).sum().log()
        loss.backward()
        optimizer.step()
        if (box.ub < 0).all():
            return True

def propagate(model, box, true_label) -> AbstractBox:
    # print(model)
    for layer in model:
        if isinstance(layer, nn.Flatten):
            box.lb = box.lb.flatten()
            box.ub = box.ub.flatten()
            box.init_lb = box.init_lb.flatten()
            box.init_ub = box.init_ub.flatten()
        elif isinstance(layer, nn.Conv2d):
            box = box.propagate_conv(layer)
        elif isinstance(layer, nn.Linear):
            box = box.propagate_linear(layer)
        elif isinstance(layer, nn.ReLU):
            box = box.propagate_relu(layer)
        elif isinstance(layer, nn.ReLU6):
            box = box.propagate_relu6(layer)
        elif isinstance(layer, SkipBlock):
            box = box.propagate_skip(layer)
        else:
            raise NotImplementedError(f'Unsupported layer type: {type(layer)}')
    return box


def main():
    parser = argparse.ArgumentParser(
        description="Neural network verification using DeepPoly relaxation."
    )
    parser.add_argument(
        "--net",
        type=str,
        choices=[
            "fc_linear",
            "fc_base",
            "fc_w",
            "fc_d",
            "fc_dw",
            "fc6_base",
            "fc6_w",
            "fc6_d",
            "fc6_dw",
            "conv_linear",
            "conv_base",
            "conv6_base",
            "conv_d",
            "skip",
            "skip_large",
            "skip6",
            "skip6_large",
        ],
        required=True,
        help="Neural network architecture which is supposed to be verified.",
    )
    parser.add_argument("--spec", type=str, required=True, help="Test case to verify.")
    args = parser.parse_args()

    true_label, dataset, image, eps = parse_spec(args.spec)

    # print(args.spec)

    if dataset == "mnist":
        in_ch, in_dim, num_class = 1, 28, 10
    elif dataset == "cifar10":
        in_ch, in_dim, num_class = 3, 32, 10
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    net = get_network(
        args.net,
        in_ch=in_ch,
        in_dim=in_dim,
        num_class=num_class,
        weight_path=f"models/{dataset}_{args.net}.pt",
    ).to(DEVICE)

    image = image.to(DEVICE)
    out = net(image.unsqueeze(0))

    pred_label = out.max(dim=1)[1].item()
    assert pred_label == true_label

    if analyze(net, image, eps, true_label):
        print("verified")
    else:
        print("not verified")


if __name__ == "__main__":
    main()
