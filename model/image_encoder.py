import torch
from torch import nn
from torchvision.models import resnet
from torchvision import transforms
from typing import Tuple, Callable
from torchvision.transforms import v2


def from_name(name: str) -> Tuple[nn.Module, int, Callable]:
    """
    :param name: Model name (UNI, resnet50, resnet18)
    :return: (image encoder, encoding dimension, transform)
    """
    if name == "UNI":
        import timm
        # pretrained=True needed to load UNI weights (and download weights for the first time)
        # init_values need to be passed in to successfully load LayerScale parameters (e.g. - block.0.ls1.gamma)
        model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)

        # Default transform includes resize and centrecrop, but patch size is controlled elsewhere
        #  and all inputs are square, so just keep the normalize
        # transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        transform = transforms.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250])

        model.eval()
        return model, 1024, transform

    elif name.startswith("kaiko"):
        preprocessing = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(size=224),
                v2.CenterCrop(size=224),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5),
                ),
            ]
        )
        model_name = name.split("-")[1]
        assert model_name in ["vits16", "vits8", "vitb16", "vitb8", "vitl14"], f"Unknown Kaiko-ai ViT: {model_name}"
        if model_name.startswith("vits"):
            dim = 384
        elif model_name.startswith("vitb"):
            dim = 768
        else:
            dim = 1024
        model = torch.hub.load("kaiko-ai/towards_large_pathology_fms", model_name, trust_repo=True)
        return model, dim, preprocessing

    elif name.startswith("resnet"):
        if name == "resnet50":
            model = resnet.resnet50(resnet.ResNet50_Weights.IMAGENET1K_V1)
        elif name == "resnet18":
            model = resnet.resnet18(resnet.ResNet18_Weights.IMAGENET1K_V1)
        else:
            raise NotImplementedError(f"Unknown resnet type: {name}")
        dim = model.fc.in_features
        model.fc = nn.Identity()  # do not perform classification layer
        return model, dim, nn.Identity()

    else:
        raise ValueError(f"Invalid patch encoder '{name}'.")