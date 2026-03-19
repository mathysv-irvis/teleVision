MODEL_REGISTRY = {
    "net": "Net",
    "tinynet": "TinyNet",
}


def get_model(name):

    if name == "net":
        from .Net import Net
        return Net

    if name == "tinynet":
        from .TinyNet import TinyNet
        return TinyNet

    raise ValueError(name)
