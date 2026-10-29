import torch


def coordinates(world, dim):
    """
    Create data for spatial x positions of each cell
    """
    metadata = world["metadata"]
    N = torch.round(metadata["length"] / metadata["resolution"]).long()
    device = N.device

    if dim == "x":
        xs = (
            torch.arange(N[0], device=device) * metadata["resolution"][0]
            + metadata["origin"][0]
        )
        return torch.tile(xs.view(-1, 1), (1, N[1]))
    elif dim == "y":
        ys = (
            torch.arange(N[1], device=device) * metadata["resolution"][1]
            + metadata["origin"][1]
        )
        return torch.tile(ys.view(1, -1), (N[0], 1))
    else:
        print("expected arg dim to be \{x, y\}")
        exit(1)
