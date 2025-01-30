import torch
from torch import Tensor

from .proj import spherical,to_homogeneous
from typing import Optional

def find_fundamental3d(p1: Tensor, p2: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    x1, y1, z1 = torch.chunk(p1, dim=-1, chunks=3)  # Bx1xN
    x2, y2, z2 = torch.chunk(p2, dim=-1, chunks=3)  # Bx1xN

    X = torch.cat(
        [
            x1 * x2,
            x1 * y2,
            z1 * z2,
            y1 * x2,
            y1 * y2,
            y1 * z2,
            z1 * x2,
            z1 * y2,
            z1 * z2,
        ],
        dim=-1,
    )
    if mask is not None:
        X = X.transpose(-2, -1) @ mask.type_as(X) @ X
    else:
        X = X.transpose(-2, -1) @ X
    _, _, V = torch.linalg.svd(X)
    f: Tensor = V[..., -1].reshape(-1, 3, 3)
    return f


def find_fundamental_equirectangular(
    p1: Tensor, p2: Tensor, mask: Optional[Tensor] = None
) -> Tensor:
    p1 = spherical(p1, True)
    p2 = spherical(p2, True)
    return find_fundamental3d(p1, p2, mask)

def find_fundamental(p1: Tensor, p2: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    r"""Find the fundamental matrix from a set of point correspondences.

    Args:
        p1: The set of points seen from the first camera frame in the camera plane
        p2: The set of points seen from the second camera frame in the camera plane
        mask: The mask to filter out outliers with shape :math:`(B, N, 1)`.

    Returns:
        The fundamental matrix with shape :math:`(B, 3, 3)`.

    """
    p1 = to_homogeneous(p1)
    p2 = to_homogeneous(p2)
    return find_fundamental3d(p1, p2, mask)