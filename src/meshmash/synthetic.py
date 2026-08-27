"""Synthetic meshes with known geometry, for validating shape descriptors.

A descriptor tested only on real meshes is tested against no answer. These
build surfaces of revolution from an explicit radius profile, which covers the
shapes a tubular-structure descriptor has to get right -- a tube, a tube that
closes, a tube that narrows and reopens, a tube with a bulb on a stalk -- and
for each of them what the answer should be is fixed by the profile rather
than by a second implementation of the descriptor.
"""

import numpy as np

from .types import Mesh

__all__ = ["revolve_profile", "tube_profile", "hemisphere_profile"]


def revolve_profile(z: np.ndarray, radius: np.ndarray, n_theta: int = 40) -> Mesh:
    """Revolve a radius profile around the z axis into a triangle mesh.

    Parameters
    ----------
    z :
        Axial positions of the profile samples, shape ``(N,)``.  Need not be
        evenly spaced.
    radius :
        Radius at each axial position, shape ``(N,)``.  A zero closes the
        surface into a pole, so a profile that starts or ends at zero comes
        back capped rather than open.
    n_theta :
        Number of vertices per ring.

    Returns
    -------
    :
        ``(vertices, faces)``, float64 vertices and int32 faces.  Rings are
        ordered along ``z`` and, within a ring, by angle.

    Notes
    -----
    A zero radius may appear only at the ends of the profile.  In the middle
    it would pinch the surface into two components joined at a point, which
    is not a manifold and not a shape anything here wants.
    """
    z = np.asarray(z, dtype=np.float64)
    radius = np.asarray(radius, dtype=np.float64)
    if z.shape != radius.shape or z.ndim != 1 or len(z) < 2:
        raise ValueError(
            f"z and radius must be matching 1-D arrays of length >= 2, got "
            f"{z.shape} and {radius.shape}"
        )
    if (radius < 0).any():
        raise ValueError("radius must be non-negative")
    if (radius[1:-1] == 0).any():
        raise ValueError(
            "a zero radius inside the profile would pinch the surface to a "
            "point; zeros belong at the ends, where they cap it"
        )

    angles = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    circle = np.column_stack([np.cos(angles), np.sin(angles)])

    vertices: list[np.ndarray] = []
    rings: list[np.ndarray] = []
    for level, level_radius in zip(z, radius):
        start = sum(len(ring) for ring in rings)
        if level_radius == 0:
            vertices.append(np.array([[0.0, 0.0, level]]))
            rings.append(np.array([start]))
        else:
            ring = np.column_stack([circle * level_radius, np.full(n_theta, level)])
            vertices.append(ring)
            rings.append(np.arange(start, start + n_theta))

    faces: list[np.ndarray] = []
    following = np.roll(np.arange(n_theta), -1)
    for lower, upper in zip(rings[:-1], rings[1:]):
        if len(lower) == 1:
            # A fan from the pole, wound the same way round the axis as the
            # bands below, so the whole surface stays consistently oriented.
            faces.append(
                np.column_stack([np.repeat(lower, n_theta), upper[following], upper])
            )
        elif len(upper) == 1:
            faces.append(
                np.column_stack([lower, lower[following], np.repeat(upper, n_theta)])
            )
        else:
            faces.append(np.column_stack([lower, lower[following], upper]))
            faces.append(np.column_stack([lower[following], upper[following], upper]))

    return (
        np.concatenate(vertices, axis=0),
        np.concatenate(faces, axis=0).astype(np.int32),
    )


def tube_profile(
    length: float, radius: float, spacing: float
) -> tuple[np.ndarray, np.ndarray]:
    """A straight tube's profile, sampled at roughly ``spacing`` along z.

    Parameters
    ----------
    length :
        Axial extent, centred on zero.
    radius :
        Constant radius.
    spacing :
        Target axial spacing between rings.

    Returns
    -------
    :
        ``(z, radius)``, ready for [revolve_profile][meshmash.synthetic.revolve_profile].
    """
    n = max(int(round(length / spacing)) + 1, 2)
    z = np.linspace(-length / 2, length / 2, n)
    return z, np.full(n, float(radius))


def hemisphere_profile(
    radius: float, z0: float, spacing: float, opening_up: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """A hemispherical cap's profile, closing a tube of the same radius.

    Sampled by polar angle rather than by z, so the samples stay evenly spaced
    along the surface as the profile turns over -- an even-in-z sampling would
    put almost nothing near the pole, which is exactly where the curvature the
    descriptor is being asked about lives.

    Parameters
    ----------
    radius :
        Cap radius, which is also the radius of the tube it closes.
    z0 :
        Axial position of the cap's rim.
    spacing :
        Target arc-length spacing between rings.
    opening_up :
        If ``True`` the cap closes upward, its pole at ``z0 + radius``;
        otherwise downward.

    Returns
    -------
    :
        ``(z, radius)`` in increasing ``z`` order either way, so profiles
        concatenate directly.
    """
    n = max(int(round((np.pi / 2) * radius / spacing)) + 1, 2)
    polar = np.linspace(0.0, np.pi / 2, n)
    along = radius * np.sin(polar)
    across = radius * np.cos(polar)
    if opening_up:
        return z0 + along, across
    return (z0 - along)[::-1], across[::-1]
