from __future__ import annotations

from typing import Annotated

import typer

from meshmash.datasets import _REGISTRY, fetch_sample_mesh

_AVAILABLE = [k.removesuffix(".ply") for k in _REGISTRY]


def datasets(
    name: Annotated[
        str,
        typer.Argument(
            help=(
                "Dataset name to fetch (without file extension), "
                f"or 'all' to fetch every available dataset. "
                f"Available: {', '.join(_AVAILABLE)}."
            )
        ),
    ] = "all",
):
    if name == "all":
        names = _AVAILABLE
    else:
        if name not in _AVAILABLE:
            raise typer.BadParameter(
                f"Unknown dataset {name!r}. Available: {', '.join(_AVAILABLE)}."
            )
        names = [name]

    for n in names:
        typer.echo(f"Fetching '{n}'...")
        fetch_sample_mesh(n)
        typer.echo("  cached.")
