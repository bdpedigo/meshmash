import typer

from .datasets import datasets
from .hks import hks

app = typer.Typer(
    name="meshmash",
    help="Mesh analysis tools.",
    no_args_is_help=True,
)

app.command(
    name="hks",
    help="Compute condensed Heat Kernel Signature features on a mesh.",
    no_args_is_help=True,
)(hks)

app.command(
    name="datasets",
    help="Fetch and cache sample datasets.",
)(datasets)
