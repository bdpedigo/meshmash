import typer

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
