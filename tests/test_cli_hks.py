import meshio
import typer.main
from click.testing import CliRunner

from meshmash.cli import app

runner = CliRunner()
_cli = typer.main.get_command(app)

_FAST_PARAMS = [
    "--n-components", "4",
    "--max-eigenvalue", "1e-8",
    "--simplify-target-reduction", "0.7",
    "--no-auxiliary-features",
    "--n-jobs", "1",
]


def test_hks_cli_runs(mesh, tmp_path):
    vertices, faces = mesh
    mesh_path = tmp_path / "test_mesh.ply"
    meshio.write(str(mesh_path), meshio.Mesh(points=vertices, cells=[("triangle", faces)]))

    output_path = tmp_path / "output.npz"
    result = runner.invoke(
        _cli,
        ["hks", str(mesh_path), "--output", str(output_path)] + _FAST_PARAMS,
    )

    assert result.exit_code == 0, result.output
    assert output_path.exists()


def test_hks_cli_missing_file(tmp_path):
    output_path = tmp_path / "output.npz"
    result = runner.invoke(
        _cli,
        ["hks", str(tmp_path / "nonexistent.ply"), "--output", str(output_path)],
    )
    assert result.exit_code != 0
