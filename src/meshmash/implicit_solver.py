import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from sksparse.cholmod import cholesky
from tqdm import tqdm

from meshmash import cotangent_laplacian


class HeatSolver:
    def __init__(
        self,
        timescales: np.ndarray,
        robust=True,
        factorization="cholesky",
        dtype=np.float32,
    ):
        self.timescales = timescales
        self.robust = robust
        self.factorization = factorization
        self.dtype = dtype

    def fit(self, mesh):
        self.mesh_ = mesh

        L, M = cotangent_laplacian(mesh, robust=self.robust)

        self.laplacian_ = L.astype(self.dtype)
        self.mass_matrix_ = M.astype(self.dtype)

    def _get_rhs(self, initial_nodes):
        initial = np.zeros((self.laplacian_.shape[0], len(initial_nodes)))
        initial[initial_nodes, np.arange(len(initial_nodes))] = 1
        # TODO not sure if this is correct
        initial = self.mass_matrix_ @ initial
        return initial

    def _solve(self, mat, rhs):
        if self.factorization == "splu":
            solver = splu(mat)
            soln = solver.solve(rhs)
        elif self.factorization == "cholesky":
            # TODO rewrite to reuse the solver's permutation ordering, save time
            solver = cholesky(csc_matrix(mat).astype(self.dtype))
            soln = solver.solve_A(rhs)
        else:
            raise NotImplementedError("Invalid factorization method")
        return soln

    def _get_matrix_for_timescale(self, timescale):
        M = self.mass_matrix_.tocsc().astype(self.dtype)
        L = self.laplacian_.tocsc().astype(self.dtype)
        mat = (M + timescale * L).tocsc().astype(self.dtype)
        return mat

    def solve(self, initial_nodes):
        initial = self._get_rhs(initial_nodes)

        solutions = []
        for timescale in tqdm(self.timescales):
            mat = self._get_matrix_for_timescale(timescale)
            soln = self._solve(mat, initial)
            solutions.append(soln)

        solutions = np.array(solutions).swapaxes(0, 1)
        return solutions

    def solve_hks(self, initial_nodes):
        initial = self._get_rhs(initial_nodes)

        all_hks = []
        for timescale in tqdm(self.timescales):
            mat = self._get_matrix_for_timescale(timescale)
            soln = self._solve(mat, initial)
            hks = soln[initial_nodes, np.arange(len(initial_nodes))]
            all_hks.append(hks)

        hks = np.array(all_hks).T
        return hks
