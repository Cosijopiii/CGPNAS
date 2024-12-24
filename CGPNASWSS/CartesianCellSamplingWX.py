import numpy as np
from CGPNASWSS.CellCGPWX import CellCgpWX
from Evolution.CartesianCellSamplingW import CartesianCellGeneticProgrammingW

class CartesianCellGeneticProgrammingWX(CartesianCellGeneticProgrammingW):
    def _do(self, problem, n_samples, **kwargs):
        """
        Generates a sample population for the given problem.

        Parameters
        ----------
        problem : pymoo.core.problem.Problem
            The problem to solve.
        n_samples : int
            Number of samples to generate.
        **kwargs : dict
            Additional arguments.

        Returns
        -------
        numpy.ndarray
            An array of generated samples.
        """
        X = np.full((n_samples, 1), None, dtype=object)
        for i in range(n_samples):
            gen = CellCgpWX(self.conf_net, self.Mpool, self.N, self.R)
            X[i, 0] = gen
        return X

