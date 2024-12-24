
import copy
from Evolution.CrossoverCellCgpW import CrossoverCellCgpW

class CrossoverCellCgpWX(CrossoverCellCgpW):
    """
    A class to perform crossover operations for Cartesian Genetic Programming (CGP) cells.
    Inherits from CrossoverCellCgpW.
    """

    def _do(self, problem, X, **kwargs):
        """
        Perform the crossover operation on the given population.

        Parameters
        ----------
        problem : pymoo.core.problem.Problem
            The problem to solve.
        X : numpy.ndarray
            The population to perform crossover on.
        **kwargs : dict
            Additional arguments, including the current generation number.

        Returns
        -------
        numpy.ndarray
            The population after crossover.
        """
        X_temp = copy.deepcopy(X)

        n_gen = kwargs['algorithm'].n_gen

        if n_gen <= 10:
            X_temp = self.croosover_base(X_temp, problem, 0)
        elif 10 < n_gen <= 20:
            X_temp = self.croosover_base(X_temp, problem, 1)
        elif 20 < n_gen <= 30:
            X_temp = self.croosover_base(X_temp, problem, 2)

        return X_temp

