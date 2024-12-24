import copy

from Evolution.MutationCellCgpW import MutationCellCgpW


class MutationCellCgpWX(MutationCellCgpW):
    """
    A class to perform mutation operations on a population of individuals using different mutation strategies
    based on the generation number.

    Inherits from:
        MutationCellCgpW: Base class for mutation operations.

    Methods:
        _do(problem, X, **kwargs):
            Execute the mutation operation on the given population based on the generation number.
    """
    def _do(self, problem, X, **kwargs):
        """
        Execute the mutation operation on the given population based on the generation number.

        Args:
            problem (Problem): Problem definition, providing variable bounds.
            X (ndarray): Population to mutate.
            **kwargs: Additional arguments, including the algorithm state.

        Returns:
            ndarray: Updated population after mutation.
        """
        X_temp = copy.deepcopy(X)

        if kwargs['algorithm'].n_gen <= 10:
            X_temp = self.mutation_base(X_temp, problem, 0)

        if 10 < kwargs['algorithm'].n_gen <= 20:
            X_temp = self.mutation_base(X_temp, problem, 1)

        if 20 < kwargs['algorithm'].n_gen <= 30:
            X_temp = self.mutation_base(X_temp, problem, 2)

        return X_temp