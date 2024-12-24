import copy
import numpy as np
from pymoo.core.mutation import Mutation
from pymoo.operators.repair.to_bound import set_to_bounds_if_outside_by_problem
from Evolution.CellCGPW import CellCgpW
from Evolution.GenCGPW import GenCgpW


class MutationCellCgpW(Mutation):
    """
    A custom mutation operator for Cell-based Cartesian Genetic Programming with weights (CellCgpW).
    This operator applies a polynomial mutation technique tailored for evolutionary algorithms.

    Attributes:
        eta (float): Distribution index controlling the mutation intensity.
        prob (float): Probability of mutation for each variable. If not specified, it defaults to 1/number of variables.
    """

    def __init__(self, eta, prob=None):
        """
        Initialize the mutation operator.

        Args:
            eta (float): Distribution index controlling mutation intensity.
            prob (float, optional): Probability of mutation for each variable. Defaults to None, which sets it to 1/number of variables.
        """
        super().__init__()
        self.eta = float(eta)
        self.prob = float(prob) if prob is not None else None

    def mutation_base(self, X, problem, index):
        """
        Perform the base mutation operation on the given population.

        Args:
            X (ndarray): Population to mutate, with shape (number_of_matings, variables).
            problem (Problem): Problem definition, providing variable bounds.
            index (int): Index of the individual in the population to perform mutation on.

        Returns:
            ndarray: New population after mutation.
        """
        n_matings, n_var = X.shape
        X_copy = copy.deepcopy(X)
        X_t = np.zeros(shape=(n_matings, X[0, 0].n_var_size))

        # Convert individuals to real representation
        for i in range(n_matings):
            X[i, 0].individual[index].toReal()
            X_t[i, :] = X[i, 0].individual[index].realgene.flatten()
        X = X_t.astype(float)

        Y = np.full(X.shape, np.inf)
        if self.prob is None:
            self.prob_var = 1.0 / n_var
        else:
            self.prob_var = self.prob

        # Determine which variables will undergo mutation
        do_mutation = np.random.random(X.shape) < self.prob_var
        Y[:, :] = X

        # Problem bounds for mutation
        xl = np.repeat(problem.xl[None, :], X.shape[0], axis=0)[do_mutation]
        xu = np.repeat(problem.xu[None, :], X.shape[0], axis=0)[do_mutation]
        X = X[do_mutation]

        # Calculate mutation parameters
        delta1 = (X - xl) / (xu - xl)
        delta2 = (xu - X) / (xu - xl)
        mut_pow = 1.0 / (self.eta + 1.0)
        rand = np.random.random(X.shape)

        mask = rand <= 0.5
        mask_not = np.logical_not(mask)
        deltaq = np.zeros(X.shape)

        # Apply polynomial mutation
        xy = 1.0 - delta1
        val = 2.0 * rand + (1.0 - 2.0 * rand) * (np.power(xy, (self.eta + 1.0)))
        d = np.power(val, mut_pow) - 1.0
        deltaq[mask] = d[mask]

        xy = 1.0 - delta2
        val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (np.power(xy, (self.eta + 1.0)))
        d = 1.0 - np.power(val, mut_pow)
        deltaq[mask_not] = d[mask_not]

        # Update mutated values
        _Y = X + deltaq * (xu - xl)

        # Ensure values stay within bounds
        _Y[_Y < xl] = xl[_Y < xl]
        _Y[_Y > xu] = xu[_Y > xu]

        Y[do_mutation] = _Y
        Y = set_to_bounds_if_outside_by_problem(problem, Y)

        # Reconstruct individuals with updated genetic data
        X_ori = np.full(X_copy.shape, None)
        shapeX = X_copy[0, 0].individual[index].realgene.shape

        for k in range(Y.shape[0]):
            X_ori[k, 0] = copy.deepcopy(X_copy[k, 0])
            X_ori[k, 0].individual[index] = GenCgpW(
                X_copy[k, 0].conf_net[index],
                np.copy(Y[k].reshape(shapeX)),
                weights=X_copy[k, 0].individual[index].weightcolums
            )
            X_ori[k, 0].individual[index].to_int_cgp()

        return X_ori

    def _do(self, problem, X, **kwargs):
        """
        Execute the mutation operation on the given population.

        Args:
            problem (Problem): Problem definition, providing variable bounds.
            X (ndarray): Population to mutate.

        Returns:
            ndarray: Updated population after mutation.
        """
        X_temp = copy.deepcopy(X)

        for i in range(X[0, 0].normal):
            X_temp = self.mutation_base(X_temp, problem, i)

        return X_temp


# import copy
#
# import numpy as np
# from pymoo.core.mutation import Mutation
# from pymoo.operators.repair.to_bound import set_to_bounds_if_outside_by_problem
# from Evolution.CellCGPW import CellCgpW
#
# from Evolution.GenCGPW import GenCgpW
#
#
# class MutationCellCgpW(Mutation):
#     def __init__(self, eta, prob=None):
#         super().__init__()
#         self.eta = float(eta)
#
#         if prob is not None:
#             self.prob = float(prob)
#         else:
#             self.prob = None
#         # self.xu=1.0
#         # self.xl=0.0
#
#     def mutation_base(self, X, problem, index):
#
#         n_matings, n_var = X.shape
#         X_copy = copy.deepcopy(X)
#         X_t = np.zeros(shape=(n_matings, X[0, 0].n_var_size))
#         for i in range(n_matings):
#             X[i, 0].individual[index].toReal()
#
#             X_t[i, :] = X[i, 0].individual[index].realgene.flatten()
#         X = X_t
#         X = X.astype(float)
#         Y = np.full(X.shape, np.inf)
#         n_matings, n_var = X.shape
#         if self.prob_var is None:
#             self.prob_var = 1.0 / n_var
#
#         do_mutation = np.random.random(X.shape) < self.prob_var
#
#         Y[:, :] = X
#
#         xl = np.repeat(problem.xl[None, :], X.shape[0], axis=0)[do_mutation]
#         xu = np.repeat(problem.xu[None, :], X.shape[0], axis=0)[do_mutation]
#
#         X = X[do_mutation]
#
#         delta1 = (X - xl) / (xu - xl)
#         delta2 = (xu - X) / (xu - xl)
#
#         mut_pow = 1.0 / (self.eta + 1.0)
#
#         rand = np.random.random(X.shape)
#         mask = rand <= 0.5
#         mask_not = np.logical_not(mask)
#
#         deltaq = np.zeros(X.shape)
#
#         xy = 1.0 - delta1
#         val = 2.0 * rand + (1.0 - 2.0 * rand) * (np.power(xy, (self.eta + 1.0)))
#         d = np.power(val, mut_pow) - 1.0
#         deltaq[mask] = d[mask]
#
#         xy = 1.0 - delta2
#         val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (np.power(xy, (self.eta + 1.0)))
#         d = 1.0 - (np.power(val, mut_pow))
#         deltaq[mask_not] = d[mask_not]
#
#         # mutated values
#         _Y = X + deltaq * (xu - xl)
#
#         # back in bounds if necessary (floating point issues)
#         _Y[_Y < xl] = xl[_Y < xl]
#         _Y[_Y > xu] = xu[_Y > xu]
#
#         # set the values for output
#         Y[do_mutation] = _Y
#
#         # in case out of bounds repair (very unlikely)
#         Y = set_to_bounds_if_outside_by_problem(problem, Y)
#
#         X_ori = np.full(X_copy.shape, None)
#         shapeX = X_copy[0, 0].individual[index].realgene.shape
#
#         for k in range(Y.shape[0]):
#             X_ori[k, 0] = copy.deepcopy(X_copy[k, 0])
#             X_ori[k, 0].individual[index]= GenCgpW(X_copy[k, 0].conf_net[index], np.copy(Y[k].reshape(shapeX)),
#                                   weights=X_copy[k, 0].individual[index].weightcolums)
#             X_ori[k, 0].individual[index].to_int_cgp()
#         return X_ori
#
#     def _do(self, problem, X, **kwargs):
#
#         X_temp = copy.deepcopy(X)
#
#         for i in range(X[0, 0].normal):
#             X_temp = self.mutation_base(X_temp, problem, i)
#
#         return X_temp
#
# #
