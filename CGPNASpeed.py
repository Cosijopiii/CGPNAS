from datetime import datetime
import pickle

# Import necessary modules for CGP and NAS configuration
from pymoo.algorithms.moo.nsga2 import NSGA2, binary_tournament
from pymoo.operators.selection.tournament import TournamentSelection

from Evolution.CartesianCellSamplingW import CartesianCellGeneticProgrammingW
from Evolution.CartesianDefinitionW import CartesianGPConfigurationW
from Evolution.CrossoverCellCgpW import CrossoverCellCgpW
from Evolution.MutationCellCgpW import MutationCellCgpW
from Evolution.NnevaluationW import NNevalW

if __name__ == '__main__':
    # Define available functions for CGP and their arities
    FUNCTION_LIST = [
        'Concat', 'Sum', 'Bottleneck', 'FusedMBconv', 'MBconv',
        'SepConv', 'DiConv', 'Indentity', 'ResBlock',
        'ConvBlock', 'C1x7x7x1'
    ]
    ARITY_FUNCTIONS = [2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    def configure_cgp_layer(rows=10, cols=4, channels=(32, 64)):
        """
        Configures a CGP layer with the specified parameters.

        Args:
            rows (int): Number of rows in the CGP grid.
            cols (int): Number of columns in the CGP grid.
            channels (tuple): Channel configurations for the layer.

        Returns:
            CartesianGPConfigurationW: A configured CGP layer.
        """
        return CartesianGPConfigurationW(
            rows=rows,
            cols=cols,
            level_back=1,
            min_active_num=1,
            max_active_num=0,
            funcs=FUNCTION_LIST,
            funAry=ARITY_FUNCTIONS,
            channels=channels
        )


    # Define CGP layers with specific configurations
    Normal_Block_1 = configure_cgp_layer(rows=10, cols=4, channels=(32, 64))
    Normal_Block_2 = configure_cgp_layer(rows=10, cols=4, channels=(64, 128))
    Normal_Block_3 = configure_cgp_layer(rows=10, cols=4, channels=(128, 256))
    CONF_LAYERS = [Normal_Block_1, Normal_Block_2, Normal_Block_3]

    # Set problem parameters for CGP
    n_weight = 2
    n_var = (Normal_Block_1.node_num + Normal_Block_1.out_num) * (Normal_Block_1.max_in_num + 1 + n_weight)
    n_gens = 10
    epochs = 36
    population_size = 24
    batchsize=128
    # Record start time
    start_time = datetime.now()
    print("Execution started at:", start_time)

    # Configure the NSGA-II algorithm
    model_pool = [1, 1]
    normal_blocks = 3
    algorithm = NSGA2(
        pop_size=population_size,
        selection=TournamentSelection(func_comp=binary_tournament),
        sampling=CartesianCellGeneticProgrammingW(CONF_LAYERS, model_pool, N=normal_blocks, R=normal_blocks - 1),
        crossover=CrossoverCellCgpW(eta=15, prob=0.9),
        mutation=MutationCellCgpW(prob=0.3, eta=20),
        eliminate_duplicates=False
    )

    # Configure dataset and evaluation problem
    dataset_name = 'CIFAR-100'
    problem = NNevalW(nvar_real=n_var, epochNum=epochs, dataimageset=dataset_name,batch_size=batchsize)

    # Initialize the algorithm with the problem
    algorithm.setup(problem, termination=('n_gen', n_gens), seed=1, verbose=True, save_history=True)


    def generate_stepped_values(start, end, steps):
        """
        Generate stepped values between two points for incremental learning.

        Args:
            start (float): Starting value.
            end (float): Ending value.
            steps (int): Number of steps.

        Returns:
            list: A list of stepped values.
        """
        if start > end:
            start, end = end, start

        increment = (end - start) / (steps - 1)
        return [start + i * increment for i in range(steps)]


    # Generate stepped values for dynamic problem size
    stepped_values = generate_stepped_values(20, 80, n_gens)
    step_index = 0

    # Run the optimization loop until the algorithm terminates
    while algorithm.has_next():

        # Request the next set of solutions to evaluate
        population = algorithm.ask()

        # Dynamically adjust the problem size and evaluate the population
        problem.size = int(stepped_values[step_index])
        algorithm.evaluator.eval(problem, population)

        # Return the evaluated solutions to the algorithm
        algorithm.tell(infills=population)

        # Log progress
        print(f"Generation {algorithm.n_gen}, Evaluations {algorithm.evaluator.n_eval}")
        step_index += 1

    # Obtain the final optimization results
    results = algorithm.result()

    # Save results to a file
    output_filename = "Optimized_CGP_Results.pkl"
    with open(output_filename, mode='wb') as file:
        pickle.dump(results, file)

    # Report execution times
    end_time = datetime.now()
    print("Execution ended at:", end_time)
    print("Total duration:", end_time - start_time)
    print("Results saved to:", output_filename)
