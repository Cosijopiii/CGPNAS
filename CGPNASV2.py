from datetime import datetime
import pickle
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

# Import custom modules for the CGP model
from Evolution.CartesianCellSamplingW import CartesianCellGeneticProgrammingW
from Evolution.CartesianDefinitionW import CartesianGPConfigurationW
from Evolution.CrossoverCellCgpW import CrossoverCellCgpW
from Evolution.MutationCellCgpW import MutationCellCgpW
from Evolution.NnevaluationW import NNevalW


if __name__ == '__main__':
    # Configuration of functions and their arities for the CGP model
    FUNCTION_LIST = [
        'Concat', 'Sum', 'Bottleneck', 'FusedMBconv', 'MBconv',
        'SepConv', 'DiConv', 'Indentity', 'ResBlock',
        'ConvBlock', 'C1x7x7x1'
    ]
    ARITY_FUNCTIONS = [2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1]


    def configure_cgp_layer(rows=10, cols=4, channels=(32, 64)):
        """
        Configures a single CGP layer with specified parameters.

        Args:
            rows (int): Number of rows in the CGP grid.
            cols (int): Number of columns in the CGP grid.
            channels (tuple): Channels to be used in the layer.

        Returns:
            CartesianGPConfigurationW: Configured CGP layer.
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

    # CGP configuration for each layer with different channel settings
    Normal_Block_1 = configure_cgp_layer(rows=10, cols=4, channels=(32, 64))
    Normal_Block_2 = configure_cgp_layer(rows=10, cols=4, channels=(64, 128))
    Normal_Block_3 = configure_cgp_layer(rows=10, cols=4, channels=(128, 256))
    CONF_LAYERS = [Normal_Block_1, Normal_Block_2, Normal_Block_3]

    # CGP and NSGA-II configuration parameters
    n_weight = 2
    n_var = (Normal_Block_1.node_num + Normal_Block_1.out_num) * (Normal_Block_1.max_in_num + 1 + n_weight)
    n_gens = 20
    epochs = 1#36
    population_size = 3
    batchsize=128
    # Start time for execution tracking
    start_time = datetime.now()
    print("Execution started at:", start_time)

    # Define model pool for reduction blocks
    model_pool = [1, 1]  # Defines reduction blocks between normal blocks
    normal_blocks = 3

    # Configure the NSGA-II algorithm with CGP settings
    algorithm = NSGA2(
        pop_size=population_size,
        sampling=CartesianCellGeneticProgrammingW(CONF_LAYERS, model_pool, N=normal_blocks, R=normal_blocks - 1),
        crossover=CrossoverCellCgpW(eta=20, prob=0.9),
        mutation=MutationCellCgpW(prob=0.3, eta=15),
        eliminate_duplicates=False
    )

    # Dataset configuration for evaluation
    dataset_name = 'CIFAR-100'
    evaluation_problem = NNevalW(
        nvar_real=n_var,
        epochNum=epochs,
        dataimageset=dataset_name,
        batch_size=batchsize
    )

    # Run optimization using the CGP model evaluator
    results = minimize(
        evaluation_problem,
        algorithm,
        ('n_gen', n_gens),
        save_history=True,
        verbose=True
    )

    # Save results to a file
    output_filename = "Optimized_CGP_Results.pkl"
    with open(output_filename, mode='wb') as file:
        pickle.dump(results, file)

    # Report execution times
    end_time = datetime.now()
    print("Execution ended at:", end_time)
    print("Total Duration:", end_time - start_time)
    print("Results saved in:", output_filename)
    print("Execution completed.")
