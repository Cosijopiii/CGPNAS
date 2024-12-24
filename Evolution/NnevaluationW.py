
import numpy as np
import torch
from pymoo.core.problem import Problem
from Evolution.TrainCnn_nW import CNNTrainW
from torchprofile import profile_macs
from pathos.pools import ProcessPool as Pool


class NNevalW(Problem):
    """
    Custom evaluation problem for neural networks using CGP-NAS.

    Attributes:
        nvar_real (int): Number of real-valued variables.
        epochNum (int): Number of epochs for training.
        dataimageset (str): Dataset name for training and validation.
    """

    def __init__(self, nvar_real=0, epochNum=1, dataimageset=None,size=80, validation=True, verbose=True,image_size=32,batch_size=128):
        """
        Initialize the evaluation problem.

        Args:
            nvar_real (int): Number of real-valued variables (default: 0).
            epochNum (int): Number of training epochs (default: 1).
            dataimageset (str): Dataset name for training and validation.
            size(int): Number of training dataset size % (default: 80)
        """

        super().__init__(n_var=1, n_obj=2, n_constr=0, requires_kwargs=True)
        # Attributes
        self.validation = validation
        self.verbose = verbose
        self.imgSize = image_size
        self.batchsize = batch_size
        self.datasetname = dataimageset
        self.epochNum = epochNum
        self.size=size
        # Bounds for real-valued variables
        self.xl = np.zeros(nvar_real)
        self.xu = 0.999999 * np.ones(nvar_real)




    @staticmethod
    def count_parameters(model):
        """
        Count the number of trainable parameters in a model.

        Args:
            model (torch.nn.Module): PyTorch model.

        Returns:
            int: Number of trainable parameters.
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def eval_single_solution(self, individual, gpu_id):
        """
        Evaluate a single solution on a specific GPU.

        Args:
            individual: Individual neural network configuration.
            gpu_id (int): GPU ID for computation.

        Returns:
            list: Evaluation metrics [error, MACs, parameters].
        """

        error, macs, parameters = 100, 99999999999, 99999999999
        try:
            # Initialize training
            trainer = CNNTrainW(
                individual[0].active_net_list(),
                self.datasetname,
                validation=True,
                verbose=self.verbose,
                img_size=self.imgSize,
                batch_size=self.batchsize,
                size=self.size
            )
            evaluation, model = trainer(gpu_id, self.epochNum)
            error = (1 - float(evaluation)) * 100

            # Calculate MACs and parameters
            device = torch.device(f"mps" if torch.backends.mps.is_available() else f"cuda:{gpu_id}" if gpu_id >= 0 else "cpu")
            input_tensor = torch.zeros(1, 3, self.imgSize, self.imgSize, dtype=torch.float, device=device, requires_grad=False)
            macs = profile_macs(model, input_tensor) / 1e6  # Convert to millions
            parameters = self.count_parameters(model) / 1e6  # Convert to millions
        except Exception as e:
            print(f"Error during evaluation: {e}")

        print(f"GPU {gpu_id}: Metrics - [Error: {error}, MACs: {macs}, Parameters: {parameters}]")
        return [error, macs]

 

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate the population of solutions.

        Args:
            x: Population of solutions.
            out: Dictionary to store the output metrics.
        """
        gpu_list = list(range(1))  # Adjust based on available GPUs
        index = 0
        all_solutions = []

        for batch_idx in range(len(x) // len(gpu_list)):
            tasks = [(x[index + j], gpu_list[j]) for j in range(len(gpu_list))]
            index += len(gpu_list)

            # Parallel evaluation using process pool
            with Pool(nodes=len(gpu_list)) as pool:
                batch_results = pool.map(lambda args: self.eval_single_solution(*args), tasks)
                all_solutions.extend(batch_results)

        # Store results
        out["F"] = np.array(all_solutions, dtype=float)
        print("Final Evaluation Results:", out["F"])



