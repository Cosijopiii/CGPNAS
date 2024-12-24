import sys

from Evolution.CgpTonn_nW import CgpNetW
from Evolution.NnevaluationW import NNevalW
from torchprofile import profile_macs
import numpy as np
import torch
from pathos.pools import ProcessPool as Pool
from Evolution.TrainCnn_nW import CNNTrainW

class NNevalWX(NNevalW):
    """
    Class for evaluating neural network solutions using a specific GPU.
    Inherits from NNevalW.
    """
    def __init__(self, nvar_real=0, epochNum=1, dataimageset=None,size=80,batch_size=128,image_size=32):
        super().__init__(nvar_real, epochNum, dataimageset, size,batch_size=batch_size,image_size=image_size)
        self.xl = np.zeros(nvar_real)
        self.xu = 0.999999 * np.ones(nvar_real)

    @staticmethod
    def getmodel(cnnmodel):
        """
        Get the model for evaluation.

        Args:
            cnnmodel: The CNN model configuration.

        Returns:
            n_nmodel: The configured model.
        """
        num_classes = 4  # Four rotation levels
        n_nmodel = CgpNetW(cnnmodel, 3, num_classes, 32, False)
        return n_nmodel



    def eval_single_solution(self, individual, gpu_id, info):
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

            Cnn_cgp_macs = individual[0].active_net_list()

            generation = info.n_gen

            if generation <= 10:
                Cnn_cgp = individual[0].active_net_list_S(stage=1)
            elif 10 < generation <= 20:
                Cnn_cgp = individual[0].active_net_list_S(stage=2)
            else:
                Cnn_cgp = individual[0].active_net_list()
            # Initialize training
            trainer = CNNTrainW(
                Cnn_cgp,
                self.datasetname,
                validation=True,
                verbose=self.verbose,
                img_size=self.imgSize,
                batch_size=self.batchsize,
                size=self.size
            )
            evaluation, model = trainer(gpu_id, self.epochNum)
            error = (1 - float(evaluation)) * 100

            device = torch.device(f"mps" if torch.backends.mps.is_available() else f"cuda:{gpu_id}" if gpu_id >= 0 else "cpu")
            # Calculate MACs and parameters
            input_tensor = torch.zeros(1, 3, self.imgSize, self.imgSize, dtype=torch.float, device=device, requires_grad=False)
            macs = profile_macs(self.getmodel(Cnn_cgp_macs).to(torch.device(device)), input_tensor) / 1e6  # Convert to millions
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
        algorithm_info=kwargs['algorithm']
        for batch_idx in range(len(x) // len(gpu_list)):
            tasks = [(x[index + j], gpu_list[j]) for j in range(len(gpu_list))]
            index += len(gpu_list)

            # Parallel evaluation using process pool
            with Pool(nodes=len(gpu_list)) as pool:
                batch_results = pool.map(lambda args: self.eval_single_solution(*args,algorithm_info), tasks)
                all_solutions.extend(batch_results)

        # Store results
        out["F"] = np.array(all_solutions, dtype=float)
        print("Final Evaluation Results:", out["F"])