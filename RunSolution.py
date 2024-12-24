import pickle
import numpy as np
import torch
from datetime import datetime
from torchprofile import profile_macs
from Evolution.CgpTonn_nW import CgpNetW
from Evolution.TrainCnn_nW import CNNTrainW
from Evolution.CellCGPW import CellCgpW
from Utils.DatasetFactory import DatasetFactory
import traceback


def load_results(filename):
    """
    Load the results from a pickle file.

    Args:
        filename (str): Path to the pickle file.

    Returns:
        object: Loaded results.
    """
    try:
        with open(filename, mode='rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except Exception as e:
        print(f"Error loading file '{filename}': {e}")
        return None


def count_parameters(model):
    """
    Count the total number of trainable parameters in a model.

    Args:
        model (torch.nn.Module): The model to evaluate.

    Returns:
        int: Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_model(modelfromPareto, datasetname, dataset, imgSize, batchsize, gpuId, epochNum, aux_head=True):
    """
    Train and evaluate the model generated from the Pareto front solution.

    Args:
        modelfromPareto (list): Network architecture definition from Pareto front.
        datasetname (str): Dataset name.
        dataset (object): Dataset object.
        imgSize (int): Image size.
        batchsize (int): Batch size.
        gpuId (int): GPU ID to use for training.
        epochNum (int): Number of training epochs.
        aux_head (bool): Whether to include the auxiliary head.

    Returns:
        tuple: Evaluation accuracy, trained model, and MACs.
    """
    try:
        # Initialize the training process
        train = CNNTrainW(
            modelfromPareto,
            datasetname,
            validation=False,
            verbose=True,
            img_size=imgSize,
            batch_size=batchsize,
            aux_head=aux_head
        )

        # Train the model
        evaluation, model = train(gpuId, epochNum)
        error_rate = (1 - float(evaluation)) * 100

        # Compute MACs
        inp = torch.zeros(1, 3, imgSize, imgSize, dtype=torch.float, device='cuda', requires_grad=False)
        macs = profile_macs(model, inp)

        return evaluation, model, error_rate, macs

    except Exception as e:
        print(f"Error during model evaluation: {e}")
        traceback.print_exc()
        return None, None, None, None


def main():
    # Start time for performance measurement
    start_time = datetime.now()

    # Configuration Parameters
    filename = "Optimized_CGP_Results.pkl"  # Path to the results file
    solution_index = 0  # Index of the solution to test
    datasetname = 'CIFAR-100'  # Dataset name
    imgSize = 32  # Image size
    batchsize = 96  # Batch size
    gpuId = 0  # GPU ID for training
    epochNum = 600  # Number of training epochs
    AuxHead = True  # Include auxiliary head or not

    # Load the results
    results = load_results(filename)
    if results is None:
        return

    # Prepare the dataset
    dataset = DatasetFactory.getDataset(
        datasetname,
        seed=None,
        validation=False,
        download=True,
        pin_memory=False,
        num_workers=1,
        cutout=16,
        batch_size_test=batchsize,
        batch_size_train=batchsize
    )


    # Select the solution from the Pareto front
    modelfromPareto = results.X[solution_index][0].active_net_list(AuxHead)

     # Evaluate the selected solution
    evaluation, model, err, macs = evaluate_model(
        modelfromPareto, datasetname, dataset, imgSize, batchsize, gpuId, epochNum, aux_head=AuxHead
    )
    if model is None:
        return

    # Additional evaluation without the auxiliary head
    modelfromPareto_no_aux = results.X[solution_index][0].active_net_list(AuxHead=False)
    evaluation_no_aux, model_no_aux, _, _ = evaluate_model(
        modelfromPareto_no_aux, datasetname, dataset, imgSize, batchsize, gpuId, epochNum=1, aux_head=False
    )

    # Output Results
    print("\n=== Evaluation Results ===")
    print(f"Error Rate: {err:.2f}%")
    print(f"Total Parameters (With AuxHead): {count_parameters(model)}")
    print(f"Total Parameters (No AuxHead): {count_parameters(model_no_aux)}")
    print(f"MACs (Million): {macs / 1e6:.2f}M")
    print(f"Total Time: {datetime.now() - start_time}")


# Run the script
if __name__ == "__main__":
    main()
