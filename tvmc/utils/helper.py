import datetime
import glob
import os
from glob import glob

import h5py
import torch

from tvmc.models.RNN import PRNN


def hdf5_writer(queue, file_path):
    """HDF5 writer thread

    Args:
        queue (mp.queue): Queue
        file_path (str): File path
    """
    f = h5py.File(file_path, "a")

    try:
        while True:
            data = queue.get()
            if data is None:
                break

            step, samplebatch, debug_dict = data
            step_key = f"step_{step:05d}"

            # Create a group for the current step if it doesn't exist
            grp = f.create_group(step_key)

            # Save the sample batch
            grp.create_dataset("sample", data=samplebatch, dtype="uint8")

            # Save the debug information
            for key, value in debug_dict.items():
                grp.create_dataset(key, data=value)

    finally:
        print("HDF5 writer process finishing, flushing and closing.")
        f.flush()
        f.close()


def new_rnn_with_optim(rnntype, op, beta1=0.9, beta2=0.999):
    """Create a new RNN model with an optimizer.

    Args:
        rnntype (str): The type of RNN to create.
        op (dict): The operation configuration.
        beta1 (float, optional): The beta1 parameter for the optimizer. Defaults to 0.9.
        beta2 (float, optional): The beta2 parameter for the optimizer. Defaults to 0.999.

    Returns:
        tuple: A tuple containing the RNN model and the optimizer.
    """
    rnn = torch.jit.script(PRNN(op.L, **PRNN.DEFAULTS))
    optimizer = torch.optim.Adam(rnn.parameters(), lr=op.lr, betas=(beta1, beta2))
    return rnn, optimizer


def momentum_update(m, target_network, network):
    """Update the target network using momentum.

    Args:
        m (float): Momentum factor.
        target_network (torch.nn.Module): Target network.
        network (torch.nn.Module): Current network.
    """
    for target_param, param in zip(target_network.parameters(), network.parameters()):
        target_param.data.copy_(target_param.data * m + param.data * (1.0 - m))


def setup_dir(op_dict, resume=False):
    """Set up the output directory for training.

    Args:
        op_dict (dict): Dictionary containing operation settings.
        resume (bool, optional): Whether to resume from a previous run. Defaults to False.

    Returns:
        str: The path to the output directory.
    """
    op = op_dict["TRAIN"]

    # if op.dir == "<NONE>":
    #     return

    hname = op_dict["HAMILTONIAN"]["name"] if "HAMILTONIAN" in op_dict else "NA"

    output_path = os.path.join(op["dir"], hname)

    os.makedirs(output_path, exist_ok=True)

    if resume:
        timestamps = [path for path in glob.glob(os.path.join(output_path, "*"))]
        timestamps.sort()
        if timestamps:
            output_path = timestamps[-1]
            return output_path
        else:
            print("No previous runs found, starting fresh.")

    output_path = os.path.join(output_path, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(output_path, exist_ok=True)

    print("Output folder path established at:", output_path)
    return output_path
