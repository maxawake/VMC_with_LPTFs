import torch
import h5py
import os

from tvmc.models.RNN import PRNN


# Define HDF5 writer process (Separate entry for each training step)
def hdf5_writer(queue, file_path):
    with h5py.File(file_path, "w") as f:
        while True:
            data = queue.get()
            if data is None:  # Stop signal
                break

            step, samplebatch = data  # Unpack data (step number, sample tensor)
            step_key = f"step_{step:05d}"  # Store each step under "step_00001", "step_00002", etc.

            f.create_dataset(step_key, data=samplebatch, dtype="uint8")

    print("HDF5 writer process finished.")


def new_rnn_with_optim(rnntype, op, beta1=0.9, beta2=0.999):
    rnn = torch.jit.script(PRNN(op.L, **PRNN.DEFAULTS))
    optimizer = torch.optim.Adam(rnn.parameters(), lr=op.lr, betas=(beta1, beta2))
    return rnn, optimizer


def momentum_update(m, target_network, network):
    for target_param, param in zip(target_network.parameters(), network.parameters()):
        target_param.data.copy_(target_param.data * m + param.data * (1.0 - m))


def setup_dir(op_dict, resume=False):
    """Makes directory for output and saves the run settings there
    Inputs:
        op_dict (dict) - Dictionary of Options objects
    Outputs:
        Output directory output_path
    """
    op = op_dict["TRAIN"]

    # if op.dir == "<NONE>":
    #     return

    hname = op_dict["HAMILTONIAN"]["name"] if "HAMILTONIAN" in op_dict else "NA"

    output_path = op["dir"] + "/%s/%d-B=%d-K=%d%s" % (hname, op["L"], op["B"], op["K"], op["sub_directory"])

    os.makedirs(output_path, exist_ok=True)
    biggest = -1
    for paths, folders, files in os.walk(output_path):
        for f in folders:
            try:
                biggest = max(biggest, int(f))
            except Exception as e:
                print("Error in folder naming, please check your folder structure")
                print("Error: ", e)
                pass

    if resume:
        number = str(biggest)
    else:
        number = str(biggest + 1)
    output_path += "/" + number

    os.makedirs(output_path, exist_ok=True)

    print("Output folder path established")
    return output_path
