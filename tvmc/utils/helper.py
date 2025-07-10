import os

import h5py
import torch
import sys
from tvmc.models.RNN import PRNN


import signal

def hdf5_writer(queue, file_path):
    

    f = h5py.File(file_path, "w")

    def handler(signum, frame):
        print(f"HDF5 writer caught signal {signum}. Closing file safely.")
        f.flush()
        f.close()
        sys.exit(0)

    # Register signal handlers
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)

    try:
        while True:
            data = queue.get()
            if data is None:
                break

            step, samplebatch, debug_dict = data
            step_key = f"step_{step:05d}"

            grp = f.create_group(step_key)
            grp.create_dataset("sample", data=samplebatch, dtype="uint8")
            for key, value in debug_dict.items():
                grp.create_dataset(key, data=value)

    finally:
        print("HDF5 writer process finishing, flushing and closing.")
        f.flush()
        f.close()



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
