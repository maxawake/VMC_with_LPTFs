import multiprocessing as mp
import sys
import time

import numpy as np
import torch

from tvmc.hamiltonians.rydberg import Rydberg
from tvmc.utils.cuda_helper import DEVICE
from tvmc.utils.helper import hdf5_writer, new_rnn_with_optim, setup_dir


def reg_train(op, net_optim=None, printf=False, mydir=None):
    try:
        # Prepare queue for writing samples to disk
        sample_queue = mp.Queue()
        file_path = "samples.h5"

        # Start writer process
        writer_process = mp.Process(target=hdf5_writer, args=(sample_queue, file_path))
        writer_process.start()

        h = Rydberg(**op["HAMILTONIAN"])

        if mydir == None:
            mydir = setup_dir(op)

        op = op["TRAIN"]

        if op["true_grad"]:
            assert op["Q"] == 1

        if net_optim is None:
            net, optimizer = new_rnn_with_optim("GRU", op)
        else:
            net, optimizer = net_optim

        debug = []
        losses = []
        true_energies = []

        # samples
        samplebatch = torch.zeros([op["B"], op["L"], 1], device=DEVICE)
        # sum of off diagonal labels for each sample (scaled)
        sump_batch = torch.zeros([op["B"]], device=DEVICE)
        # scaling factors for the off-diagonal sums
        sqrtp_batch = torch.zeros([op["B"]], device=DEVICE)

        def fill_batch():
            with torch.no_grad():
                for i in range(op["Q"]):
                    sample, logp = net.sample(op["K"], op["L"])
                    # get the off diagonal info
                    sump, sqrtp = net.off_diag_labels_summed(sample, nloops=op["NLOOPS"])
                    samplebatch[i * op["K"] : (i + 1) * op["K"]] = sample
                    sump_batch[i * op["K"] : (i + 1) * op["K"]] = sump
                    sqrtp_batch[i * op["K"] : (i + 1) * op["K"]] = sqrtp
            return logp

        i = 0
        t = time.time()
        print("Start Training")
        for x in range(op["steps"]):
            # gather samples and probabilities
            if op["Q"] != 1:
                fill_batch()
                logp = net.logprobability(samplebatch)
            else:
                if op["sgrad"]:
                    samplebatch, logp = net.sample(op["B"], op["L"])
                else:
                    with torch.no_grad():
                        samplebatch, _ = net.sample(op["B"], op["L"])
                    # if you sample without gradients you have to recompute probabilities with gradients
                    logp = net.logprobability(samplebatch)

                if op["true_grad"]:
                    sump_batch, sqrtp_batch = net.off_diag_labels_summed(samplebatch, nloops=op["NLOOPS"])
                else:
                    # don't need gradients on the off diagonal when approximating gradients
                    with torch.no_grad():
                        sump_batch, sqrtp_batch = net.off_diag_labels_summed(samplebatch, nloops=op["NLOOPS"])

            # obtain energy
            with torch.no_grad():
                E = h.localenergy(samplebatch, logp, sump_batch, sqrtp_batch)
                # energy mean and variance
                Ev, Eo = torch.var_mean(E)

                MAG, ABS_MAG, SQ_MAG, STAG_MAG = h.magnetizations(samplebatch)
                mag_v, mag = torch.var_mean(MAG)
                abs_mag_v, abs_mag = torch.var_mean(ABS_MAG)
                sq_mag_v, sq_mag = torch.var_mean(SQ_MAG)
                stag_mag_v, stag_mag = torch.var_mean(STAG_MAG)

            ERR = Eo / (op["L"])

            if op["true_grad"]:
                # get the extra loss term
                h_x = h.offDiag * sump_batch * torch.exp(sqrtp_batch - logp / 2)
                loss = (logp * E).mean() + h_x.mean()

            else:
                loss = (logp * (E - Eo)).mean() if op["B"] > 1 else (logp * E).mean()

            # Main loss curve to follow
            losses.append(ERR.cpu().item())

            # Send samples to HDF5 writer asynchronously
            sample_queue.put((x, samplebatch.cpu().numpy()))

            # update weights
            net.zero_grad()
            loss.backward()
            optimizer.step()

            debug += [
                [
                    Eo.item(),
                    Ev.item(),
                    mag.item(),
                    mag_v.item(),
                    abs_mag.item(),
                    abs_mag_v.item(),
                    sq_mag.item(),
                    sq_mag_v.item(),
                    stag_mag.item(),
                    stag_mag_v.item(),
                    time.time() - t,
                ]
            ]

            if x % 10 == 0:
                print(f"Step: {int(time.time() - t)}, Loss: {losses[-1]:.3f}")
                if x % 100 == 0:
                    print()
                if printf:
                    sys.stdout.flush()
        print(time.time() - t, x + 1)

        DEBUG = np.array(debug)

        if op.dir != "<NONE>":
            np.save(mydir + "/DEBUG", DEBUG)
            net.save(mydir + "/T")

        # Signal writer process to stop
        sample_queue.put(None)
        writer_process.join()

    except KeyboardInterrupt:
        sample_queue.put(None)
        writer_process.join()
        if op.dir != "<NONE>":
            DEBUG = np.array(debug)
            np.save(mydir + "/DEBUG", DEBUG)
            net.save(mydir + "/T")
    return DEBUG
