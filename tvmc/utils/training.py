import multiprocessing as mp
import sys
import time
import os
import numpy as np
import torch

from tvmc.hamiltonians.rydberg import Rydberg
from tvmc.hamiltonians.ising import Ising
from tvmc.utils.cuda_helper import DEVICE
from tvmc.utils.helper import hdf5_writer, new_rnn_with_optim, setup_dir

import signal

stop_training = False

def handle_sigterm(signum, frame):
    global stop_training
    print(f"Received signal {signum}. Preparing to terminate gracefully.")
    stop_training = True

# Register the signal handler
signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGINT, handle_sigterm)  # Optional: handle Ctrl+C similarly

def reg_train(op, net_optim=None, printf=False, output_path=None, resume=False):
    try:
        print(op["HAMILTONIAN"])
        if op["HAMILTONIAN"]["name"] == "Rydberg":
            h = Rydberg(**op["HAMILTONIAN"])
        elif op["HAMILTONIAN"]["name"] == "Ising":
            h = Ising(**op["HAMILTONIAN"])
        else:
            raise ValueError("Hamiltonian not implemented")

        # Set up output directory and checkpoint path
        output_path = setup_dir(op, resume=resume)
        checkpoint_path = os.path.join(output_path, "checkpoint.pt")
        start_step = 0

        # Prepare queue for writing samples to disk
        sample_queue = mp.Queue()
        print(output_path)
        file_path = os.path.join(output_path, "samples.h5")

        # Start writer process
        writer_process = mp.Process(target=hdf5_writer, args=(sample_queue, file_path))
        writer_process.start()

        op = op["TRAIN"]

        if op["true_grad"]:
            assert op["Q"] == 1

        if net_optim is None:
            net, optimizer = new_rnn_with_optim("GRU", op)
        else:
            net, optimizer = net_optim

        if resume and os.path.exists(checkpoint_path):
            print("Resuming from checkpoint...")
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            net.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_step = checkpoint["step"] + 1

        debug = []
        losses = []

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

        t = time.time()
        print("Start Training")
        for step in range(start_step, op["steps"]):
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

            step_debug = {
                "Eo": Eo.item(),
                "Ev": Ev.item(),
                "mag": mag.item(),
                "mag_var": mag_v.item(),
                "abs_mag": abs_mag.item(),
                "abs_mag_var": abs_mag_v.item(),
                "sq_mag": sq_mag.item(),
                "sq_mag_var": sq_mag_v.item(),
                "stag_mag": stag_mag.item(),
                "stag_mag_var": stag_mag_v.item(),
                "time": time.time() - t,
            }

            # Send samples to HDF5 writer asynchronously
            sample_queue.put((step, samplebatch.cpu().numpy(), step_debug))

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

            if step % 10 == 0:
                print(f"Step: {step}, Loss: {losses[-1]:.3f}")

                if printf:
                    sys.stdout.flush()
            if step % 100 == 0:
                print("Saving checkpoint...")
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": net.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    checkpoint_path,
                )
                
            if stop_training:
                print("Graceful termination requested. Saving checkpoint and exiting.")

                # Save checkpoint safely
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": net.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    checkpoint_path,
                )
                print(f"Checkpoint saved at step {step}.")

                # Signal hdf5_writer to close
                sample_queue.put(None)
                writer_process.join()
                print("HDF5 writer closed.")

                # Optionally save debug arrays if needed
                DEBUG = np.array(debug)
                if op["dir"] is not None:
                    np.save(output_path + "/DEBUG", DEBUG)
                    net.save(output_path + "/T")

                print("Exiting gracefully due to SIGTERM.")
                return DEBUG

        DEBUG = np.array(debug)

        if op["dir"] is not None:
            np.save(output_path + "/DEBUG", DEBUG)
            net.save(output_path + "/T")

        # Signal writer process to stop
        sample_queue.put(None)
        writer_process.join()

    except KeyboardInterrupt:
        sample_queue.put(None)
        writer_process.join()
        if op["dir"] is not None:
            DEBUG = np.array(debug)
            np.save(output_path + "/DEBUG", DEBUG)
            net.save(output_path + "/T")
    return DEBUG
