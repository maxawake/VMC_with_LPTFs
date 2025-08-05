import gc
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import torch

from tvmc.hamiltonians.ising import Ising
from tvmc.hamiltonians.rydberg import Rydberg
from tvmc.utils.builder import build_model
from tvmc.utils.config import save_config
from tvmc.utils.cuda_helper import DEVICE
from tvmc.utils.helper import hdf5_writer, setup_dir

# Set the maximum duration
MAX_DURATION = timedelta(days=6, hours=22)


def reg_train(config, plot_queue=None, printf=False, output_path=None, start_time=None, resume=False):
    """Regular training loop.

    Args:
        config (dict): Configuration dictionary.
        plot_queue (mp.Queue, optional): Queue for plotting. Defaults to None.
        printf (bool, optional): Whether to print progress. Defaults to False.
        output_path (str, optional): Path to the output directory. Defaults to None.
        start_time (datetime, optional): Start time of the training. Defaults to None.
        resume (bool, optional): Whether to resume from a checkpoint. Defaults to False.

    Raises:
        ValueError: If the configuration is invalid.

    Returns:
        None
    """
    try:
        net, config = build_model(config)

        # Initialize optimizer
        beta1 = 0.9
        beta2 = 0.999
        optimizer = torch.optim.Adam(net.parameters(), lr=config["TRAIN"]["lr"], betas=(beta1, beta2))

        print(config["HAMILTONIAN"])
        if config["HAMILTONIAN"]["name"] == "Rydberg":
            h = Rydberg(**config["HAMILTONIAN"])
        elif config["HAMILTONIAN"]["name"] == "Ising":
            h = Ising(**config["HAMILTONIAN"])
        else:
            raise ValueError("Hamiltonian not implemented")

        # Set up output directory
        output_path = setup_dir(config, resume=resume)
        save_config(config, output_path)
        checkpoint_path = os.path.join(output_path, "checkpoint.pt")
        start_step = 0

        # Prepare queue for writing samples to disk
        sample_queue = mp.Queue()
        file_path = os.path.join(output_path, "samples.h5")

        # Start writer process
        writer_process = mp.Process(target=hdf5_writer, args=(sample_queue, file_path))
        writer_process.start()

        # Record the start time
        stop_training = False

        # Get the training options
        op = config["TRAIN"]

        if op["true_grad"]:
            assert op["Q"] == 1

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

            if plot_queue is not None:
                if plot_queue.qsize() < 100:  # avoid memory bloat
                    plot_queue.put({"samples": samplebatch.cpu().numpy(), "step": step, "stag_mag": stag_mag.item()})

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

            # Check if the elapsed time exceeds the limit
            if start_time is not None:
                if datetime.now() - start_time > MAX_DURATION:
                    print("Saving checkpoint and exiting.")

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

                    print("Exiting.")
                    sys.exit(0)

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
        sys.exit(0)
