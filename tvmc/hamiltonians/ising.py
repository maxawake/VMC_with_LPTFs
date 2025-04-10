import numpy as np
import torch

from tvmc.hamiltonians.hamiltonian import Hamiltonian
from tvmc.utils.cuda_helper import DEVICE


class Ising(Hamiltonian):
    """
    Implementation of the Transverse field Ising model with Periodic Boundary Conditions
    Default parameters are:
    L=16 (number of spins)
    h_x=-1.0 (transverse field)
    J=1.0 (coupling constant)
    """

    def __init__(self, L, h_x, J, device=DEVICE, **kwargs):
        self.J = J
        self.h = h_x
        super(Ising, self).__init__(L, h_x, device)

    def buildlattice(self):
        # building hamiltonian matrix for diagonal part
        mat = np.zeros([self.L, self.L], dtype=np.float64)
        for i in range(self.L):
            mat[i, (i + 1) % self.L] = -self.J

        with torch.no_grad():
            self.Vij.weight[:, :] = torch.Tensor(mat)
            self.Vij.bias.fill_(0.0)  # no longitudinal field

    def localenergy(self, samples, logp, sumsqrtp, logsqrtp):
        return super(Ising, self).localenergy(2 * samples - 1, logp, sumsqrtp, logsqrtp)

    def ground(self):
        """Exact solution for the ground state the 1D Ising model with PBC"""
        N, h = self.L, self.h / self.J
        Pn = np.pi / N * (np.arange(-N + 1, N, 2))
        E0 = -1 / N * np.sum(np.sqrt(1 + h**2 - 2 * h * np.cos(Pn)))
        return E0 * self.J
