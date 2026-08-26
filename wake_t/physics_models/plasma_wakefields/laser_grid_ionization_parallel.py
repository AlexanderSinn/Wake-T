import numpy as np
import scipy.constants as ct

from .laser_grid_ionization import LaserGridIonization
from wake_t.utilities.numba import num_threads
from wake_t.utilities.other import ProfStart, ProfStop

import os
import cppimport

print("C++ Import Begin")
cppmodule = cppimport.imp_from_filepath(
    os.path.join(os.path.dirname(__file__), "parallel_solver.cpp")
)
print("C++ Import End")


class LaserGridIonizationParallel(LaserGridIonization):
    def _evolve_properties(self, bunches):
        ProfStart("EvolveLaserParallel")
        if self.laser is not None:
            if self.laser_evolution:
                k_0 = 2 * np.pi / self.laser.l_0
                k_p = np.sqrt(ct.e**2 * self.n_p / (ct.m_e * ct.epsilon_0)) / ct.c

                assert not self.laser.use_subgrid
                # if self.laser.use_subgrid:
                #     self.chi[2:-2, 2:-2] = self.laser._interpolate_chi_to_subgrid(self.chi[2:-2, 2:-2])

                omega0 = 2 * ct.pi * ct.c / self.laser.l_0

                cppmodule.parallel_solver(
                    self.laser._a_env,
                    self.laser._a_env_old,
                    self.chi[2:-2, 2:-2],
                    k_0,
                    k_p,
                    self.laser.solver_params["zmin"],
                    self.laser.solver_params["zmax"],
                    self.laser.solver_params["nz"],
                    self.laser.solver_params["rmax"],
                    self.laser.solver_params["nr"],
                    self.laser.solver_params["dt"],
                    self.laser.solver_params["nt"],
                    self.laser.solver_params["use_phase"],
                    self.laser.n_steps == 0,
                    len(self.ion_species),
                    self.ion_densities[:, 2:-1, 2:-2],
                    self.elec_density[2:-1, 2:-2],
                    self.ion_start_index.astype(np.int32),
                    self.ion_atomic_number.astype(np.int32),
                    self.ion_mass,
                    omega0,
                    self.adk_prefactors,
                    self.laser.polarization == "linear",
                    1 / np.abs(self.xi_fld[1] - self.xi_fld[0]),
                    num_threads,
                )

                # Update arrays and step count.
                self.laser._update_output_envelope()
                self.laser.n_steps += 1

        ProfStop("EvolveLaserParallel")
