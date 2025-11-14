"""Contains the definition of the `PlasmaParticles` class."""

from typing import Optional, List, Callable

import numpy as np
import scipy.constants as ct

from .psi_and_derivatives import (
    calculate_psi_with_interpolation,
    calculate_psi_and_derivatives_at_particles,
)
from .deposition import deposit_plasma_particles
from .gather import gather_bunch_sources, gather_laser_sources
from .b_theta import (
    calculate_b_theta_at_particles,
    calculate_b_theta_with_interpolation,
)
from .plasma_push.ab2 import evolve_plasma_ab2
from .utils import (
    calculate_chi,
    calculate_rho,
    update_gamma_and_pz,
    sort_particle_arrays,
    check_gamma,
    log,
)
from .plasma_particle_container import PlasmaParticleContainerPy


class PlasmaParticles:
    """
    Class containing a 1D slice of plasma particles.

    In the current implementation, this class stores both the plasma electrons
    and ions. It would be useful to change this in the future so that it
    stores only a single species. This would allow us to more easily
    extend the wakefield model to cases with more than 2 species, which would
    be great to model ionization, for example.

    Parameters
    ----------
    r_max : float
        Maximum radial extension of the simulation box in normalized units.
    r_max_plasma : float
        Maximum radial extension of the plasma column in normalized units.
    dr : float
        Radial step size of the discretized simulation box.
    ppc : float
        Number of particles per cell.
    nr, nz : int
        Number of grid elements along `r` and `z`.
    radial_density : callable
        Function defining the radial density profile.
    max_gamma : float, optional
        Plasma particles whose ``gamma`` exceeds ``max_gamma`` are
        considered to violate the quasistatic condition and are put at
        rest (i.e., ``gamma=1.``, ``pr=pz=0.``). By default 10.
    ion_motion : bool, optional
        Whether to allow the plasma ions to move. By default, False.
    ion_mass : float, optional
        Mass of the plasma ions. By default, the mass of a proton.
    free_electrons_per_ion : int, optional
        Number of free electrons per ion. The ion charge is adjusted
        accordingly to maintain a quasi-neutral plasma (i.e.,
        ion charge = e * free_electrons_per_ion). By default, 1.
    pusher : str, optional
        The pusher used to evolve the plasma particles. Possible values
        are ``'ab2'`` (Adams-Bashforth 2nd order).
    shape : str
        Particle shape to be used for the beam charge deposition. Possible
        values are 'linear' or 'cubic'. By default 'linear'.
    store_history : bool, optional
        Whether to store the plasma particle evolution. This might be needed
        for diagnostics or because of the use of adaptive grids. By default,
        ``False``.
    diags : list, optional
        List of particle quantities to save to diagnostics.
    """

    def __init__(
        self,
        r_max: float,
        r_max_plasma: float,
        dr: float,
        ppc: float,
        nr: int,
        nz: int,
        radial_density: Callable[[float], float],
        max_gamma: Optional[float] = 10.0,
        ion_motion: Optional[bool] = True,
        ion_mass: Optional[float] = ct.m_p,
        free_electrons_per_ion: Optional[int] = 1,
        pusher: Optional[str] = "ab2",
        shape: Optional[str] = "linear",
        store_history: Optional[bool] = False,
        diags: Optional[List[str]] = [],
    ):
        # Store parameters.
        self.r_max = r_max
        self.r_max_plasma = r_max_plasma
        self.radial_density = radial_density
        self.dr = dr
        self.ppc = ppc
        self.pusher = pusher
        self.shape = shape
        self.max_gamma = max_gamma
        self.nr = nr
        self.nz = nz
        self.ion_motion = ion_motion
        self.ion_mass = ion_mass
        self.free_electrons_per_ion = free_electrons_per_ion
        self.store_history = store_history
        self.diags = diags
        self.species_list = [
            PlasmaParticleContainerPy(),
            PlasmaParticleContainerPy()
        ]
        self.species_list[0].is_ion = False
        self.species_list[1].is_ion = True

    def initialize(self):
        """Initialize column of plasma particles."""

        # Create radial distribution of plasma particles.
        rmin = 0.0
        for i in range(self.ppc.shape[0]):
            rmax = self.ppc[i, 0]
            ppc = self.ppc[i, 1]

            n_elec = int(np.round((rmax - rmin) / self.dr * ppc))
            dr_p_i = self.dr / ppc
            rmax = rmin + n_elec * dr_p_i

            r_i = np.linspace(rmin + dr_p_i / 2, rmax - dr_p_i / 2, n_elec)
            dr_p_i = np.ones(n_elec) * dr_p_i
            if i == 0:
                r = r_i
                dr_p = dr_p_i
            else:
                r = np.concatenate((r, r_i))
                dr_p = np.concatenate((dr_p, dr_p_i))

            rmin = rmax

        # Determine number of particles.
        num_per_species = r.shape[0]

        # Initialize particle arrays.
        # `q_center` represents the charge until the particle center. That is,
        # the charge of the first half of the particle.
        pr = np.zeros(num_per_species)
        pz = np.zeros(num_per_species)
        gamma = np.ones(num_per_species)
        id = np.arange(num_per_species, dtype=np.int32)
        w = dr_p * r * self.radial_density(r)
        w_center = w / 2 - dr_p**2 / 8

        for s in self.species_list:

            s.num_particles = num_per_species
            s.do_push = not s.is_ion or self.ion_motion
            s.store_history = self.store_history

            s.r = r
            s.dr_p = dr_p
            s.pr = pr
            s.pz = pz
            s.gamma = gamma
            s.w = w
            s.w_center = w_center
            s.r_to_x = np.ones(s.num_particles, dtype=np.int32)
            s.id = id
            if s.is_ion:
                s.mass = float(self.ion_mass / ct.m_e)
                s.charge = float(-self.free_electrons_per_ion)
            else:
                s.mass = float(self.free_electrons_per_ion)
                s.charge = float(self.free_electrons_per_ion)

            if s.store_history:
                s.r_hist = np.zeros((self.nz, s.num_particles))
                s.log_r_hist = np.zeros((self.nz, s.num_particles))
                s.xi_hist = np.zeros((self.nz, s.num_particles))
                s.pr_hist = np.zeros((self.nz, s.num_particles))
                s.pz_hist = np.zeros((self.nz, s.num_particles))
                s.w_hist = np.zeros((self.nz, s.num_particles))
                s.r_to_x_hist = np.zeros((self.nz, s.num_particles), dtype=np.int32)
                s.id_hist = np.zeros((self.nz, s.num_particles), dtype=np.int32)
                s.sum_1_hist = np.zeros((self.nz, s.num_particles + 1))
                s.sum_2_hist = np.zeros((self.nz, s.num_particles + 1))
                if not s.is_ion:
                    s.a_i_hist = np.zeros((self.nz, s.num_particles))
                    s.b_i_hist = np.zeros((self.nz, s.num_particles))
                    s.a_0_hist = np.zeros(self.nz)
                else:
                    s.a_i_hist = np.zeros((self.nz, 0))
                    s.b_i_hist = np.zeros((self.nz, 0))
                    s.a_0_hist = np.zeros(0)
                s.i_push = 0
                s.xi_current = 0.0
            else:
                s.r_hist = np.zeros((self.nz, 0))
                s.log_r_hist = np.zeros((self.nz, 0))
                s.xi_hist = np.zeros((self.nz, 0))
                s.pr_hist = np.zeros((self.nz, 0))
                s.pz_hist = np.zeros((self.nz, 0))
                s.w_hist = np.zeros((self.nz, 0))
                s.r_to_x_hist = np.zeros((self.nz, 0), dtype=np.int32)
                s.id_hist = np.zeros((self.nz, 0), dtype=np.int32)
                s.sum_1_hist = np.zeros((self.nz, 0))
                s.sum_2_hist = np.zeros((self.nz, 0))
                s.a_i_hist = np.zeros((self.nz, 0))
                s.b_i_hist = np.zeros((self.nz, 0))
                s.a_0_hist = np.zeros((0))
                s.i_push = 0
                s.xi_current = 0.0


        self.ions_computed = False

        # Allocate arrays that will contain the fields experienced by the
        # particles.
        self._allocate_field_arrays()

        # Allocate arrays needed for the particle pusher.
        if self.pusher == "ab2":
            self._allocate_ab2_arrays()

    def sort(self):
        """Sort plasma particles radially.

        The `q_species` and `m` arrays do not need to be sorted because all
        particles have the same value.
        """
        for s in self.species_list:
            if s.do_push:
                indices = np.argsort(s.r, kind="stable")
                sort_particle_arrays(s.serialize(), indices)

    def gather_laser_sources(self, a2, nabla_a2, r_min, r_max, dr):
        """Gather the source terms (a^2 and nabla(a)^2) from the laser."""
        for s in self.species_list:
            if s.do_push:
                gather_laser_sources(
                    a2,
                    nabla_a2,
                    r_min,
                    r_max,
                    dr,
                    s.r,
                    s.a2,
                    s.nabla_a2
                )

    def gather_bunch_sources(
        self, source_arrays, source_xi_indices, source_metadata, slice_i
    ):
        """Gather the source terms (b_theta) from the particle bunches."""
        for s in self.species_list:
            s.b_t_0[:] = 0.0
        for i in range(len(source_arrays)):
            array = source_arrays[i]
            idx = source_xi_indices[i]
            md = source_metadata[i]
            r_min = md[0]
            r_max = md[1]
            dr = md[2]
            if slice_i in idx:
                xi_index = slice_i + 2 - idx[0]
                for s in self.species_list:
                    if s.do_push:
                        gather_bunch_sources(
                            array[xi_index], r_min, r_max, dr, s.r, s.b_t_0
                        )

    def calculate_fields(self):
        """Calculate the fields at the plasma particles."""
        # Precalculate logarithms (expensive) to avoid doing so several times.
        for s in self.species_list:
            if s.do_push or not self.ions_computed:
                log(s.r, s.log_r)

        calculate_psi_and_derivatives_at_particles(
            list(s.serialize() for s in self.species_list),
            self.ions_computed
        )

        for s in self.species_list:
            if s.do_push:
                update_gamma_and_pz(
                    s.gamma,
                    s.pz,
                    s.pr,
                    s.a2,
                    s.psi,
                    s.charge,
                    s.mass,
                )
            if not s.is_ion:
                check_gamma(s.gamma, s.pz, s.pr, self.max_gamma)
        calculate_b_theta_at_particles(
            list(s.serialize() for s in self.species_list)
        )

    def calculate_psi_at_grid(self, r_eval, psi):
        """Calculate psi on the current grid slice."""
        add = False
        for s in self.species_list:
            calculate_psi_with_interpolation(
                r_eval,
                s.r,
                s.log_r,
                s.sum_1,
                s.sum_2,
                psi,
                add
            )
            add = True

    def calculate_b_theta_at_grid(self, r_eval, b_theta):
        """Calculate b_theta on the current grid slice."""
        for s in self.species_list:
            if not s.is_ion:
                calculate_b_theta_with_interpolation(
                    r_eval, s.a_0[0], s.a_i, s.b_i, s.r, b_theta
                )
                return

    def evolve(self, dxi):
        """Evolve plasma particles to next longitudinal slice."""
        for s in self.species_list:
            if s.do_push:
                evolve_plasma_ab2(
                    dxi,
                    s.r,
                    s.pr,
                    s.gamma,
                    s.mass,
                    s.charge,
                    s.r_to_x,
                    s.nabla_a2,
                    s.b_t_0,
                    s.b_t,
                    s.psi,
                    s.dr_psi,
                    s.dr,
                    s.dpr,
                )
            if s.store_history:
                s.i_push += 1
                s.xi_current -= dxi

                if not s.is_ion:
                    s.a_i = s.a_i_hist[-1 - s.i_push]
                    s.b_i = s.b_i_hist[-1 - s.i_push]
                s.sum_1 = s.sum_1_hist[-1 - s.i_push]
                s.sum_2 = s.sum_2_hist[-1 - s.i_push]
                s.rho = s.w_hist[-1 - s.i_push]
                s.log_r = s.log_r_hist[-1 - s.i_push]

                if not s.do_push:
                    s.sum_1[:] = s.sum_1_hist[-s.i_push, :]
                    s.sum_2[:] = s.sum_2_hist[-s.i_push, :]
                    s.log_r[:] = s.log_r_hist[-s.i_push, :]

    def calculate_weights(self):
        """Calculate the plasma density weights of each particle."""
        for s in self.species_list:
            if s.do_push or not self.ions_computed:
                calculate_rho(
                    s.charge,
                    s.w,
                    s.pz,
                    s.gamma,
                    s.rho,
                )

    def deposit_rho(self, rho, rho_e, rho_i, r_fld, nr, dr):
        """Deposit plasma density on a grid slice."""
        self.calculate_weights()
        # Deposit electrons
        for s in self.species_list:
            # TODO: fix add on second iteration
            if s.is_ion:
                deposit_plasma_particles(
                    s.r, s.rho, r_fld[0], nr, dr, rho_i, self.shape
                )
            else:
                deposit_plasma_particles(
                    s.r, s.rho, r_fld[0], nr, dr, rho_e, self.shape
                )
        rho[:] = rho_e
        rho += rho_i

    def deposit_chi(self, chi, r_fld, nr, dr):
        """Deposit plasma susceptibility on a grid slice."""
        for s in self.species_list:
            if not s.is_ion:
                calculate_chi(
                    s.charge,
                    s.w,
                    s.pz,
                    s.gamma,
                    s.chi,
                )
                deposit_plasma_particles(
                    s.r, s.chi, r_fld[0], nr, dr, chi, self.shape
                )

    def get_history(self):
        """Get the history of the evolution of the plasma particles.

        Returns
        -------
        dict
            A dictionary containing the particle history arrays.
        """
        if self.store_history:
            # TODO: return a per-species hisory
            history = {
                "r_hist": np.concatenate(list(s.r_hist for s in self.species_list), axis=1),
                "log_r_hist": np.concatenate(list(s.log_r_hist for s in self.species_list), axis=1),
                "xi_hist": np.concatenate(list(s.xi_hist for s in self.species_list), axis=1),
                "pr_hist": np.concatenate(list(s.pr_hist for s in self.species_list), axis=1),
                "pz_hist": np.concatenate(list(s.pz_hist for s in self.species_list), axis=1),
                "w_hist": np.concatenate(list(s.w_hist for s in self.species_list), axis=1),
                "r_to_x_hist": np.concatenate(list(s.r_to_x_hist for s in self.species_list), axis=1),
                "id_hist": np.concatenate(list(s.id_hist for s in self.species_list), axis=1),
                "sum_1_hist": np.concatenate(list(s.sum_1_hist for s in self.species_list), axis=1),
                "sum_2_hist": np.concatenate(list(s.sum_2_hist for s in self.species_list), axis=1),
                "a_i_hist": np.concatenate(list(s.a_i_hist for s in self.species_list if not s.is_ion), axis=1),
                "b_i_hist": np.concatenate(list(s.b_i_hist for s in self.species_list if not s.is_ion), axis=1),
                "a_0_hist": np.concatenate(list(s.a_0_hist for s in self.species_list if not s.is_ion)),
            }
            return history

    def store_current_step(self):
        """Store current particle properties in the history arrays."""
        for s in self.species_list:
            if "r" in self.diags or s.store_history:
                s.r_hist[-1 - s.i_push] = s.r
            if "z" in self.diags:
                s.xi_hist[-1 - s.i_push] = s.xi_current
            if "pr" in self.diags:
                s.pr_hist[-1 - s.i_push] = s.pr
            if "pz" in self.diags:
                s.pz_hist[-1 - s.i_push] = s.pz
            if "w" in self.diags:
                s.w_hist[-1 - s.i_push] = s.rho
            if "r_to_x" in self.diags:
                s.r_to_x_hist[-1 - s.i_push] = s.r_to_x
            if "id" in self.diags:
                s.id_hist[-1 - s.i_push] = s.id
            if s.store_history and not s.is_ion:
                s.a_0_hist[-1 - s.i_push] = s.a_0[0]

    def _allocate_field_arrays(self):
        """Allocate arrays for the fields experienced by the particles.

        In order to evolve the particles to the next longitudinal position,
        it is necessary to know the fields that they are experiencing. These
        arrays are used for storing the value of these fields at the location
        of each particle.
        """
        for s in self.species_list:
            if s.store_history:
                if not s.is_ion:
                    s.a_i = s.a_i_hist[-1]
                    s.b_i = s.b_i_hist[-1]
                else:
                    s.a_i = np.zeros((0))
                    s.b_i = np.zeros((0))
                s.sum_1 = s.sum_1_hist[-1]
                s.sum_2 = s.sum_2_hist[-1]
                s.rho = s.w_hist[-1]
                s.log_r = s.log_r_hist[-1]
            else:
                if not s.is_ion:
                    s.a_i = np.zeros(s.num_particles)
                    s.b_i = np.zeros(s.num_particles)
                else:
                    s.a_i = np.zeros((0))
                    s.b_i = np.zeros((0))
                s.sum_1 = np.zeros(s.num_particles + 1)
                s.sum_2 = np.zeros(s.num_particles + 1)
                s.rho = np.zeros(s.num_particles)
                s.log_r = np.zeros(s.num_particles)

            s.a2 = np.zeros(s.num_particles)
            s.nabla_a2 = np.zeros(s.num_particles)
            s.b_t_0 = np.zeros(s.num_particles)
            s.b_t = np.zeros(s.num_particles)
            s.psi = np.zeros(s.num_particles)
            s.dr_psi = np.zeros(s.num_particles)
            s.dxi_psi = np.zeros(s.num_particles)
            s.chi = np.zeros(s.num_particles)
            s.sum_3 = np.zeros(s.num_particles + 1)

            if not s.is_ion:
                s.a_0 = np.zeros(1)
                s.A = np.zeros(s.num_particles)
                s.B = np.zeros(s.num_particles)
                s.C = np.zeros(s.num_particles)
                s.K = np.zeros(s.num_particles)
                s.U = np.zeros(s.num_particles)
            else:
                s.a_0 = np.zeros(0)
                s.A = np.zeros(0)
                s.B = np.zeros(0)
                s.C = np.zeros(0)
                s.K = np.zeros(0)
                s.U = np.zeros(0)

    def _allocate_ab2_arrays(self):
        """Allocate the arrays needed for the 2nd order Adams-Bashforth pusher.

        The AB2 pusher needs the derivatives of r and pr for each particle
        at the last 2 plasma slices. This method allocates the arrays that will
        store these derivatives.
        """
        for s in self.species_list:
            if s.do_push:
                s.dr = np.zeros((2, s.num_particles))
                s.dpr = np.zeros((2, s.num_particles))
            else:
                s.dr = np.zeros((0, 0))
                s.dpr = np.zeros((0, 0))
