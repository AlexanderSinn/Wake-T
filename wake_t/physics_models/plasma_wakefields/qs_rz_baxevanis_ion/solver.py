"""
This module implements the methods for calculating the plasma wakefields
using the 2D r-z reduced model from P. Baxevanis and G. Stupakov.

See https://journals.aps.org/prab/abstract/10.1103/PhysRevAccelBeams.21.071301
for the full details about this model.
"""

import numpy as np
import scipy.constants as ct
import aptools.plasma_accel.general_equations as ge
from numba.typed import List

import time

from .plasma_particles import (
    pp_initialize,
    pp_sort,
    pp_gather_laser_sources,
    pp_gather_bunch_sources,
    pp_calculate_fields,
    pp_calculate_psi_at_grid,
    pp_calculate_b_theta_at_grid,
    pp_calculate_weights,
    pp_deposit_rho,
    pp_deposit_chi,
    pp_store_current_step,
    pp_evolve,
    pp_get_history,
)
from .utils import longitudinal_gradient, radial_gradient

from wake_t.utilities.numba import njit_serial

from .plasma_particle_container import PlasmaParticleContainer


time_sort = 0.
time_gather_laser = 0.
time_gather_bunch = 0.
time_calc_fields = 0.
time_calc_psi_grid = 0.
time_calc_bt_grid = 0.
time_depos_rho = 0.
time_calc_weights = 0.
time_depos_chi = 0.
time_store_hist = 0.
time_evolve = 0.

# @njit_serial
def evolve_one_step(
    pp_serialized_list,
    n_xi,
    n_r,
    dxi,
    dr,
    r_fld,
    has_laser_source,
    laser_a2,
    nabla_a2,
    has_beam_source,
    bunch_source_arrays,
    bunch_source_xi_indices,
    bunch_source_metadata,
    max_gamma,
    psi,
    B_t,
    shape,
    calculate_rho,
    rho,
    rho_e,
    rho_i,
    chi,
    store_plasma_history,
    particle_diags,
):
    """
    Compute the wakefield by evolving the plasma over all zeta slices.
    For performance reasons, this is done in a single JIT-compiled
    function to minimize the number of Python-to-Numba function calls.

    pp_serialized_list is passed in as a Tuple[Tuple[np.ndarray]]
    instead of a class so that Numba can cache the JIT compiled function.

    See calculate_wakefields() for parametes.
    """
    global time_sort, time_gather_laser, time_gather_bunch, time_calc_fields, time_calc_psi_grid, time_calc_bt_grid, time_depos_rho, time_calc_weights, time_depos_chi, time_store_hist, time_evolve
    ions_computed = False
    pp_species_list = List(
        PlasmaParticleContainer(species) for species in pp_serialized_list
    )

    # Evolve plasma from right to left and calculate psi, b_t_bar, rho and
    # chi on a grid.
    for step in range(n_xi):
        slice_i = n_xi - step - 1
        time_sort -= time.time()
        pp_sort(pp_species_list)
        time_sort += time.time()

        if has_laser_source:
            time_gather_laser -= time.time()
            pp_gather_laser_sources(
                pp_species_list,
                laser_a2[slice_i + 2],
                nabla_a2[slice_i + 2],
                r_fld[0],
                r_fld[-1],
                dr,
            )
            time_gather_laser += time.time()
        if has_beam_source:
            time_gather_bunch -= time.time()
            pp_gather_bunch_sources(
                pp_species_list,
                bunch_source_arrays,
                bunch_source_xi_indices,
                bunch_source_metadata,
                slice_i,
            )
            time_gather_bunch += time.time()

        time_calc_fields -= time.time()
        pp_calculate_fields(pp_species_list, ions_computed, max_gamma)
        time_calc_fields += time.time()

        time_calc_psi_grid -= time.time()
        pp_calculate_psi_at_grid(pp_species_list, r_fld, psi[slice_i + 2, 2:-2])
        time_calc_psi_grid += time.time()

        time_calc_bt_grid -= time.time()
        pp_calculate_b_theta_at_grid(pp_species_list, r_fld, B_t[slice_i + 2, 2:-2])
        time_calc_bt_grid += time.time()

        if calculate_rho:
            time_depos_rho -= time.time()
            pp_deposit_rho(
                pp_species_list,
                ions_computed,
                shape,
                rho[slice_i + 2],
                rho_e[slice_i + 2],
                rho_i[slice_i + 2],
                r_fld,
                n_r,
                dr,
            )
            time_depos_rho += time.time()
        elif "w" in particle_diags:
            time_calc_weights -= time.time()
            pp_calculate_weights(pp_species_list, ions_computed)
            time_calc_weights += time.time()
        if has_laser_source:
            time_depos_chi -= time.time()
            pp_deposit_chi(pp_species_list, shape, chi[slice_i + 2], r_fld, n_r, dr)
            time_depos_chi += time.time()

        ions_computed = True

        if store_plasma_history:
            time_store_hist -= time.time()
            pp_store_current_step(pp_species_list, particle_diags)
            time_store_hist += time.time()

        if slice_i > 0:
            time_evolve -= time.time()
            pp_evolve(pp_species_list, dxi)
            time_evolve += time.time()


def calculate_wakefields(
    laser_a2,
    r_max,
    xi_min,
    xi_max,
    n_r,
    n_xi,
    ppc,
    n_p,
    r_max_plasma=None,
    radial_density=None,
    p_shape="cubic",
    max_gamma=10.0,
    plasma_pusher="ab2",
    ion_motion=False,
    ion_mass=ct.m_p,
    free_electrons_per_ion=1,
    bunch_source_arrays=[],
    bunch_source_xi_indices=[],
    bunch_source_metadata=[],
    store_plasma_history=False,
    calculate_rho=True,
    particle_diags=[],
    fld_arrays=[],
):
    """
    Calculate the plasma wakefields generated by the given laser pulse and
    electron beam in the specified grid points.

    Parameters
    ----------
    laser_a2 : ndarray
        A (nz x nr) array containing the square of the laser envelope.
    r_max : float
        Maximum radial position up to which plasma wakefield will be
        calculated.
    xi_min : float
        Minimum longitudinal (speed of light frame) position up to which
        plasma wakefield will be calculated.
    xi_max : float
        Maximum longitudinal (speed of light frame) position up to which
        plasma wakefield will be calculated.
    n_r : int
        Number of grid elements along r in which to calculate the wakefields.
    n_xi : int
        Number of grid elements along xi in which to calculate the wakefields.
    ppc : array_like
        see Quasistatic2DWakefieldIons.
    n_p : float
        On-axis plasma density in units of m^{-3}.
    r_max_plasma : float
        Maximum radial extension of the plasma column. If `None`, the plasma
        extends up to the `r_max` boundary of the simulation box.
    radial_density : callable
        Function defining the radial density profile.
    p_shape : str
        Particle shape to be used for the beam charge deposition. Possible
        values are 'linear' or 'cubic'.
    max_gamma : float
        Plasma particles whose `gamma` exceeds `max_gamma` are considered to
        violate the quasistatic condition and are put at rest (i.e.,
        `gamma=1.`, `pr=pz=0.`).
    plasma_pusher : str
        Numerical pusher for the plasma particles. Possible values are `'ab2'`.
    ion_motion : bool, optional
        Whether to allow the plasma ions to move. By default, False.
    ion_mass : float, optional
        Mass of the plasma ions. By default, the mass of a proton.
    free_electrons_per_ion : int, optional
        Number of free electrons per ion. The ion charge is adjusted
        accordingly to maintain a quasi-neutral plasma (i.e.,
        ion charge = e * free_electrons_per_ion). By default, 1.
    bunch_source_arrays : list, optional
        List containing the array from which the bunch source terms (the
        azimuthal magnetic field) will be gathered. It can be a single
        array for the whole domain, or one array per bunch when using
        adaptive grids.
    bunch_source_xi_indices : list, optional
        List containing 1d arrays that with the indices of the longitudinal
        plasma slices that can gather from them. This is needed because the
        adaptive grids might not extend the whole longitudinal domain of the
        plasma, so the plasma slices should only try to gather the source terms
        if they are available at the current slice.
    bunch_source_metadata : list, optional
        Metadata of each bunch source array.
    store_plasma_history : bool, optional
        Whether to store the plasma particle evolution. This might be needed
        for diagnostics or the use of adaptive grids. By default, False.
    calculate_rho : bool, optional
        Whether to deposit the plasma density. This might be needed for
        diagnostics. By default, False.
    particle_diags : list, optional
        List of particle quantities to save to diagnostics.
    fld_arrays : list, optional
        List of all the fields.
    """
    global time_sort, time_gather_laser, time_gather_bunch, time_calc_fields, time_calc_psi_grid, time_calc_bt_grid, time_depos_rho, time_calc_weights, time_depos_chi, time_store_hist, time_evolve
    rho, rho_e, rho_i, chi, E_r, E_z, B_t, xi_fld, r_fld = fld_arrays

    # Convert to normalized units.
    s_d = ge.plasma_skin_depth(n_p * 1e-6)
    r_max = r_max / s_d
    xi_min = xi_min / s_d
    xi_max = xi_max / s_d
    dr = r_max / n_r
    dxi = (xi_max - xi_min) / (n_xi - 1)
    ppc = ppc.copy()
    ppc[:, 0] /= s_d
    r_max_plasma = r_max_plasma / s_d

    def radial_density_normalized(r):
        return radial_density(r * s_d) / n_p

    # Field node coordinates.
    r_fld = r_fld / s_d
    xi_fld = xi_fld / s_d

    # Initialize field arrays, including guard cells.
    nabla_a2 = np.zeros((n_xi + 4, n_r + 4))
    psi = np.zeros((n_xi + 4, n_r + 4))

    # Laser source.
    has_laser_source = laser_a2 is not None
    if has_laser_source:
        radial_gradient(laser_a2[2:-2, 2:-2], dr, nabla_a2[2:-2, 2:-2])
    else:
        # need to set the dtype for JIT
        laser_a2 = np.zeros((0, 0))
        nabla_a2 = np.zeros((0, 0))

    has_beam_source = len(bunch_source_arrays) > 0
    if not has_beam_source:
        # need to set the dtype for JIT
        bunch_source_arrays.append(np.zeros((0, 0)))
        bunch_source_xi_indices.append(np.zeros(0, dtype=np.int64))
        bunch_source_metadata.append(np.zeros(0))

    if len(particle_diags) == 0:
        # need to set the type for JIT
        particle_diags = ["none"]

    # Calculate plasma response (including density, susceptibility, potential
    # and magnetic field)

    # Initialize plasma particles.
    # Set parameters for electron and ion species in normalized units
    init_list = [
        {
            "charge": free_electrons_per_ion,
            "mass": free_electrons_per_ion,
            "is_ion": False,
        },
        {
            "charge": -free_electrons_per_ion,
            "mass": ion_mass / ct.m_e,
            "is_ion": True,
        },
    ]

    species_list = pp_initialize(
        init_list,
        n_xi,
        ppc,
        dr,
        radial_density_normalized,
        ion_motion,
        store_plasma_history,
        plasma_pusher,
    )

    time_full = -time.time()
    time_sort = 0.
    time_gather_laser = 0.
    time_gather_bunch = 0.
    time_calc_fields = 0.
    time_calc_psi_grid = 0.
    time_calc_bt_grid = 0.
    time_depos_rho = 0.
    time_calc_weights = 0.
    time_depos_chi = 0.
    time_store_hist = 0.
    time_evolve = 0.

    evolve_one_step(
        List(s.serialize() for s in species_list),
        n_xi,
        n_r,
        dxi,
        dr,
        r_fld,
        has_laser_source,
        laser_a2,
        nabla_a2,
        has_beam_source,
        List(bunch_source_arrays),
        List(bunch_source_xi_indices),
        List(bunch_source_metadata),
        max_gamma,
        psi,
        B_t,
        p_shape,
        calculate_rho,
        rho,
        rho_e,
        rho_i,
        chi,
        store_plasma_history,
        List(particle_diags),
    )

    time_full += time.time()

    print(f"full:               {1000*(time_full):.02f} ms")
    print(f"time_sort:          {1000*(time_sort):.02f} ms")
    print(f"time_gather_laser:  {1000*(time_gather_laser):.02f} ms")
    print(f"time_gather_bunch:  {1000*(time_gather_bunch):.02f} ms")
    print(f"time_calc_fields:   {1000*(time_calc_fields):.02f} ms")
    print(f"time_calc_psi_grid: {1000*(time_calc_psi_grid):.02f} ms")
    print(f"time_calc_bt_grid:  {1000*(time_calc_bt_grid):.02f} ms")
    print(f"time_depos_rho:     {1000*(time_depos_rho):.02f} ms")
    print(f"time_calc_weights:  {1000*(time_calc_weights):.02f} ms")
    print(f"time_depos_chi:     {1000*(time_depos_chi):.02f} ms")
    print(f"time_store_hist:    {1000*(time_store_hist):.02f} ms")
    print(f"time_evolve:        {1000*(time_evolve):.02f} ms")

    # Calculate derived fields (E_z, W_r, and E_r).
    E_0 = ge.plasma_cold_non_relativisct_wave_breaking_field(n_p * 1e-6)
    longitudinal_gradient(psi[2:-2, 2:-2], dxi, E_z[2:-2, 2:-2])
    radial_gradient(psi[2:-2, 2:-2], dr, E_r[2:-2, 2:-2])
    E_r -= B_t
    E_z *= -E_0
    E_r *= -E_0
    # B_t[:] = (b_t_bar + b_t_beam) * E_0 / ct.c
    B_t *= E_0 / ct.c
    return pp_get_history(species_list, store_plasma_history)
