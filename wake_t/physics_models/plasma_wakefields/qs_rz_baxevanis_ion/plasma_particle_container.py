"""Contains the definition of the `PlasmaParticleContainerPy` class."""

import numba


class PlasmaParticleContainerPy:
    # Particle properties
    r: numba.float64[::1]
    log_r: numba.float64[::1]
    dr_p: numba.float64[::1]
    pr: numba.float64[::1]
    pz: numba.float64[::1]
    gamma: numba.float64[::1]
    w: numba.float64[::1]
    w_center: numba.float64[::1]
    r_to_x: numba.int32[::1]
    id: numba.int32[::1]
    mass: float
    charge: float

    # Fields at particles
    psi: numba.float64[::1]
    dr_psi: numba.float64[::1]
    dxi_psi: numba.float64[::1]
    b_t: numba.float64[::1]
    b_t_0: numba.float64[::1]
    nabla_a2: numba.float64[::1]
    a2: numba.float64[::1]
    rho: numba.float64[::1]
    chi: numba.float64[::1]

    # Temp arrays for field solvers
    a_i: numba.float64[::1]
    b_i: numba.float64[::1]
    sum_1: numba.float64[::1]
    sum_2: numba.float64[::1]
    sum_3: numba.float64[::1]
    a_0: numba.float64[::1]
    A: numba.float64[::1]
    B: numba.float64[::1]
    C: numba.float64[::1]
    K: numba.float64[::1]
    U: numba.float64[::1]

    # AB2 pusher
    dr: numba.float64[:, ::1]
    dpr: numba.float64[:, ::1]

    # History arrays
    r_hist: numba.float64[:, ::1]
    log_r_hist: numba.float64[:, ::1]
    xi_hist: numba.float64[:, ::1]
    pr_hist: numba.float64[:, ::1]
    pz_hist: numba.float64[:, ::1]
    w_hist: numba.float64[:, ::1]
    r_to_x_hist: numba.int32[:, ::1]
    id_hist: numba.int32[:, ::1]
    sum_1_hist: numba.float64[:, ::1]
    sum_2_hist: numba.float64[:, ::1]
    a_i_hist: numba.float64[:, ::1]
    b_i_hist: numba.float64[:, ::1]
    a_0_hist: numba.float64[::1]
    i_push: int
    xi_current: float

    # Flags
    num_particles: int
    is_ion: bool
    do_push: bool
    store_history: bool

    def __init__(self, serialized_list=None):
        if serialized_list is not None:
            (
                self.r,
                self.log_r,
                self.dr_p,
                self.pr,
                self.pz,
                self.gamma,
                self.w,
                self.w_center,
                self.r_to_x,
                self.id,
                self.mass,
                self.charge,
                self.psi,
                self.dr_psi,
                self.dxi_psi,
                self.b_t,
                self.b_t_0,
                self.nabla_a2,
                self.a2,
                self.rho,
                self.chi,
                self.a_i,
                self.b_i,
                self.sum_1,
                self.sum_2,
                self.sum_3,
                self.a_0,
                self.A,
                self.B,
                self.C,
                self.K,
                self.U,
                self.dr,
                self.dpr,
                self.r_hist,
                self.log_r_hist,
                self.xi_hist,
                self.pr_hist,
                self.pz_hist,
                self.w_hist,
                self.r_to_x_hist,
                self.id_hist,
                self.sum_1_hist,
                self.sum_2_hist,
                self.a_i_hist,
                self.b_i_hist,
                self.a_0_hist,
                self.i_push,
                self.xi_current,
                self.num_particles,
                self.is_ion,
                self.do_push,
                self.store_history,
            ) = serialized_list

    def serialize(self):
        return (
            self.r,
            self.log_r,
            self.dr_p,
            self.pr,
            self.pz,
            self.gamma,
            self.w,
            self.w_center,
            self.r_to_x,
            self.id,
            self.mass,
            self.charge,
            self.psi,
            self.dr_psi,
            self.dxi_psi,
            self.b_t,
            self.b_t_0,
            self.nabla_a2,
            self.a2,
            self.rho,
            self.chi,
            self.a_i,
            self.b_i,
            self.sum_1,
            self.sum_2,
            self.sum_3,
            self.a_0,
            self.A,
            self.B,
            self.C,
            self.K,
            self.U,
            self.dr,
            self.dpr,
            self.r_hist,
            self.log_r_hist,
            self.xi_hist,
            self.pr_hist,
            self.pz_hist,
            self.w_hist,
            self.r_to_x_hist,
            self.id_hist,
            self.sum_1_hist,
            self.sum_2_hist,
            self.a_i_hist,
            self.b_i_hist,
            self.a_0_hist,
            self.i_push,
            self.xi_current,
            self.num_particles,
            self.is_ion,
            self.do_push,
            self.store_history,
        )


# PlasmaParticleContainer can be used inside numba jit functions
# PlasmaParticleContainerPy can be used outside numba jit functions.
# Both can not be used as parameters that pass from outside to inside a numba jit function
# because that would break function caching
PlasmaParticleContainer = numba.experimental.jitclass(PlasmaParticleContainerPy)
