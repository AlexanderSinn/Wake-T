// cppimport
<%
setup_pybind11(cfg)
cfg['extra_compile_args'] = ['-fopenmp', '-O3', '-ffast-math', '-march=native'] if OSError != "nt" else ['/openmp']
cfg['extra_link_args'] = ['-lgomp'] if OSError != "nt" else []
%>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>
#include <complex>
#include <vector>
#include <atomic>
#include <omp.h>
#include <thread>
#include <chrono>

namespace py = pybind11;

template<class T>
struct FArray2D {
    T* ptr = nullptr;
    long long int stride = 0;

    FArray2D () = default;

    FArray2D (py::array_t<T, py::array::c_style>& arr) {
        py::buffer_info arr_info = arr.request();
        ptr = static_cast<T*>(arr_info.ptr);
        stride = arr_info.strides[0] / arr_info.itemsize;
    }

    T& operator()(int i, int j) const {
        return ptr[i + j * stride];
    }
};

template<class T>
FArray2D(py::array_t<T, py::array::c_style>& arr) -> FArray2D<T>;

using cplx = std::complex<double>;


void TDMA(
    long long int n,
    const double* a,
    const cplx* b,
    const double* c,
    const cplx* d,
    cplx* p
)
{
    std::vector<cplx> w(n-1);
    std::vector<cplx> g(n);

    w[0] = c[0] / b[0];
    g[0] = d[0] / b[0];

    for (int i=1; i<n-1; ++i) {
        double a_im1 = a[i-1];
        cplx inv_coef = 1.0 / (b[i] - a_im1 * w[i - 1]);
        g[i] = (d[i] - a_im1 * g[i - 1]) * inv_coef;
        w[i] = c[i] * inv_coef;
    }

    g[n-1] = (d[n-1] - a[n-3] * g[n-2]) / (b[n-1] - a[n-3] * w[n-3]);

    p[n-1] = g[n-1];
    for (int i=n-1; i>0; --i) {
        p[i - 1] = g[i - 1] - w[i - 1] * p[i];
    }
}



void parallel_solver(
    py::array_t<cplx, py::array::c_style> a,
    py::array_t<cplx, py::array::c_style> a_old,
    py::array_t<double, py::array::c_style> chi,
    double k0,
    double kp,
    double zmin,
    double zmax,
    long long int nz,
    double rmax,
    long long int nr,
    double dt,
    long long int nt,
    bool use_phase,
    int num_threads
)
{
    FArray2D a_arr{a};
    FArray2D a_old_arr{a_old};
    FArray2D chi_arr{chi};

    py::gil_scoped_release release;

    omp_set_num_threads(num_threads);

    constexpr double ct_c = 299792458.;

    double dz = (zmax - zmin) * kp / (nz - 1);
    double dr = rmax * kp / nr;
    dt = dt * ct_c * kp;

    double inv_dt = 1 / dt;
    double inv_dr = 1 / dr;
    double inv_dz = 1 / dz;
    double inv_dzdt = inv_dt * inv_dz;
    double k0_over_kp = k0 / kp;

    cplx C_minus = (
        -2.0 * inv_dr * inv_dr* 0.5
        - cplx(0, 1) * k0_over_kp * inv_dt
        + 1.5 * inv_dzdt
        - inv_dt*inv_dt
    );
    cplx C_plus = (
        -2.0 * inv_dr * inv_dr * 0.5
        + cplx(0, 1) * k0_over_kp * inv_dt
        - 1.5 * inv_dzdt
        - inv_dt*inv_dt
    );

    std::vector<double> L_minus_over_2(nr);
    std::vector<double> L_plus_over_2(nr);

    for (int i=0; i<nr; ++i) {
        double L_base = 1.0 / (2.0 * (i + 0.5));
        L_minus_over_2[i] = (1.0 - L_base) * inv_dr*inv_dr* 0.5;
        L_plus_over_2[i] = (1.0 + L_base) * inv_dr*inv_dr * 0.5;
    }

    std::vector<std::atomic<int>> current_time_step (nz+2);
    for (auto& step : current_time_step) {
        step = 0;
    }

    auto recv = [&](int step, int slice){
        while (step != current_time_step[slice].load(std::memory_order_acquire)) {
            using namespace std::chrono_literals;
            std::this_thread::sleep_for(10us);
        }
    };
    auto send = [&](int step, int slice){
        current_time_step[slice].store(step, std::memory_order_release);
    };

// #pragma omp parallel
//     {
// #pragma omp critical
//         {
//             std::cout << omp_get_thread_num() << " " << omp_get_num_threads() << std::endl;
//         }
//     }
#pragma omp parallel
    {
        const int ithread = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();

        for (int n=ithread; n<nt; n += nthreads) {
            std::vector<cplx> rhs(nr);
            std::vector<cplx> a_new_jp1(nr, 0);
            std::vector<cplx> a_new_jp2(nr, 0);
            std::vector<cplx> d_main(nr);

            recv(n, nz+1);
            recv(n, nz);

            for (int j=nz-1; j>=0; --j) {

                recv(n, j);

// #pragma omp critical
//                 {
//                     std::cout << "thread " << ithread << " step " << n << " slice " << j << std::endl;
//                 }

                for (int k=0; k<nr; ++k) {

                    cplx rhs_k = (
                        -2 * inv_dt * inv_dt * a_arr(k, j)
                        - ((C_minus - chi_arr(k, j) * 0.5) * a_old_arr(k, j))
                        - (
                            2
                            * inv_dzdt
                            * (a_new_jp1[k] - a_old_arr(k, j + 1))
                        )
                        + (
                            0.5
                            * inv_dzdt
                            * (a_new_jp2[k] - a_old_arr(k, j + 2))
                        )
                    );

                    if (k > 0) {
                        rhs_k -= L_minus_over_2[k] * a_old_arr(k - 1, j);
                    }
                    if (k + 1 < nr) {
                        rhs_k -= L_plus_over_2[k] * a_old_arr(k + 1, j);
                    }
                    rhs[k] = rhs_k;

                    d_main[k] = C_plus - chi_arr(k, j) * 0.5;

                }

                for (int k=0; k<nr; ++k) {
                    a_old_arr(k, j+2) = a_arr(k, j+2);
                    a_arr(k, j+2) = a_new_jp2[k];
                }

                send(n+1, j+2);

                std::swap(a_new_jp2, a_new_jp1);

                TDMA(nr, L_minus_over_2.data()+1, d_main.data(), L_plus_over_2.data(),
                    rhs.data(), a_new_jp1.data());

            }

            for (int k=0; k<nr; ++k) {
                a_old_arr(k, 0) = a_arr(k, 0);
                a_old_arr(k, 1) = a_arr(k, 1);
                a_arr(k, 0) = a_new_jp1[k];
                a_arr(k, 1) = a_new_jp2[k];
            }

            send(n+1, 1);
            send(n+1, 0);
        }
    }
}

PYBIND11_MODULE(parallel_solver, m) {
    m.def("parallel_solver", &parallel_solver, "doc");
}
