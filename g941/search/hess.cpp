// Hessian and gradient of the log-det barrier  Phi(mu, delta) = - sum_r c_r log det S_r,
//   S_r = U_r diag(mu) U_r^H + diag(delta),   column o of U_r = sum_p conj(psi_p(o)) e_{oa[o,p]},  psi_p(o) = w_r^{oi[o,p]}.
// d2 Phi / d y_i d y_j = sum_r c_r |u~_i^H G_r u~_j|^2  with G_r = S_r^(-1), u~ = columns of [U_r, I].
// build:  g++ -O3 -march=native -fopenmp -shared -fPIC -o exp/libhess.so exp/hess.cpp
#include <complex>
#include <vector>
#include <cstring>
#include <cmath>
#include <omp.h>
typedef std::complex<double> cd;

extern "C" {

// G: NH blocks (nrep x nrep, row-major, Hermitian).  H: (no+nrep)^2 output (full symmetric).  grad: no+nrep output.
// If G2 != nullptr the entries are  sum_r c_r Re( (u_i^H G_r u_j) * conj(u_i^H G2_r u_j) )  (cross term), grad is not touched.
void hess_sym(int nrep, int no, int NH, int q, const cd* G, const cd* G2, const int* oa, const int* oi,
              const double* cr, double* H, double* grad, int accumulate) {
    const long m = (long)no + nrep;
    std::vector<cd> ph((size_t)NH * q);
    for (int r = 0; r < NH; r++) for (int k = 0; k < q; k++) ph[(size_t)r * q + k] = std::polar(1.0, 2 * M_PI * r * k / q);
    if (!accumulate) {
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < m; i++) std::memset(H + i * m, 0, sizeof(double) * m);
        if (grad) for (long i = 0; i < m; i++) grad[i] = 0.0;
    } else {
        // only the upper triangle is accumulated; the mirror step at the end rewrites the lower one
    }
    for (int r = 0; r < NH; r++) {
        const cd* Gr = G + (size_t)r * nrep * nrep;
        const cd* Gs = G2 ? G2 + (size_t)r * nrep * nrep : nullptr;
        const cd* p = &ph[(size_t)r * q];
        std::vector<cd> psi((size_t)no * 3), cps((size_t)no * 3);
        for (long i = 0; i < (long)no * 3; i++) { psi[i] = p[oi[i]]; cps[i] = std::conj(psi[i]); }
        const double c = cr[r];
        #pragma omp parallel
        {
            std::vector<cd> xrow(nrep), xrow2(nrep);
            #pragma omp for schedule(dynamic, 16)
            for (int o = 0; o < no; o++) {
                const cd a0 = psi[3 * o], a1 = psi[3 * o + 1], a2 = psi[3 * o + 2];
                {
                    const cd* g0 = Gr + (size_t)oa[3 * o] * nrep; const cd* g1 = Gr + (size_t)oa[3 * o + 1] * nrep; const cd* g2 = Gr + (size_t)oa[3 * o + 2] * nrep;
                    for (int a = 0; a < nrep; a++) xrow[a] = a0 * g0[a] + a1 * g1[a] + a2 * g2[a];
                }
                if (Gs) {
                    const cd* g0 = Gs + (size_t)oa[3 * o] * nrep; const cd* g1 = Gs + (size_t)oa[3 * o + 1] * nrep; const cd* g2 = Gs + (size_t)oa[3 * o + 2] * nrep;
                    for (int a = 0; a < nrep; a++) xrow2[a] = a0 * g0[a] + a1 * g1[a] + a2 * g2[a];
                }
                double* Hrow = H + (size_t)o * m;
                if (!Gs) {
                    for (int o2 = o; o2 < no; o2++) {
                        const int* b = oa + 3 * o2;
                        cd qv = xrow[b[0]] * cps[3 * o2] + xrow[b[1]] * cps[3 * o2 + 1] + xrow[b[2]] * cps[3 * o2 + 2];
                        Hrow[o2] += c * std::norm(qv);
                        if (o2 == o && grad) grad[o] -= c * qv.real();
                    }
                    double* Hd = Hrow + no;
                    for (int a = 0; a < nrep; a++) Hd[a] += c * std::norm(xrow[a]);
                } else {
                    for (int o2 = o; o2 < no; o2++) {
                        const int* b = oa + 3 * o2;
                        cd qv = xrow[b[0]] * cps[3 * o2] + xrow[b[1]] * cps[3 * o2 + 1] + xrow[b[2]] * cps[3 * o2 + 2];
                        cd qw = xrow2[b[0]] * cps[3 * o2] + xrow2[b[1]] * cps[3 * o2 + 1] + xrow2[b[2]] * cps[3 * o2 + 2];
                        Hrow[o2] += c * (qv.real() * qw.real() + qv.imag() * qw.imag());
                    }
                    double* Hd = Hrow + no;
                    for (int a = 0; a < nrep; a++) Hd[a] += c * (xrow[a].real() * xrow2[a].real() + xrow[a].imag() * xrow2[a].imag());
                }
            }
            #pragma omp for schedule(static)
            for (int a = 0; a < nrep; a++) {
                double* Hrow = H + (size_t)(no + a) * m + no;
                const cd* g = Gr + (size_t)a * nrep;
                if (!Gs) {
                    for (int b = a; b < nrep; b++) Hrow[b] += c * std::norm(g[b]);
                } else {
                    const cd* g2 = Gs + (size_t)a * nrep;
                    for (int b = a; b < nrep; b++) Hrow[b] += c * (g[b].real() * g2[b].real() + g[b].imag() * g2[b].imag());
                }
            }
        }
        if (grad && !Gs) for (int a = 0; a < nrep; a++) grad[no + a] -= c * Gr[(size_t)a * nrep + a].real();
    }
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < m; i++) for (long j = 0; j < i; j++) H[i * m + j] = H[j * m + i];
}

}
