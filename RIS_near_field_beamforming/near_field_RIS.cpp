#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <iomanip>
#include <cstdint>
#include <omp.h>

#define PI 3.14159265358979323846

struct Params {
    double lambda;
    double dx, dy;
    int M, N;
    double Pt;
    double Gt_dB, Gr_dB;
    double G;
    double A;
    double d1;
    double x_rx, y_rx, z_rx;
};

double db2lin(double db) {
    return std::pow(10.0, db / 10.0);
}

double distance3D(double x1, double y1, double z1,
                  double x2, double y2, double z2) {
    return std::sqrt((x1 - x2)*(x1 - x2) +
                     (y1 - y2)*(y1 - y2) +
                     (z1 - z2)*(z1 - z2));
}

// Compute Pr for a given binary pattern
double compute_Prx(const std::vector<std::vector<int>>& binary, const Params& p) {
    std::complex<double> sum = 0.0;
    double k = 2.0 * PI / p.lambda;
    double z_tx = -p.d1;

    for (int n = 0; n < p.N; ++n) {
        for (int m = 0; m < p.M; ++m) {
            double x_nm = (m - (p.M - 1) / 2.0) * p.dx;
            double y_nm = (n - (p.N - 1) / 2.0) * p.dy;
            double z_nm = 0.0;

            double rt = distance3D(x_nm, y_nm, z_nm, 0, 0, z_tx);
            double rr = distance3D(x_nm, y_nm, z_nm, p.x_rx, p.y_rx, p.z_rx);
            double total_path = rt + rr;

            double phase_shift = binary[n][m] == 1 ? PI : 0.0;
            std::complex<double> exp_term = std::exp(std::complex<double>(0, -1) * (k * total_path - phase_shift));
            sum += exp_term / (rt * rr);
        }
    }

    double Gt = db2lin(p.Gt_dB);
    double Gr = db2lin(p.Gr_dB);
    double prefactor = (p.Pt * Gt * Gr * p.G * p.dx * p.dy * p.lambda * p.lambda * p.A * p.A) / (64.0 * std::pow(PI, 3));
    return prefactor * std::norm(sum);
}

int main() {
    Params p;

    // System configuration
    p.lambda = 0.085;
    p.dx = p.dy = 0.015;
    p.M = p.N = 16;
    p.Pt = 1e-3;
    p.Gt_dB = 60;
    p.Gr_dB = 34;
    p.G = 8.0;
    p.A = 1.0;
    p.d1 = 3.5;

    // Target Rx location
    double d2 = 1.0;
    double azimuth_deg = 30.0;
    double azimuth_rad = azimuth_deg * PI / 180.0;
    p.x_rx = d2 * std::cos(azimuth_rad);
    p.y_rx = d2 * std::sin(azimuth_rad);
    p.z_rx = 0.0;

    uint64_t total_patterns = 65536;
    double max_power = -1.0;
    double min_power = 1e9;
    uint16_t best_pattern = 0;
    uint16_t worst_pattern = 0;

    #pragma omp parallel for
    for (uint64_t i = 0; i < total_patterns; ++i) {
        std::vector<std::vector<int>> binary(p.N, std::vector<int>(p.M, 0));

        // Build binary RIS matrix from column bitmask
        for (int m = 0; m < p.M; ++m) {
            int on = (i >> m) & 1;
            for (int n = 0; n < p.N; ++n) {
                binary[n][m] = on;
            }
        }

        double Pr = compute_Prx(binary, p);

        #pragma omp critical
        {
            if (Pr > max_power) {
                max_power = Pr;
                best_pattern = i;
            }
            if (Pr < min_power) {
                min_power = Pr;
                worst_pattern = i;
            }
        }
    }

    // Output best pattern
    std::cout << "\n=== BEST PATTERN ===\n";
    std::cout << "Bitmask: 0x" << std::hex << best_pattern << std::dec << "\nColumns: ";
    for (int m = 0; m < p.M; ++m) {
        std::cout << ((best_pattern >> m) & 1) << " ";
    }
    std::cout << "\nMax Power: " << std::scientific << max_power << " W\n";

    // Output worst pattern
    std::cout << "\n=== WORST PATTERN ===\n";
    std::cout << "Bitmask: 0x" << std::hex << worst_pattern << std::dec << "\nColumns: ";
    for (int m = 0; m < p.M; ++m) {
        std::cout << ((worst_pattern >> m) & 1) << " ";
    }
    std::cout << "\nMin Power: " << std::scientific << min_power << " W\n";

    return 0;
}
