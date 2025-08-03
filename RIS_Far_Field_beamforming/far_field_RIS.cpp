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
    double Gt_dB, Gr_dB, G_dB;
    double A;
    double d_t;
    double x_rx, y_rx, z_rx;
    double theta_t_deg, phi_t_deg;
};

double db2lin(double db) {
    return std::pow(10.0, db / 10.0);
}

double deg2rad(double deg) {
    return deg * PI / 180.0;
}

double mod2pi(double x) {
    return std::fmod(std::fmod(x, 2 * PI) + 2 * PI, 2 * PI);
}

std::complex<double> jexp(double phase) {
    return std::exp(std::complex<double>(0, phase));
}

double compute_farfield_power(const std::vector<std::vector<int>>& binary, const Params& p) {
    std::complex<double> sum = 0.0;
    double k = 2 * PI / p.lambda;

    std::vector<double> kt = {0, 0, 1};
    std::vector<double> kr = {p.x_rx, p.y_rx, p.z_rx};
    double d_r = std::sqrt(p.x_rx * p.x_rx + p.y_rx * p.y_rx + p.z_rx * p.z_rx);
    for (int i = 0; i < 3; ++i) kr[i] /= d_r;

    for (int n = 0; n < p.N; ++n) {
        for (int m = 0; m < p.M; ++m) {
            double x = (m + 0.5) * p.dx;
            double y = (n + 0.5) * p.dy;
            double z = 0;

            double dot = k * ((kt[0] + kr[0]) * x + (kt[1] + kr[1]) * y + (kt[2] + kr[2]) * z);
            double theta_d = std::asin(std::sqrt(p.x_rx * p.x_rx + p.y_rx * p.y_rx) / d_r);
            double phi_d = std::atan2(p.y_rx, p.x_rx);
            double theta_t = deg2rad(p.theta_t_deg);
            double phi_t = deg2rad(p.phi_t_deg);
            double delta1 = -std::sin(theta_t) * std::cos(phi_t) - std::sin(theta_d) * std::cos(phi_d);
            double delta2 = -std::sin(theta_t) * std::sin(phi_t) - std::sin(theta_d) * std::sin(phi_d);
            double phase = (2 * PI / p.lambda) * (delta1 * x + delta2 * y);
            double phi_shift = mod2pi(phase);

            double binary_phase = binary[n][m] == 1 ? PI : 0.0;
            sum += std::exp(std::complex<double>(0, dot - binary_phase));
        }
    }

    double Gt = db2lin(p.Gt_dB);
    double Gr = db2lin(p.Gr_dB);
    double G = db2lin(p.G_dB);
    double prefactor = (p.Pt * Gt * Gr * G * p.dx * p.dy * p.lambda * p.lambda * p.A * p.A) /
                       (64.0 * PI * PI * PI * p.d_t * p.d_t * d_r * d_r);
    return prefactor * std::norm(sum);
}

int main() {
    Params p;
    p.lambda = 0.085;
    p.dx = p.dy = 0.015;
    p.M = p.N = 16;
    p.Pt = 1e-3;
    p.Gt_dB = 60;
    p.Gr_dB = 34;
    p.G_dB = 0;
    p.A = 1.0;
    p.d_t = 3.5;

    double d2 = 10.0;
    double azimuth_deg = 30.0;
    double azimuth_rad = deg2rad(azimuth_deg);
    p.x_rx = d2 * std::cos(azimuth_rad);
    p.y_rx = d2 * std::sin(azimuth_rad);
    p.z_rx = 0.0;
    p.theta_t_deg = 0;
    p.phi_t_deg = 0;

    double max_power = -1.0;
    double min_power = 1e9;
    uint16_t best_pattern = 0, worst_pattern = 0;

    #pragma omp parallel for
    for (uint64_t i = 0; i < 65536; ++i) {
        std::vector<std::vector<int>> binary(p.N, std::vector<int>(p.M, 0));
        for (int m = 0; m < p.M; ++m) {
            int on = (i >> m) & 1;
            for (int n = 0; n < p.N; ++n) binary[n][m] = on;
        }

        double Pr = compute_farfield_power(binary, p);
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

    std::cout << "\n=== BEST FAR-FIELD PATTERN ===\n";
    std::cout << "Bitmask: 0x" << std::hex << best_pattern << std::dec << "\nColumns: ";
    for (int m = 0; m < p.M; ++m) std::cout << ((best_pattern >> m) & 1) << " ";
    std::cout << "\nMax Power: " << std::scientific << max_power << " W ("
              << std::fixed << std::setprecision(2) << 10 * std::log10(max_power) << " dBW)\n";

    std::cout << "\n=== WORST FAR-FIELD PATTERN ===\n";
    std::cout << "Bitmask: 0x" << std::hex << worst_pattern << std::dec << "\nColumns: ";
    for (int m = 0; m < p.M; ++m) std::cout << ((worst_pattern >> m) & 1) << " ";
    std::cout << "\nMin Power: " << std::scientific << min_power << " W ("
              << std::fixed << std::setprecision(2) << 10 * std::log10(min_power) << " dBW)\n";

    return 0;
}
