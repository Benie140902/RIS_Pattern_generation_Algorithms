import numpy as np
import random

PI = np.pi

class Params:
    def __init__(self):
        self.lambda_ = 0.085
        self.dx = self.dy = 0.015
        self.M = self.N = 16
        self.Pt = 1e-3
        self.Gt_dB = 60
        self.Gr_dB = 34
        self.G_dB = 0
        self.A = 1.0
        self.d_t = 3.5
        azimuth_deg = 30.0
        d2 = 10.0
        self.x_rx = d2 * np.cos(np.radians(azimuth_deg))
        self.y_rx = d2 * np.sin(np.radians(azimuth_deg))
        self.z_rx = 0.0
        self.theta_t_deg = 0
        self.phi_t_deg = 0

def db2lin(db):
    return 10 ** (db / 10.0)

def mod2pi(x):
    return np.mod(np.mod(x, 2 * PI) + 2 * PI, 2 * PI)

def compute_farfield_power(binary, p):
    k = 2 * PI / p.lambda_
    total_field = 0 + 0j
    kt = np.array([0, 0, 1])
    r_rx = np.array([p.x_rx, p.y_rx, p.z_rx])
    d_r = np.linalg.norm(r_rx)
    kr = r_rx / d_r

    theta_t = np.radians(p.theta_t_deg)
    phi_t = np.radians(p.phi_t_deg)
    theta_d = np.arcsin(np.sqrt(p.x_rx ** 2 + p.y_rx ** 2) / d_r)
    phi_d = np.arctan2(p.y_rx, p.x_rx)
    delta1 = -np.sin(theta_t) * np.cos(phi_t) - np.sin(theta_d) * np.cos(phi_d)
    delta2 = -np.sin(theta_t) * np.sin(phi_t) - np.sin(theta_d) * np.sin(phi_d)

    for n in range(p.N):
        for m in range(p.M):
            x = (m + 0.5) * p.dx
            y = (n + 0.5) * p.dy
            r_nm = np.array([x, y, 0])
            dot = k * np.dot(kt + kr, r_nm)
            phase = (2 * PI / p.lambda_) * (delta1 * x + delta2 * y)
            phi_shift = mod2pi(phase)
            phase_shift = PI if binary[n][m] == 1 else 0.0
            total_field += np.exp(1j * (dot - phase_shift))

    Gt = db2lin(p.Gt_dB)
    Gr = db2lin(p.Gr_dB)
    G = db2lin(p.G_dB)
    prefactor = (p.Pt * Gt * Gr * G * p.dx * p.dy * p.lambda_ ** 2 * p.A ** 2) / (
        64 * PI ** 3 * p.d_t ** 2 * d_r ** 2)
    return prefactor * abs(total_field) ** 2

def generate_random_pattern(M, N):
    return np.random.randint(0, 2, (N, M))

if __name__ == "__main__":
    p = Params()
    best = {"pattern": None, "power": -np.inf}
    worst = {"pattern": None, "power": np.inf}

    num_trials = 100000  # Number of random patterns to evaluate

    for _ in range(num_trials):
        binary = generate_random_pattern(p.M, p.N)
        power = compute_farfield_power(binary, p)
        if power > best["power"]:
            best = {"pattern": binary.copy(), "power": power}
        if power < worst["power"]:
            worst = {"pattern": binary.copy(), "power": power}

    
    print("\nBest Pattern Binary Map:")
    for row in best['pattern']:
        print(" ".join(map(str, row)))

    print("\nWorst Pattern Binary Map:")
    for row in worst['pattern']:
        print(" ".join(map(str, row)))

    print("\n=== BEST PATTERN ===")
    print(f"Max Power: {best['power']:.3e} W ({10 * np.log10(best['power']):.2f} dBW)")

    print("\n=== WORST PATTERN ===")
    print(f"Min Power: {worst['power']:.3e} W ({10 * np.log10(worst['power']):.2f} dBW)")
