## Not reliable
import numpy as np

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
        self.d_t = 11.5
        azimuth_deg = 30.0
        d2 = 10.0
        self.x_rx = d2 * np.cos(np.radians(azimuth_deg))
        self.y_rx = d2 * np.sin(np.radians(azimuth_deg))
        self.z_rx = 0.0
        self.theta_t_deg = 0
        self.phi_t_deg = 0

def db2lin(db):
    return 10 ** (db / 10)

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
            dot = k * np.dot(kt + kr, [x, y, 0])
            phase = (2 * PI / p.lambda_) * (delta1 * x + delta2 * y)
            phi_shift = mod2pi(phase)
            binary_phase = PI if binary[n][m] == 1 else 0.0
            total_field += np.exp(1j * (dot - binary_phase))

    Gt = db2lin(p.Gt_dB)
    Gr = db2lin(p.Gr_dB)
    G = db2lin(p.G_dB)
    prefactor = (p.Pt * Gt * Gr * G * p.dx * p.dy * p.lambda_ ** 2 * p.A ** 2) / (64 * PI ** 3 * p.d_t ** 2 * d_r ** 2)
    return prefactor * abs(total_field) ** 2

def pattern_from_columns(bitmask, M, N):
    binary = np.zeros((N, M), dtype=int)
    for m in range(M):
        on = (bitmask >> m) & 1
        binary[:, m] = on
    return binary

if __name__ == "__main__":
    p = Params()
    best = {"pattern": 0, "power": -np.inf}
    worst = {"pattern": 0, "power": np.inf}

    for i in range(2 ** p.M):
        binary = pattern_from_columns(i, p.M, p.N)
        Pr = compute_farfield_power(binary, p)
        if Pr > best["power"]:
            best = {"pattern": i, "power": Pr}
        if Pr < worst["power"]:
            worst = {"pattern": i, "power": Pr}

    def format_columns(mask, M):
        return [(mask >> m) & 1 for m in range(M)]

    print("\n=== BEST FAR-FIELD PATTERN ===")
    print(f"Bitmask: 0x{best['pattern']:04X}")
    print("Columns:", format_columns(best["pattern"], p.M))
    print(f"Max Power: {best['power']:.3e} W ({10 * np.log10(best['power']):.2f} dBW)")

    print("\n=== WORST FAR-FIELD PATTERN ===")
    print(f"Bitmask: 0x{worst['pattern']:04X}")
    print("Columns:", format_columns(worst["pattern"], p.M))
    print(f"Min Power: {worst['power']:.3e} W ({10 * np.log10(worst['power']):.2f} dBW)")
