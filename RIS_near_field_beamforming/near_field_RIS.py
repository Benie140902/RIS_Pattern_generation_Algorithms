import numpy as np
import math
from concurrent.futures import ThreadPoolExecutor

# Constants
PI = np.pi

# Parameters
class Params:
    def __init__(self):
        self.lambda_ = 0.085          # wavelength (m)
        self.dx = self.dy = 0.015     # RIS spacing (m)
        self.M = self.N = 16          # RIS size
        self.Pt = 1e-3                # transmit power (W)
        self.Gt_dB = 60               # Tx gain in dB
        self.Gr_dB = 34               # Rx gain in dB
        self.G = 8.0                  # unit cell gain (linear)
        self.A = 1.0                  # reflection magnitude
        self.d1 = 3.5                 # Tx distance behind RIS

        # Rx position (near-field, azimuth 30°)
        azimuth_deg = 30.0
        azimuth_rad = np.radians(azimuth_deg)
        d2 = 1.0
        self.x_rx = d2 * np.cos(azimuth_rad)
        self.y_rx = d2 * np.sin(azimuth_rad)
        self.z_rx = 0.0

def db2lin(db):
    return 10 ** (db / 10)

def distance3D(x1, y1, z1, x2, y2, z2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)

def compute_Prx(binary, p):
    k = 2 * PI / p.lambda_
    z_tx = -p.d1
    total_field = 0 + 0j

    for n in range(p.N):
        for m in range(p.M):
            x_nm = (m - (p.M - 1) / 2.0) * p.dx
            y_nm = (n - (p.N - 1) / 2.0) * p.dy

            rt = distance3D(x_nm, y_nm, 0, 0, 0, z_tx)
            rr = distance3D(x_nm, y_nm, 0, p.x_rx, p.y_rx, p.z_rx)
            total_path = rt + rr

            phase = PI if binary[n][m] == 1 else 0.0
            exp_term = np.exp(-1j * (k * total_path - phase))
            total_field += exp_term / (rt * rr)

    Gt = db2lin(p.Gt_dB)
    Gr = db2lin(p.Gr_dB)
    prefactor = (p.Pt * Gt * Gr * p.G * p.dx * p.dy * p.lambda_**2 * p.A**2) / (64 * PI**3)
    return prefactor * abs(total_field)**2

def generate_pattern_from_int(pattern_int, M, N):
    binary = np.zeros((N, M), dtype=int)
    for m in range(M):
        binary[:, m] = (pattern_int >> m) & 1
    return binary

def evaluate_pattern(i, p):
    binary = generate_pattern_from_int(i, p.M, p.N)
    power = compute_Prx(binary, p)
    return i, power

def format_pattern(pattern, M):
    return [(pattern >> m) & 1 for m in range(M)]

# Main execution
if __name__ == "__main__":
    p = Params()
    best = {"pattern": 0, "power": -np.inf}
    worst = {"pattern": 0, "power": np.inf}

    total_patterns = 2**16
    print("Evaluating 65,536 column patterns...")

    with ThreadPoolExecutor() as executor:
        results = executor.map(lambda i: evaluate_pattern(i, p), range(total_patterns))

        for pattern, power in results:
            if power > best["power"]:
                best = {"pattern": pattern, "power": power}
            if power < worst["power"]:
                worst = {"pattern": pattern, "power": power}

    # Display results
    print("\n=== BEST PATTERN ===")
    print(f"Bitmask: 0x{best['pattern']:04X}")
    print("Columns ON/OFF:", format_pattern(best["pattern"], p.M))
    print(f"Max Power: {best['power']:.3e} W")

    print("\n=== WORST PATTERN ===")
    print(f"Bitmask: 0x{worst['pattern']:04X}")
    print("Columns ON/OFF:", format_pattern(worst["pattern"], p.M))
    print(f"Min Power: {worst['power']:.3e} W")
