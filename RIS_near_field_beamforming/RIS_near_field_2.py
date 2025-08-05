import numpy as np
import math
from concurrent.futures import ThreadPoolExecutor

# Constants
PI = np.pi

# Parameters
class Params:
    def __init__(self):
        self.lambda_ = 0.085             # Wavelength (m)
        self.dx = self.dy = 0.026        # Element spacing (m)
        self.M = self.N = 16             # RIS size
        self.Pt = 1e-2                      # Transmit Power (W)
        self.Gt_dB = 20                  # Tx gain in dB
        self.Gr_dB = 20                  # Rx gain in dB
        self.G_dB = 20                   # RIS unit gain in dB
        self.A = 0.8                     # Reflection amplitude
        self.d1 = 3.0                    # Tx to RIS distance (z-axis)

        
        self.azimuth_deg = 50.0
        azimuth_rad = np.radians(self.azimuth_deg)
        d2 = 3.0
        self.x_rx = d2 * np.cos(azimuth_rad)
        self.y_rx = d2 * np.sin(azimuth_rad)
        self.z_rx = 0.0

def db2lin(db):
    return 10 ** (db / 10)

def distance3D(x1, y1, z1, x2, y2, z2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)

# Near-field received power calculation
def compute_Prx(binary, p):
    k = 2 * PI / p.lambda_
    z_tx = p.d1
    total_field = 0 + 0j

    phase_on  = np.pi / 2   # +90°
    phase_off = -np.pi / 2  # -90°

    for m in range(p.M):
        for n in range(p.N):
            x_m = (m - (p.M - 1) / 2.0) * p.dx
            y_n = (n - (p.N - 1) / 2.0) * p.dy

            rt = distance3D(x_m, y_n, 0, 0, 0, z_tx)                         # Tx → element
            rr = distance3D(x_m, y_n, 0, p.x_rx, p.y_rx, p.z_rx)            # element → Rx
            total_path = rt + rr

            # Set 180° phase shift using 1-bit state
            if p.azimuth_deg >0:
                phase_shift = phase_on if binary[n][m] == 1 else phase_off
            else:
                phase_shift = phase_off if binary[n][m] == 1 else phase_on    

            # Compute RIS-controlled phase φ_nm = mod(2π/λ * total_path, 2π)
            #phi_nm = np.mod(k * total_path, 2 * PI)

            # Total field contribution from (m,n)
            exp_term = np.exp(-1j * (k * total_path - phase_shift))
            total_field += exp_term / (rt * rr)

    # Prefactor from RIS near-field path-loss model
    Gt = db2lin(p.Gt_dB)
    Gr = db2lin(p.Gr_dB)
    G  = db2lin(p.G_dB)

    prefactor = (p.Pt * Gt * Gr * G * p.dx * p.dy * p.lambda_**2 * p.A**2) / (64 * PI**3)
    return prefactor * abs(total_field)**2

def generate_pattern_from_int(pattern_int, M, N):
    binary = np.zeros((N, M), dtype=int)
    for m in range(M):
        if (pattern_int >> m) & 1:
            binary[:, m] = 1
    return binary

def convert(binary):
    hex_pattern = "!0X"
    for bit in binary:
        hex_pattern += "FFFF" if bit == 1 else "0000"
    return hex_pattern

def generate_256bit_hex_pattern(binary_matrix):
    pattern_256 = 0
    for row in range(16):
        row_bits = 0
        for bit in range(16):
            row_bits |= (binary_matrix[row][bit] << (15 - bit))
        pattern_256 = (pattern_256 << 16) | row_bits
    return f"0x{pattern_256:064X}"

def evaluate_pattern(i, p):
    binary = generate_pattern_from_int(i, p.M, p.N)
    power = compute_Prx(binary, p)
    return i, power

def format_pattern(pattern, M):
    return [(pattern >> m) & 1 for m in range(M)]

# Main
if __name__ == "__main__":
    p = Params()
    best_patterns = []
    worst_patterns = []

    total_patterns = 2**16
    print("Evaluating 65,536 column patterns...")

    with ThreadPoolExecutor() as executor:
        results = executor.map(lambda i: evaluate_pattern(i, p), range(total_patterns))

        for pattern, power in results:
            if len(best_patterns) < 10 or power > best_patterns[-1]['power']:
                binary = generate_pattern_from_int(pattern, p.M, p.N)
                hex256 = generate_256bit_hex_pattern(binary)
                entry = {
                    "pattern": pattern,
                    "power": power,
                    "binary": binary,
                    "hex256": hex256
                }
                best_patterns.append(entry)
                best_patterns = sorted(best_patterns, key=lambda x: -x["power"])[:10]

            if len(worst_patterns) < 10 or power < worst_patterns[-1]['power']:
                binary = generate_pattern_from_int(pattern, p.M, p.N)
                hex256 = generate_256bit_hex_pattern(binary)
                entry = {
                    "pattern": pattern,
                    "power": power,
                    "binary": binary,
                    "hex256": hex256
                }
                worst_patterns.append(entry)
                worst_patterns = sorted(worst_patterns, key=lambda x: x["power"])[:10]

    # Best 10
    print("\n=== TOP 10 BEST PATTERNS ===")
    for i, item in enumerate(best_patterns):
        print("Columns ON/OFF   :", format_pattern(item["pattern"], p.M))
        print(convert(format_pattern(item["pattern"], p.M)))
        print(f"Power            : {item['power']:.3e} W ({10 * np.log10(item['power']):.2f} dBW)")
