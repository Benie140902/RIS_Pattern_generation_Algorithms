import numpy as np
import math
from concurrent.futures import ThreadPoolExecutor

# Constants
PI = np.pi

# Parameters
class Params:
    def __init__(self):
        self.lambda_ = 0.085
        self.dx = self.dy = 0.026
        self.M = self.N = 16
        self.Pt = 1e-3
        self.Gt_dB = 20
        self.Gr_dB = 20
        self.G_dB = 15
        self.A = 0.8
        self.d1 = 3.0

        azimuth_deg = -40.0
        azimuth_rad = np.radians(azimuth_deg)
        d2 = 3.0
        self.x_rx = d2 * np.cos(azimuth_rad)
        self.y_rx = d2 * np.sin(azimuth_rad)
        self.z_rx = 0.0

def db2lin(db):
    return 10 ** (db / 10)

def distance3D(x1, y1, z1, x2, y2, z2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)

def compute_Prx(binary, p):
    k = 2 * PI / p.lambda_
    z_tx = p.d1
    total_field = 0 + 0j

    for m in range(p.N):
        for n in range(p.M):
            x_m = (m - (p.M - 1) / 2.0) * p.dx
            y_n = (n - (p.N - 1) / 2.0) * p.dy

            rt = distance3D(x_m, y_n, 0, 0, 0, z_tx)
            rr = distance3D(x_m, y_n, 0, p.x_rx, p.y_rx, p.z_rx)
            total_path = rt + rr

            phase = PI if binary[n][m] == 1 else 0
            if phase !=0:
                azimuth_nm=((k*total_path)%(2*phase))
            else:
                azimuth_nm = 0
            exp_term = np.exp(-1j * (k * total_path - azimuth_nm))
            total_field += exp_term / (rt * rr)

    Gt = db2lin(p.Gt_dB)
    Gr = db2lin(p.Gr_dB)
    G = db2lin(p.G_dB)
    prefactor = (p.Pt * Gt * Gr * G * p.dx * p.dy * p.lambda_**2 * p.A**2) / (64 * PI**3)
    return prefactor * abs(total_field)**2

def generate_pattern_from_int(pattern_int, M, N):
    binary = np.zeros((N, M), dtype=int)
    for m in range(M):
        if (pattern_int >> m) & 1:
            binary[:, m] = 1
    return binary

def convert(binary):
    # patterns = []
    # binary = bin(i)[2:].zfill(16)
    hex_pattern = "!0X"
    for bit in binary:
        hex_pattern += "FFFF" if bit == 1 else "0000"
    # patterns.append(hex_pattern)
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

# Main execution
if __name__ == "__main__":
    p = Params()
    best_patterns = []
    worst_patterns = []

    total_patterns = 2**16
    print("Evaluating 65,536 column patterns...")

    with ThreadPoolExecutor() as executor:
        results = executor.map(lambda i: evaluate_pattern(i, p), range(total_patterns))

        for pattern, power in results:
            # Insert into top 10 best
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

            # Insert into top 10 worst
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

    # Display Best 10
    print("\n=== TOP 10 BEST PATTERNS ===")
    for i, item in enumerate(best_patterns):
        # print(f"\nRank {i+1}")
        # print(f"Bitmask (16-bit): 0x{item['pattern']:04X}")
        print("Columns ON/OFF   :", format_pattern(item["pattern"], p.M))
        print(convert(format_pattern(item["pattern"], p.M)))
        # print("256-bit Hex Pattern:", item["hex256"])
        print(f"Power            : {item['power']:.3e} W ({10 * np.log10(item['power']):.2f} dBW)")

    # Display Worst 10
    # print("\n=== TOP 10 WORST PATTERNS ===")
    # for i, item in enumerate(worst_patterns):
    #     print(f"\nRank {i+1}")
    #     print(f"Bitmask (16-bit): 0x{item['pattern']:04X}")
    #     print("Columns ON/OFF   :", format_pattern(item["pattern"], p.M))
    #     print("256-bit Hex Pattern:", item["hex256"])
    #     print(f"Power            : {item['power']:.3e} W ({10 * np.log10(item['power']):.2f} dBW)")



