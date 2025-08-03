import numpy as np
import math
import cmath
from dataclasses import dataclass
from typing import List

PI = math.pi

@dataclass
class Params:
    lambda_: float
    dx: float
    dy: float
    M: int
    N: int
    Pt: float
    Gt_dB: float
    Gr_dB: float
    G_dB: float
    A: float
    d_t: float
    x_rx: float
    y_rx: float
    z_rx: float
    theta_t_deg: float
    phi_t_deg: float

def db2lin(db):
    return 10 ** (db / 10.0)

def deg2rad(deg):
    return deg * PI / 180.0

def mod2pi(x):
    return (x % (2 * PI) + 2 * PI) % (2 * PI)

def compute_farfield_power(binary: List[List[int]], p: Params):
    sum_val = 0 + 0j
    k = 2 * PI / p.lambda_

    kt = [0, 0, 1]
    kr = [p.x_rx, p.y_rx, p.z_rx]
    d_r = math.sqrt(p.x_rx**2 + p.y_rx**2 + p.z_rx**2)
    kr = [kr_i / d_r for kr_i in kr]

    for n in range(p.N):
        for m in range(p.M):
            x = (m + 0.5) * p.dx
            y = (n + 0.5) * p.dy
            z = 0

            dot = k * ((kt[0] + kr[0]) * x + (kt[1] + kr[1]) * y + (kt[2] + kr[2]) * z)
            theta_d = math.asin(math.sqrt(p.x_rx**2 + p.y_rx**2) / d_r)
            phi_d = math.atan2(p.y_rx, p.x_rx)
            theta_t = deg2rad(p.theta_t_deg)
            phi_t = deg2rad(p.phi_t_deg)
            delta1 = -math.sin(theta_t) * math.cos(phi_t) - math.sin(theta_d) * math.cos(phi_d)
            delta2 = -math.sin(theta_t) * math.sin(phi_t) - math.sin(theta_d) * math.sin(phi_d)
            phase = (2 * PI / p.lambda_) * (delta1 * x + delta2 * y)
            binary_phase = PI if binary[n][m] == 1 else 0.0
            sum_val += cmath.exp(1j * (dot - binary_phase))

    Gt = db2lin(p.Gt_dB)
    Gr = db2lin(p.Gr_dB)
    G = db2lin(p.G_dB)
    prefactor = (p.Pt * Gt * Gr * G * p.dx * p.dy * p.lambda_**2 * p.A**2) / \
                (64.0 * PI**3 * p.d_t**2 * d_r**2)
    return prefactor * abs(sum_val)**2

def main():
    p = Params(
        lambda_=0.085,
        dx=0.015,
        dy=0.015,
        M=16,
        N=16,
        Pt=1e-3,
        Gt_dB=60,
        Gr_dB=44,
        G_dB=0,
        A=1.0,
        d_t=11.5,
        x_rx=0, y_rx=0, z_rx=0,
        theta_t_deg=0,
        phi_t_deg=0
    )

    d2 = 11.0
    azimuth_deg = 50.0
    azimuth_rad = deg2rad(azimuth_deg)
    p.x_rx = d2 * math.cos(azimuth_rad)
    p.y_rx = d2 * math.sin(azimuth_rad)
    p.z_rx = 0.0

    max_power = -1.0
    min_power = 1e9
    best_pattern = 0
    worst_pattern = 0

    for i in range(65536):
        binary = [[0 for _ in range(p.M)] for _ in range(p.N)]
        for m in range(p.M):
            on = (i >> m) & 1
            for n in range(p.N):
                binary[n][m] = on

        Pr = compute_farfield_power(binary, p)
        if Pr > max_power:
            max_power = Pr
            best_pattern = i
        if Pr < min_power:
            min_power = Pr
            worst_pattern = i

    print("\n=== BEST FAR-FIELD PATTERN ===")
    print(f"Bitmask: 0x{best_pattern:04X}\nColumns: ", end="")
    print(" ".join(str((best_pattern >> m) & 1) for m in range(p.M)))
    print(f"Max Power: {max_power:.4e} W ({10 * math.log10(max_power):.2f} dBW)")

    print("\n=== WORST FAR-FIELD PATTERN ===")
    print(f"Bitmask: 0x{worst_pattern:04X}\nColumns: ", end="")
    print(" ".join(str((worst_pattern >> m) & 1) for m in range(p.M)))
    print(f"Min Power: {min_power:.4e} W ({10 * math.log10(min_power):.2f} dBW)")

if __name__ == "__main__":
    main()
