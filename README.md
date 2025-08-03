RIS Beamforming Pattern Evaluator

This project evaluates all possible binary ON/OFF column patterns (65,536 total) for a 16×16 Reconfigurable Intelligent Surface (RIS) to determine:

    The best column-wise RIS pattern (maximum received power)

    The worst column-wise RIS pattern (minimum received power)

It simulates beamforming in the near-field region and uses realistic system parameters for 3.5 GHz operation.
Files

    ris_best_worst_column_pattern.py – Main script to run the evaluation

    (Optional) requirements.txt – List of required packages if you want to isolate dependencies

Key Concepts

    Each RIS column is binary: ON = 180° phase shift, OFF = 0°

    For a 16×16 RIS, there are 216=65, ⁣536216=65,536 column combinations

    For each pattern:

        Construct full 16×16 binary matrix

        Compute received power at a fixed target location using path loss and phase addition (from Eq. 10 in RIS literature)

    Use multithreading for fast evaluation using all CPU cores

Technical Details
Assumptions

    Transmitter is placed 3.5 m behind RIS (Tx at z = -3.5)

    Receiver is at 1 m from RIS center in 30° azimuth direction

    Operating frequency: 3.5 GHz (wavelength = 0.085 m)

    RIS cell size: 15 mm × 15 mm

Output and Customization

This project helps identify the best and worst RIS column patterns under a given set of parameters such as:

    RIS dimensions (e.g., 16×16)

    Operating frequency and wavelength (e.g., 3.5 GHz, λ = 0.085 m)

    Tx/Rx position and gains

    Cell spacing

    Rx direction (e.g., 30° azimuth)

Example Output:

=== BEST PATTERN ===
Bitmask: 0xFA5B
Columns ON/OFF: [1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1]
Max Power: 3.421e-05 W (-44.66 dBW)

=== WORST PATTERN ===
Bitmask: 0x012C
Columns ON/OFF: [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
Min Power: 3.02e-09 W (-85.2 dBW)

Customize for Your Use Case

    Note: In this example, we have used specific RIS parameters (e.g., 3.5 GHz, 3.5 m Tx distance, 30° Rx azimuth) to find the best RIS pattern.
    Anyone can manipulate these parameters (RIS size, frequency, Tx/Rx location, antenna gains, etc.) to find the best binary pattern for their own technical setup.
