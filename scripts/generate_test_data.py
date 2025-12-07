#!/usr/bin/env python3
"""Generate sample test Excel file for sportmodel."""

import sys

try:
    from openpyxl import Workbook
except ImportError:
    print("Installing openpyxl...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl", "-q"])
    from openpyxl import Workbook

from datetime import date, timedelta
import random

def main():
    wb = Workbook()
    ws = wb.active
    ws.title = "Training Data"

    # Header row
    ws.append(["Date", "Weight", "Repetitions", "Movement"])

    # Sample data (intentionally out of chronological order)
    data = [
        # Bodyweight entries
        (date(2024, 1, 5), 82.5, None, "bodyweight"),
        (date(2024, 1, 15), 83.0, None, "bodyweight"),
        (date(2024, 2, 1), 82.0, None, "bodyweight"),
        (date(2024, 3, 10), 81.5, None, "bodyweight"),

        # Squat entries (out of order)
        (date(2024, 2, 15), 140, 5, "squat"),
        (date(2024, 1, 10), 130, 5, "squat"),
        (date(2024, 1, 20), 135, 3, "squat"),
        (date(2024, 3, 1), 150, 1, "squat"),  # Actual 1RM
        (date(2024, 2, 28), 145, 2, "squat"),
        (date(2024, 3, 15), 140, 8, "squat"),

        # Bench entries
        (date(2024, 1, 12), 90, 5, "bench"),
        (date(2024, 2, 10), 95, 3, "bench"),
        (date(2024, 3, 5), 100, 1, "bench"),

        # Deadlift entries
        (date(2024, 1, 15), 160, 5, "deadlift"),
        (date(2024, 2, 20), 170, 3, "deadlift"),
        (date(2024, 3, 10), 180, 1, "deadlift"),

        # Snatch entries
        (date(2024, 1, 8), 70, 1, "snatch"),
        (date(2024, 2, 5), 72, 2, "snatch"),
        (date(2024, 3, 12), 75, 1, "snatch"),

        # Clean & Jerk entries
        (date(2024, 1, 8), 90, 1, "cj"),
        (date(2024, 2, 5), 95, 1, "cj"),
        (date(2024, 3, 12), 100, 1, "cj"),

        # Future entries (pre-entered planned tests)
        (date(2025, 1, 1), 155, 1, "squat"),
        (date(2025, 1, 1), 105, 1, "bench"),

        # Row with missing/invalid data (will be skipped)
        # This is represented as an empty weight cell
    ]

    # Shuffle to make order random
    random.seed(42)  # Reproducible shuffle
    random.shuffle(data)

    for row in data:
        date_val, weight, reps, movement = row
        ws.append([date_val, weight, reps if reps is not None else "", movement])

    # Add a row with missing weight (should be skipped by parser)
    ws.append([date(2024, 2, 1), "", 5, "squat"])

    # Add a row with unknown movement (should be skipped by parser)
    ws.append([date(2024, 2, 1), 100, 5, "unknown_movement"])

    # Save file
    output_path = "test_data.xlsx"
    wb.save(output_path)
    print(f"Created {output_path} with {len(data) + 2} data rows (including 2 invalid rows)")

if __name__ == "__main__":
    main()
