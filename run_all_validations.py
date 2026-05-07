"""
Run building height validation sequentially for all cities defined in city_configs.py.

Calls validate_city.py as a subprocess for each city, so outputs and logs
are isolated per city run.

Usage:
    python run_all_validations.py
"""

import subprocess
import sys
from datetime import datetime
from city_configs import CITIES


def run_all_cities():
    print("=" * 60)
    print("BUILDING HEIGHT VALIDATION — ALL CITIES")
    print("=" * 60)
    print(f"Cities : {', '.join(c['name'] for c in CITIES.values())}")
    print(f"Start  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    results = {}

    for city_key, city_conf in CITIES.items():
        city_name = city_conf['name']
        print(f"\n{'='*60}")
        print(f"PROCESSING: {city_name.upper()}")
        print(f"{'='*60}\n")

        try:
            result = subprocess.run(
                [sys.executable, 'validate_city.py', '--city', city_key],
                capture_output=False,
                text=True,
            )
            if result.returncode == 0:
                results[city_name] = "Success"
            else:
                results[city_name] = f"Failed (code {result.returncode})"

        except Exception as e:
            results[city_name] = f"Error: {e}"

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"End : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    for city, status in results.items():
        print(f"  {city:<20} {status}")

    if all("Success" in s for s in results.values()):
        print("\nAll validations completed successfully.")
    else:
        print("\nOne or more validations failed. Check output above.")


if __name__ == "__main__":
    run_all_cities()
