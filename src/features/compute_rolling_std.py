# pylint: disable=invalid-name
"""
# >>> from .compute_rolling_std import compute_rolling_std
# >>> compute_rolling_std("yahoo")
--MSG-- File saved to msft.csv
--MSG-- File saved to googl.csv
--MSG-- File saved to aapl.csv
--MSG-- A
"""

import sys, os
#sys.path.append(os.path.join(os.path.dirname(__file__),'..' ))
from .....constants import get_path

print(get_path(project="yahoo",
               type_of="data"))
import pandas as pd

#rint(ROOT_DIR_PROJECT)
print("hola mundo")

#ROOT_DIR_PROJECT =  "/Users/nataliaacevedo/modelo_tesis_volatilidad/data/yahoo"
#ROLLING_WINDOW = 5

# Compute rolling std for all stocks in processed/ directory
# using 'yield' column.


def compute_rolling_std(root_dir):
    """Adds a column with the daily yields of the adjusted close prices."""

    # Get the list of files in processed/
    processed_files = [f for f in os.listdir(
        os.path.join(root_dir, "data/yahoo/processed/prices/")) if not f.endswith('.DS_Store')]
    
    # Process each file in processed/
    for processed_file in processed_files:
        # Read the file
        df = pd.read_csv(
            os.path.join(root_dir, "data/yahoo/processed/prices", processed_file),
            parse_dates=True,
            index_col=0,
        )

        # Compute the rolling std
        df["rolling_std"] = df["log_yield"].rolling(window=ROLLING_WINDOW).std()

        # Save the file
        df.to_csv(
            os.path.join("data", root_dir, "processed/prices/", processed_file),
            index=True,
        )
        print(f"--MSG-- File saved to {processed_file}")

    # Print message
    print("--MSG-- All files processed.")


if __name__ == "__main__":
    compute_rolling_std(
        root_dir= ROOT_DIR_PROJECT
        )