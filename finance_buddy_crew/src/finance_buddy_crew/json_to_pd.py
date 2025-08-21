import json

import pandas as pd

# Define the input and output file names
json_file_path = 'GME_prices.json'
csv_file_path = 'GME_prices.csv'

try:
    # Open and load the JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # The actual price data is under the "items" key
    price_items = data['items']

    # Create a Pandas DataFrame from the list of price items
    df = pd.DataFrame(price_items)

    # --- At this point, you have your DataFrame for analysis ---
    print("Successfully created DataFrame. Here are the first 5 rows:")
    print(df.head())

    # Optional: Convert the 'date' column to a proper datetime format
    # This is highly recommended for financial analysis
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date', ascending=True)  # Good practice to sort by date

    print("\nDataFrame Info after converting 'date' column:")
    df.info()

    # Save the DataFrame to a CSV file
    # index=False prevents pandas from writing the DataFrame index as a column
    df.to_csv(csv_file_path, index=False)

    print(f"\nSuccessfully converted data and saved it to {csv_file_path}")

except FileNotFoundError:
    print(f"Error: The file '{json_file_path}' was not found.")
except KeyError:
    print("Error: The key 'items' was not found in the JSON file. The data structure might be different than expected.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
