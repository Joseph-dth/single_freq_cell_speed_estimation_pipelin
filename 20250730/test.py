import pandas as pd

# Load the data from the CSV file located in the same directory
file_path = 'ipsc_1M_1_1_speeds_filtered.csv'
df = pd.read_csv(file_path)

# The following is based on the user-provided snippet, with corrections.

# Remove rows where speed_um_per_s is greater than 200
df_filtered = df[df['speed_um_per_s'] <= 200].copy()

# Calculate average and variation of speed_um_per_s from the filtered data
average_speed = df_filtered['speed_um_per_s'].mean()
speed_variation = df_filtered['speed_um_per_s'].var()
speed_std = df_filtered['speed_um_per_s'].std()  # μm/s

df.to_csv(file_path, index=False)

# For direct access to the values, they can also be printed like this:
print("\n--- Raw Calculated Values ---")
print(f"Average Speed: {average_speed}")
print(f"Speed std: {speed_std}")
