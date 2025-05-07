import pandas as pd

# Load the CSV file
input_file = r'C:\Users\Kompiuteris\Desktop\IM AI WITH THE BRAIDS\4 LABORAS HEHE\AB_NYC_2019.csv'  # Replace with your CSV file path
output_file = r'C:\Users\Kompiuteris\Desktop\IM AI WITH THE BRAIDS\4 LABORAS HEHE\cleaned_dataset.csv'

# Read the dataset
df = pd.read_csv(input_file)

# Remove rows with any missing values
df_cleaned = df.dropna()

# Export the cleaned dataset
df_cleaned.to_csv(output_file, index=False)
print(f"Cleaned dataset saved to {output_file}")
print(f"Original rows: {len(df)}, Cleaned rows: {len(df_cleaned)}")