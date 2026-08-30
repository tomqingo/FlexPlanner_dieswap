import pandas as pd
import matplotlib.pyplot as plt
import argparse

def plot_training_curves(csv_file, output_file=None):
    """
    Plot training curves from a CSV file with two columns.
    
    Parameters:
    csv_file (str): Path to the CSV file
    output_file (str): Path to save the plot (optional)
    """
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_file)
        print(f"Successfully loaded CSV file: {csv_file}")
        print(f"Columns found: {list(df.columns)}")
        print(f"Data shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found.")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Check if we have exactly two columns
    if len(df.columns) < 2:
        print("Error: CSV file must have at least two columns.")
        return
    
    # Use the first two columns
    col1 = df.columns[1]
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    
    # Plot both curves
    plt.plot(df[col1], linewidth=4)
    
    # Customize the plot
    plt.xlabel('Epoch', fontsize=20, fontweight='bold')
    plt.ylabel('OV', fontsize=20, fontweight='bold')
    
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')

    # Add some styling
    plt.tight_layout()
    
    # Save or show the plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved as: {output_file}")
    
    plt.show()

if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Plot training curves from CSV file')
    parser.add_argument('--csv_file', help='Path to the CSV file')
    parser.add_argument('--output', '-o', help='Output file path for saving the plot')
    
    args = parser.parse_args()
    
    # Run the plotting function
    plot_training_curves(args.csv_file, args.output)