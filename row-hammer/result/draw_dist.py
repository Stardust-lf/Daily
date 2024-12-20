import matplotlib.pyplot as plt
import numpy as np

def read_latency_file(file_path):
    """
    Read the latency file and return offset and latency as lists.
    """
    offsets = []
    latencies = []

    with open(file_path, 'r') as file:
        for line in file:
            # Extract offset and latency from each line
            parts = line.strip().split(',')
            if len(parts) != 2:
                continue

            # Parse offset and latency
            try:
                offset = int(parts[0].split(':')[1].strip(), 16)  # Hex to int
                latency = float(parts[1].split(':')[1].strip().split()[0])  # Extract latency
                offsets.append(offset)
                latencies.append(latency)
            except ValueError:
                continue

    # Convert to NumPy arrays for easier computation
    offsets = np.array(offsets)
    latencies = np.array(latencies)

    # Remove outliers using IQR method
    Q1 = np.percentile(latencies, 25)  # First quartile
    Q3 = np.percentile(latencies, 75)  # Third quartile
    IQR = Q3 - Q1  # Interquartile range

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter data
    valid_indices = (latencies >= lower_bound) & (latencies <= upper_bound)
    return offsets[valid_indices], latencies[valid_indices]

def plot_latency_with_distribution(offsets, latencies):
    """
    Plot the latency data along with its distribution.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot latency vs offset
    axes[0].plot(offsets, latencies, label="Latency (ns)", linewidth=1.5)
    axes[0].set_xlabel("Offset (Bytes)", fontsize=12)
    axes[0].set_ylabel("Latency (ns)", fontsize=12)
    axes[0].set_title("Memory Latency vs Offset", fontsize=14)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].legend()

    # Plot latency distribution
    axes[1].hist(latencies, bins=50, alpha=0.75, color='blue', edgecolor='black')
    axes[1].set_xlabel("Latency (ns)", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].set_title("Latency Distribution", fontsize=14)
    axes[1].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('latency.png')
    plt.show()

if __name__ == "__main__":
    # File path to the latency results
    file_path = "latency_results.txt"

    # Read and process latency data
    offsets, latencies = read_latency_file(file_path)

    # Plot the data
    plot_latency_with_distribution(offsets,latencies)
