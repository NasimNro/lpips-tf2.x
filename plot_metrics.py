import json
import matplotlib.pyplot as plt
import argparse
import numpy as np

def plot_metrics(input_file: str, output_file: str) -> None:
    try:
        # Read the JSON file
        with open(input_file, 'r') as f:
            data = json.load(f)

        # Create figure with four subplots
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 16))
        fig.suptitle('Metrics vs Distance from Middle Slice')

        # Plot LPIPS
        ax1.plot(data['slice_indices'], data['lpips'], 'b-')
        ax1.set_ylabel('LPIPS')
        ax1.set_xlabel('Distance from Middle Slice')
        ax1.grid(True)

        # Plot PSNR
        ax2.plot(data['slice_indices'], data['psnr'], 'g-')
        ax2.set_ylabel('PSNR')
        ax2.set_xlabel('Distance from Middle Slice')
        ax2.grid(True)

        # Plot SSIM
        ax3.plot(data['slice_indices'], data['ssim'], 'r-')
        ax3.set_ylabel('SSIM')
        ax3.set_xlabel('Distance from Middle Slice')
        ax3.grid(True)

        # Normalize and plot all metrics together
        lpips_norm = (data['lpips'] - np.min(data['lpips'])) / (np.max(data['lpips']) - np.min(data['lpips']))
        psnr_norm = (data['psnr'] - np.min(data['psnr'])) / (np.max(data['psnr']) - np.min(data['psnr']))
        ssim_norm = (data['ssim'] - np.min(data['ssim'])) / (np.max(data['ssim']) - np.min(data['ssim']))
        
        # Flip LPIPS (1 - lpips_norm) since lower LPIPS means better quality
        ax4.plot(data['slice_indices'], 1 - lpips_norm, 'b-', label='LPIPS')
        ax4.plot(data['slice_indices'], psnr_norm, 'g-', label='PSNR')
        ax4.plot(data['slice_indices'], ssim_norm, 'r-', label='SSIM')
        ax4.set_ylabel('Normalized Metrics')
        ax4.set_xlabel('Distance from Middle Slice')
        ax4.grid(True)
        ax4.legend()

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")

    except Exception as e:
        print(f"Error during plotting: {str(e)}")

def main(input_file: str = "metrics_results.json", output_file: str = "metrics_plot.png") -> None:
    """
    Main function that can be called directly or through command line.
    Args:
        input_file: Path to the input JSON file with metrics
        output_file: Path where to save the plot
    """
    plot_metrics(input_file, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot metrics from JSON file')
    parser.add_argument('input', help='Input JSON file with metrics', nargs='?', default="metrics_results.json")
    parser.add_argument('output', help='Output plot file path (e.g., plot.png)', nargs='?', default="metrics_plot.png")
    
    args = parser.parse_args()
    main(args.input, args.output) 