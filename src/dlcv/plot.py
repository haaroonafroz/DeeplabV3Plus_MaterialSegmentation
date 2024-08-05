import argparse
import os
import csv
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dlcv.utils import plot_multiple_losses_and_accuracies

def read_csv_data(filepath):
    """
    Reads CSV data from a file and extracts epoch-wise train losses, test losses, and test accuracies.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        tuple: Returns train_losses, test_losses, test_accuracies lists.
    """
    train_losses, test_losses, test_ious_object, test_ious_material = [], [], [], []
    with open(filepath, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            train_losses.append(float(row['Train Loss']))
            test_losses.append(float(row['Test Loss']))
            test_ious_object.append(float(row['Test IoU: Object']))
            test_ious_material.append(float(row['Test IoU: Material']))
    return train_losses, test_losses, test_ious_object, test_ious_material


def main(args):
    """
    Main function to handle the plotting of training and testing results from multiple CSV files.
    It filters out files based on exclusion patterns and aggregates data for plotting.

    Args:
        args (Namespace): Parsed command line arguments with 'folder' and 'exclude' options.
    """
    model_data_list = []
    exclusion_patterns = args.exclude.split(',') if args.exclude else []

    for filename in os.listdir(args.folder):
        if filename.endswith(".csv") and not any(excl in filename for excl in exclusion_patterns):
            full_path = os.path.join(args.folder, filename)
            train_losses, test_losses, test_ious_object, test_ious_material = read_csv_data(full_path)
            model_data = {
                'name': filename.replace('.csv', ''),
                'train_losses': train_losses,
                'test_losses': test_losses,
                'test_ious_object': test_ious_object,
                'test_ious_material': test_ious_material
            }
            model_data_list.append(model_data)

    plot_multiple_losses_and_accuracies(model_data_list) # ToDo <- add this function in utils.py


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot training and testing results from CSV files.')
    parser.add_argument('--folder', type=str, default="./results", help='Folder containing the CSV files.')
    parser.add_argument('--exclude', type=str,
                        help='Comma-separated string patterns to exclude files from being plotted.')
    args = parser.parse_args()
    main(args)