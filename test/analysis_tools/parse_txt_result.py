import csv
import re
import argparse
import pathlib
from argparse import ArgumentParser

def parse_txt_to_csv(txt_file_path, csv_file_path):

    pattern = re.compile(r'(\w+ = [\d.]+)')

    epoch = 0
    data = []
    
    with open(txt_file_path, 'r') as file:
        for line in file:
            if line.startswith('Begin test epoch'):
                epoch = int(line.split()[3])
            else:
                match = pattern.findall(line.strip())
                if match:
                    metrics_data = [epoch] + [m.split(' = ')[1] for m in match]
                    data.append(metrics_data)

    with open(csv_file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        headers = ['Epoch', 'ACCURACY', 'ABS_ERROR1', 'ABS_ERROR2', 'ABS_ERROR3', 'ABS_ERROR4', 'ABS_ERROR5', 'RET_ERROR1', 'RET_ERROR2', 'RET_ERROR3', 'RET_ERROR4', 'RET_ERROR5', 'SAG_ERROR', 'AXI_ERROR', 'COR_ERROR']
        csvwriter.writerow(headers)
        csvwriter.writerows(data)


if __name__ == "__main__":
    
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--src_path",
                        default=pathlib.Path("./data/output/UNETR/test_epochs.log"),
                        type=pathlib.Path)
    parser.add_argument("--dst_path",
                        default=pathlib.Path("./data/output/UNETR/test_epochs.csv"),
                        type=pathlib.Path)
    args = parser.parse_args()

    parse_txt_to_csv(args.src_path, args.dst_path)
