import argparse
import json


def parse_cmd_args():
    """Parse command line arguments.

    Returns:
        parser: argparse object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "-r", 
            "--result_files", 
            type=str,
            nargs="+",
            help="List of results files for ensemble prediciton"
            )
    return parser.parse_args()


def read_in_files(*files):
    """
    """
    json_files = []
    
    for file in files:
        try:
            with open(file, "r")as f:
                json_file = json.loads(f.read())
            json_files.append(json_file)
        except FileNotFoundError:
            return False


def main():
    args = parse_cmd_args()


if __name__ == "__main__":
    main()
