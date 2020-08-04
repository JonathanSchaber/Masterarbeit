import json
import argparse

from GLIBERT import load_json


def parse_cmd_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "-f", 
            "--file", 
            type=str, 
            help="Specify file where stats are at",
            )
    return parser.parse_args()


def read_in_json(file):
    """read in file and return json object
    Args:
        param1: str
    Returns:
        json
    """
    with open(file, "r") as f:
        file = f.read()


def main():
    args = parse_cmd_args()
    
