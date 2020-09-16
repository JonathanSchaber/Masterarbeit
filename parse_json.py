import json
import argparse


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
    print_list = []
    print_list.append([
                "epoch",
                "Tr-Loss",
                "De-Loss",
                "Te-Loss",
                "Tr-Acc.",
                "De-Acc.",
                "Te-Acc."
                ])

    with open(file, "r") as f:
        data = json.loads(f.read())
    for epoch in data[2:]:
        print_list.append(map(str, map(round, [epoch[elem] for elem in epoch][:8], [3]*7)))

    print_string = "\n".join(["\t".join(elem) for elem in print_list])
    print(print_string)


def main():
    args = parse_cmd_args()
    read_in_json(args.file)


if __name__ == "__main__":
    main()
    
