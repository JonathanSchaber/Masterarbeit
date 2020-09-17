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
        values =[
                epoch["epoch"],
                epoch["Train Loss"],
                epoch["Dev Loss"],
                epoch["Test Loss"],
                epoch["Train Accur."],
                epoch["Dev Accur."],
                epoch["Test Loss"]
                ]
        print_list.append([str(values[0])] + map(str, map(round, values[1:], [3]*6)))

    print_string = "\n".join(["\t".join(elem) for elem in print_list])
    print(print_string)


def main():
    args = parse_cmd_args()
    read_in_json(args.file)


if __name__ == "__main__":
    main()
    
