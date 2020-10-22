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


def read_in_files(files):
    """
    """
    json_files = []
    
    for file in files:
        try:
            with open(file, "r")as f:
                stats_file = json.loads(f.read())
            with open(file.rstrip("json") + "results.json", "r") as f:
                results_file = json.loads(f.read())
            json_files.append((stats_file, results_file))
        except FileNotFoundError:
            return False

    return json_files


def get_best_epoch(stats_file):
    """
    """
    best_epoch = 1
    best_dev = 0.0

    for epoch in stats_file[2:]:
        cur_dev = epoch["Dev Accur."]
        if cur_dev > best_dev:
            best_dev = cur_dev
            best_eopch = epoch["epoch"]

    return best_epoch


def main():
    args = parse_cmd_args()
    files = args.result_files
    print(files)
    jsons = read_in_files(files)
    if not jsons:
        print("There was some error reading in the JSON files. Aborting")
        return
    import ipdb; ipdb.set_trace()
    


if __name__ == "__main__":
    main()
