import argparse
import json
import subprocess
from pathlib import Path


download_file = "scp jschaber@rattle.ifi.uzh.ch:~/MA/results/{} /home/joni/Documents/Uni/Master/Computerlinguistik/20HS_Masterarbeit/Masterarbeit/results/"

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
        except FileNotFoundError:
            return False
        res_file = file.rstrip("json") + "results.json"
        try:
            with open(res_file, "r") as f:
                results_file = json.loads(f.read())
            json_files.append((stats_file, results_file))
        except FileNotFoundError:
            res_file = Path(res_file).name
            subprocess.Popen(download_file.format(res_file).split(), stdout=subprocess.PIPE)
            try:
                with open(res_file, "r") as f:
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
            best_epoch = epoch["epoch"]

    return best_epoch


def main():
    args = parse_cmd_args()
    files = args.result_files
    print(files)
    jsons = read_in_files(files)
    if not jsons:
        print("There was some error reading in the JSON files. Aborting")
        import ipdb; ipdb.set_trace()

    ensemble_results = []
    for file_pair in jsons:
            stats_file, results_file = file_pair
            best_epoch = get_best_epoch(stats_file)
            dev = results_file[best_epoch][str(best_epoch)]["dev"]    
            test = results_file[best_epoch][str(best_epoch)]["test"]
            ensemble_results.append((dev, test))

    if len(set([len(ensemble_results[i][j]) for i in range(len(files)) for j in range(2)])) > 2:
        print("Files differ, are you sure they come from the same model configuration? Aborting.")
    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    main()
