import argparse
import json
import subprocess

from pathlib import Path
from typing import Dict, List, Tuple


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


#def read_in_files(files: List[str]) -> List[Tuple[List[Dict[str, float]]], Tuple[List[Dict[str, float]]]]:
def read_in_files(files):
    """Reads in the files, return json-loaded objects
    Args:
        files: list of n stats-files
    Returns:
        json_files: json-loaded stats and corresponding results files
    
    """
    json_files = []
    
    for file in files:
        try:
            with open(file, "r")as f:
                stats_file = json.loads(f.read())
        except FileNotFoundError:
            print("Stats file not found. Aborting.")
            return False
        res_file = file.rstrip("json") + "results.json"
        try:
            with open(res_file, "r") as f:
                results_file = json.loads(f.read())
            json_files.append((stats_file, results_file))
        except FileNotFoundError:
            print("Results file locally not found. Trying to fetch...")
            fetch_file = Path(res_file).name
            result = subprocess.run(download_file.format(fetch_file).split(),
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE
                                )
            if result.stderr:
                print(result.stderr)
                return False
            with open(res_file, "r") as f:
                results_file = json.loads(f.read())
            json_files.append((stats_file, results_file))

    return json_files


def get_best_epoch(stats_file):
    """Find the best epoch according to Dev Accuracy
    Args:
        stats_file: json-file of all epoch results
    Returns:
        best_epch: number of the best epoch (starting from 1)
        best_dev: corresponding Dev Accuracy
        best_test: corresponding Test Accuracy
    """
    best_epoch = 1
    best_dev = 0.0
    best_test = 0.0

    for epoch in stats_file[2:]:
        cur_dev = epoch["Dev Accur."]
        if cur_dev > best_dev:
            best_dev = cur_dev
            best_test = epoch["Test Accur."]
            best_epoch = epoch["epoch"]

    return best_epoch, best_dev, best_test


def build_dicts(results: Tuple[List[List[str]]], target_dict: Dict[str, List[str]], target_gold_dict: Dict[str, str]) -> None:
    """build dictionaries; one with all preds, one gold
    Args:
        results: dictionary of pred, golds of best epoch
        target_dict: dictionary for collecting all predictions
        target_gold_dict: dictionary for gold labels
    """
    for element in results:
        if element[0] in target_dict:
            target_dict[element[0]].append(element[1])
            if not target_gold_dict[element[0]] == element[2]:
                import ipdb; ipdb.set_trace()
        else:
            target_dict[element[0]] = [element[1]]
            target_gold_dict[element[0]] = element[2]

    return target_dict, target_gold_dict

    
def compute_acc(preds: Dict[str, str], gold: Dict[str, str]) -> float:
    """Compute the accuracy of the predictions

    Args:
        preds: predicitons dicitonary
        gold: gold labels
    Return:
        accuracy
    """
    true, false = 0, 0

    for key, value in preds.items():
        if value == gold[key]:
            true += 1
        else:
            false += 1

    return true / (true + false)
    
 
def main():
    args = parse_cmd_args()
    files = args.result_files
    #print(files)
    jsons = read_in_files(files)
    if not jsons:
        print("There was some error reading in the JSON files. Aborting")
        import ipdb; ipdb.set_trace()

    ensemble_results = []
    dev_dict, dev_gold_dict, test_dict, test_gold_dict = {}, {}, {}, {}
    
    for i, file_pair in enumerate(jsons):
            stats_file, results_file = file_pair
            best_epoch, best_dev, best_test = get_best_epoch(stats_file)
            print("")
            print("Info for file {}:".format(Path(files[i]).name))
            print("Best epoch: {}".format(best_epoch))
            print("Best dev accuracy: {0:.2f}".format(best_dev))
            print("Best test accuracy: {0:.2f}".format(best_test))
            print("")
            dev = results_file[best_epoch][str(best_epoch)]["dev"]    
            test = results_file[best_epoch][str(best_epoch)]["test"]
            ensemble_results.append((dev, test))

    if len(set([len(ensemble_results[i][j]) for i in range(len(files)) for j in range(2)])) > 2:
        print("Files differ, are you sure they come from the same model configuration? Aborting.")
        return
        
    for results in ensemble_results:
        dev_res, test_res = results

        build_dicts(dev_res, dev_dict, dev_gold_dict)
        build_dicts(test_res, test_dict, test_gold_dict)
        
    dev_ensemble = {key: max(value, key=value.count) for key, value in dev_dict.items()}
    test_ensemble = {key: max(value, key=value.count) for key, value in test_dict.items()}

    dev_accur = compute_acc(dev_ensemble, dev_gold_dict)
    test_accur = compute_acc(test_ensemble, test_gold_dict)

    print("")
    print("==== ENSEMBLE PERFORMANCE ====")
    print("")
    print("dev accur: {0:.4f}".format(dev_accur))
    print("test accur: {0:.4f}".format(test_accur))

     
if __name__ == "__main__":
    main()
    
