import os
import pandas as pd

from constants import DATA_PATH, BOX_PLOT_PATH
from visualization.box import plot_box


def create_title(file, results_string):
    splitname = file.replace(".csv", "").replace(results_string, "").split(".")
    title = splitname[0] + " " + splitname[1]

    if title == "high freq":
        title = "High Frequency Bursts"
    elif title == "noisy bursts":
        title = "Noisy Bursts"
    elif title == "long bursts":
        title = "Long Bursts"
    elif title == "non bursting":
        title = "Non-bursting"
    elif title == "non stationary":
        title = "Non-stationary"
    elif title == "reg bursting":
        title = "Regular Bursts"

    title += ("" if os.path.exists(DATA_PATH + file.replace(results_string, "").replace(".csv", ".burst.beg.csv")) else " (no bursts)")
    return title


def create_boxplots(type="true"):
    METHODS = ['ISIn', 'IRT', 'MI', 'CMA', 'RS', 'PS']

    if type == "true":
        results_string = "results.fractionTP."
        ylabel = 'True Positives'
        savepath = BOX_PLOT_PATH + "fp/"
    elif type == "false":
        results_string = "results.fractionFP."
        ylabel = 'False Positives'
        savepath = BOX_PLOT_PATH + "tp/"

    for file in os.listdir(DATA_PATH):
        if file.startswith(results_string) and file.endswith(".csv"):
            print(file)
            df = pd.read_csv(DATA_PATH + file)

            data = []
            for column in METHODS:
                data.append(df[column].tolist())
                print(df[column].tolist())

            name = create_title(file, results_string)
            plot_box(data, METHODS, [name], title=name, ylabel=ylabel, outliers=True, save=True, savefile=savepath+f"boxplot_{name}.svg")



if __name__ == "__main__":
    # requires data to be saved using 'save_detections.py'
    create_boxplots("true")
    create_boxplots("false")


