import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Tuple

from . import ROOT_PATH, get_config_dict
from .data_processing import butter_bandpass_filter

config = get_config_dict()


def generate_metadata_df():
    # indexing the csvs in a dataframe
    dataset_path = os.path.join(ROOT_PATH, config["dataset_path"])
    files = []
    for (dirpath, dirnames, filenames) in os.walk(dataset_path):
        aux = [os.path.join(dirpath, filename) for filename in filenames]
        if aux:
            files.extend(aux)

    # removing the absolute path
    files = np.array([os.path.relpath(filename, ROOT_PATH) for filename in files])

    metadata_df = pd.DataFrame({"filepaths": files})

    # differencing between ppg and hr
    metadata_df["type"] = ["ppg" if "ppg" in e else "hr"
                        for e in metadata_df["filepaths"]]

    # joining root_path, just to avoid path problems
    metadata_df["filpaths"] = [os.path.join(ROOT_PATH, s)
                                    for s in metadata_df["filepaths"]]

    # adding patient info
    metadata_df["patient"] = [os.path.basename(os.path.dirname(e))
                            for e in metadata_df["filepaths"].values]

    return metadata_df


def aggregate_patient_data(
    metadata_df: pd.DataFrame,
    downsampling_ratio: int = 2
):
    # train test split by patient
    train_pat, _ = train_test_split(metadata_df["patient"].unique(),
                                    random_state=42)

    # aggregating all data
    dfs = []
    for patient, g_df in metadata_df.groupby("patient"):

        hr_df = pd.read_csv(g_df[g_df["type"] == "hr"]["filepaths"].values[0])
        ppg_df = pd.read_csv(g_df[g_df["type"] == "ppg"]["filepaths"].values[0])

        ppg_df.rename(columns={"R": "R_raw", "G": "G_raw", "B": "B_raw"},
                      inplace=True)

        # normalization
        mean = ppg_df[["R_raw", "G_raw", "B_raw"]].mean()
        std = ppg_df[["R_raw", "G_raw", "B_raw"]].std()
        ppg_df[["R", "G", "B"]] = (ppg_df[["R_raw", "G_raw", "B_raw"]] - mean) / std

        # filtering
        period = ppg_df["Time[ms]"].values[1] - ppg_df["Time[ms]"].values[0]
        fs = 1 / ((period) / 1000)
        ppg_df["R"] = np.array(butter_bandpass_filter(ppg_df["R"].values, 0.5,
                                                      2.5, fs, order=5))
        ppg_df["G"] = np.array(butter_bandpass_filter(ppg_df["G"].values, 0.5,
                                                      2.5, fs, order=5))
        ppg_df["B"] = np.array(butter_bandpass_filter(ppg_df["B"].values, 0.5,
                                                      2.5, fs, order=5))

        # synchronising HR with PPG signals
        # rounding the times is enough to match the HR with the ppg signal
        ppg_df["Time[ms]"] = round(ppg_df["Time[ms]"])
        ret = pd.merge(
            ppg_df,
            hr_df,
            on="Time[ms]",
            how="inner"
        )
        ret["patient"] = patient
        ret["train_test"] = "train" if patient in train_pat else "test"
        
        # downsampling down_sampling_ratio:1
        ret["aux"] = (ret.index // downsampling_ratio)
        ret = ret.groupby(["patient", "train_test", "aux"]).mean().reset_index()
        ret.drop(columns=["aux"], inplace=True)
        dfs.append(ret)

        # Sanity checks
        print(hr_df.shape, ppg_df.shape, ret.shape)

    return pd.concat(dfs)


def process_timeseries_data(
    df: pd.DataFrame,
    input_width: int,
    label_width: int = 1
) -> Tuple[np.array, np.array]:
    # doing it by patient
    x_data = []
    y_data = []
    for pat, pat_df in df.groupby("patient"):

        x_pat = []
        y_pat = []
        df = pat_df[["R", "G", "B", "HR[bpm]"]].copy()

        print("Patient:", pat)
        i_limit = len(df) - label_width - input_width
        for i in tqdm(range(i_limit)):
            end = i + input_width
            x_pat.append(df[i:end][["R", "G", "B"]].values)
            y_pat.append(df.iloc[end + 1][["HR[bpm]"]].values[0])
        x_data.extend(x_pat)
        y_data.extend(y_pat)

    x_data = np.array(x_data)
    y_data = np.array(y_data)
    
    return x_data, y_data
