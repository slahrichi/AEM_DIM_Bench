"""
This file serves as a prediction interface for the network
"""
# Built in
import os
import sys
sys.path.append('../utils/')
import time
from pathlib import Path
# Torch

# Own
from NA import flag_reader
from NA.class_wrapper import Network
from NA.model_maker import NA
from utils import data_reader
from utils.helper_functions import load_flags
from utils.evaluation_helper import plotMSELossDistrib
import torch
# Libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def predict_from_model(pre_trained_model, Xpred_file, no_plot=True, load_state_dict=None):
    """
    Predicting interface. 1. Retreive the flags 2. get data 3. initialize network 4. eval
    :param model_dir: The folder to retrieve the model
    :param Xpred_file: The Prediction file position
    :param no_plot: If True, do not plot (For multi_eval)
    :param load_state_dict: The new way to load the model for ensemble MM
    :return: None
    """
    # Retrieve the flag object
    print("This is doing the prediction for file", Xpred_file)
    print("Retrieving flag object for parameters")
    if (pre_trained_model.startswith("models")):
        eval_model = pre_trained_model[7:]
        print("after removing prefix models/, now model_dir is:", eval_model)
    
    flags = load_flags(pre_trained_model)                       # Get the pre-trained model
    flags.eval_model = pre_trained_model                    # Reset the eval mode
    flags.test_ratio = 0.2              #useless number  

    # Get the data, this part is useless in prediction but just for simplicity
    #train_loader, test_loader = data_reader.read_data(flags)
    print("Making network now")

    # Make Network
    ntwk = Network(NA, flags, train_loader=None, test_loader=None, inference_mode=True, saved_model=flags.eval_model)
    print("number of trainable parameters is :")
    pytorch_total_params = sum(p.numel() for p in ntwk.model.parameters() if p.requires_grad)
    print(pytorch_total_params)
    # Evaluation process
    print("Start eval now:")
    
    if not no_plot:
        # Plot the MSE distribution
        pred_file, truth_file = ntwk.predict(Xpred_file, no_save=False, load_state_dict=load_state_dict)
        flags.eval_model = pred_file.replace('.','_') # To make the plot name different
        plotMSELossDistrib(pred_file, truth_file, flags)
    else:
        pred_file, truth_file = ntwk.predict(Xpred_file, no_save=True, load_state_dict=load_state_dict)
    
    print("Evaluation finished")

    return pred_file, truth_file, flags

def ensemble_predict(model_list, Xpred_file, model_dir=None, no_plot=True, remove_extra_files=True, state_dict=False, model_dir_root=None):
    """
    This predicts the output from an ensemble of models
    :param model_list: The list of model names to aggregate
    :param Xpred_file: The Xpred_file that you want to predict
    :param model_dir: The directory to plot the plot
    :param no_plot: If True, do not plot (For multi_eval)
    :param remove_extra_files: Remove all the files generated except for the ensemble one
    :param state_dict: New way to load model using state_dict instead of load module
    :return: The prediction Ypred_file
    """
    print("this is doing ensemble prediction for models :", model_list)
    pred_list = []
    # Get the predictions into a list of np array
    for pre_trained_model in model_list:
        if state_dict is False:
            pred_file, truth_file, flags = predict_from_model(pre_trained_model, Xpred_file)
            # This line is to plot all histogram, make sure comment the pred_list.append line below as well for getting all the histograms
            #pred_file, truth_file, flags = predict_from_model(pre_trained_model, Xpred_file, no_plot=False)
        else:
            flags_dir = str((Path(model_dir_root).parent / "model_param").resolve()) \
                        if model_dir_root else str(FLAGS_DIR.resolve())
            pred_file, truth_file, flags = predict_from_model(flags_dir, Xpred_file,
                                                              load_state_dict=pre_trained_model)
        pred_list.append(np.copy(np.expand_dims(pred_file, axis=2)))
    # Take the mean of the predictions
    pred_all = np.concatenate(pred_list, axis=2)
    pred_mean = np.mean(pred_all, axis=2)
    save_name = Xpred_file.replace('Xpred', 'Ypred')
    np.savetxt(save_name, pred_mean)
    
    # If no_plot, then return
    if no_plot:
        return

    # saving the plot down
    flags.eval_model = 'ensemble_plot' + Xpred_file.replace('/', '')
    if model_dir is None:
        return plotMSELossDistrib(save_name, truth_file, flags)
    else:
        return plotMSELossDistrib(save_name, truth_file, flags, save_dir=model_dir)




def predict_all(models_dir="data"):
    """
    This function predict all the files in the models/. directory
    :return: None
    """
    for file in os.listdir(models_dir):
        if 'Xpred' in file and 'meta_material' in file:                     # Only meta material has this need currently
            print("predicting for file", file)
            predict_from_model("models/meta_materialreg0.0005trail_2_complexity_swipe_layer1000_num6", 
            os.path.join(models_dir,file))
    return None


def ensemble_predict_master(model_dir: Path, Xpred_file: Path, no_plot=True, plot_dir: Path=None):
    model_dir = Path(model_dir)
    model_list = []
    print("entering folder to predict:", model_dir)
    
    # patch for new way to load model using state_dict
    state_dict = (model_dir.name == "state_dicts") or ("state_dicts" in str(model_dir))

    # get the list of models to load
    for entry in model_dir.iterdir():
        if entry.name.endswith(".zip") or "skip" in entry.name:
            continue
        if entry.is_dir() or entry.suffix == ".pt":
            model_list.append(str(entry)) 
    if plot_dir is None:
        return ensemble_predict(model_list, str(Xpred_file), str(model_dir), state_dict=state_dict, no_plot=no_plot, model_dir_root=model_dir)
    else:
        return ensemble_predict(model_list, str(Xpred_file), str(plot_dir), state_dict=state_dict, no_plot=no_plot, model_dir_root=model_dir)
        


def predict_ensemble_for_all(model_dir: Path, xpred_dir: Path, no_plot=True):
    model_dir = Path(model_dir)
    xpred_dir = Path(xpred_dir)
    xpred_dir.mkdir(parents=True, exist_ok=True)
    print("Predicting ensemble for all Xpred files in:", xpred_dir)
    for f in xpred_dir.iterdir():
        if not f.is_file():
            continue
        if "Xpred" not in f.name or "Yang" not in f.name:
            continue

        ypred_path = f.with_name(f.name.replace("Xpred", "Ypred"))
        if ypred_path.exists():
            continue

        ensemble_predict_master(model_dir, f, plot_dir=xpred_dir, no_plot=no_plot)

def creat_mm_dataset(
    model_param_dir = None,
    data_in_dir = None,
    state_dict_dir = None,
    num_models = 10):
    """
    Build ADM (Yang_sim) spectra via the neural simulator ensemble and save data_y.csv.
    - model_param_dir: folder that contains flags.obj (e.g., .../Data/Yang_sim/model_param)
    - data_in_dir:     folder that holds data_x.csv (e.g., .../Data/Yang_sim/dataIn)
    - state_dict_dir:  folder with mm0.pt ... mm9.pt (e.g., .../Data/Yang_sim/state_dicts)
    Returns the path to the written data_y.csv
    """
    # ---- Defaults (derived relative to repo root) ----
    repo_root = Path(__file__).resolve().parents[1]  # .../AEM_DIM_Bench
    yang_root = repo_root / "Data" / "Yang_sim"

    model_param_dir = Path(model_param_dir) if model_param_dir else (yang_root / "model_param")
    data_in_dir     = Path(data_in_dir)     if data_in_dir     else (yang_root / "dataIn")
    state_dict_dir  = Path(state_dict_dir)  if state_dict_dir  else (yang_root / "state_dicts")

    geometry_points = data_in_dir / "data_x.csv"
    y_filename      = data_in_dir / "data_y.csv"  # same name pattern as before

    # ---- Load flags & construct simulator network ----
    flags = load_flags(str(model_param_dir))   # load_flags expects a dir containing flags.obj
    flags.eval_model = str(model_param_dir)    # Network() wants a "saved_model" string
    ntwk = Network(NA, flags, train_loader=None, test_loader=None,
                   inference_mode=True, saved_model=flags.eval_model)

    # ---- Run ensemble of state dicts ----
    pred_list = []
    for i in range(num_models):
        sd_path = state_dict_dir / f"mm{i}.pt"
        if not sd_path.exists():
            print(f"[creat_mm_dataset] WARNING: missing {sd_path}, skipping.")
            continue
        print(f"[creat_mm_dataset] predicting with {sd_path.name} ...")
        pred_np, _ = ntwk.predict(
            Xpred_file=str(geometry_points),
            load_state_dict=str(sd_path),
            no_save=True
        )
        pred_list.append(pred_np)

    if not pred_list:
        raise RuntimeError("[creat_mm_dataset] No predictions produced (no state dicts found?).")

    # ---- Average predictions and save ----
    # pred_np has shape (N, 2000). Stack -> (N, 2000, K), mean over K.
    Y_ensemble = np.mean(np.stack(pred_list, axis=2), axis=2)
    np.savetxt(str(y_filename), Y_ensemble)

    print(f"[creat_mm_dataset] wrote {y_filename} with shape {Y_ensemble.shape}")
    return y_filename

if __name__ == '__main__':
    # To create Meta-material dataset, use this line 
    start = time.time()
    #creat_mm_dataset()
    #print('Time is spend on producing MM dataset is {}'.format(time.time()-start))
    
    # multi evaluation 
    method_list = ['Tandem']
    
    #method_list = ['Tandem','MDN','INN','cINN','VAE','GA','NA','NN']
    PROJECT_ROOT = Path(__file__).resolve().parents[1]  # /data/users/sl636/AEM_DIM_Bench
    STATE_DIR = PROJECT_ROOT / "Data" / "Yang_sim" / "state_dicts"  # shared state dicts
    FLAGS_DIR = STATE_DIR.parent / "model_param"   

    for method in method_list:
        print("Predicting ensemble for method:", method)
        MODEL_ROOT = PROJECT_ROOT / method
        DATA_DIR   = MODEL_ROOT / "data"
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        print(f"Using DATA_DIR:   {DATA_DIR}")
        print(f"Using STATE_DIR:  {STATE_DIR}")

        predict_ensemble_for_all(
            STATE_DIR,          # ../Data/Yang_sim/state_dicts
            DATA_DIR,            # same folder as test_Xpred_*.csv
            no_plot=True,
        )
    #predict_ensemble_for_all('../Data/Yang_sim/state_dicts/', '/home/sr365/MM_Bench/GA/temp-dat/GA1_chrome_gen_300/Yang_sim', no_plot=True)  
    
    #predict_from_model("models/Yang_sim_best_model", 'data/test_Xpred_Yang_best_model.csv', no_plot=False, load_state_dict=None)
