# change to the root directory of the project
import os
from pathlib import Path

cwd = Path.cwd()
#or set manual: 
#cwd = '.../ScanDy'

for parent in [cwd] + list(cwd.parents):
    if (parent / "data").exists():
        os.chdir(parent)
        break

import importlib
import os
import pickle
import random

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.legend_handler import HandlerTuple

import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns

from neurolib.optimize.evolution import Evolution
from neurolib.utils.parameterSpace import ParameterSpace

import scandy_pfc.models.ObjectModel as     objectmodel_module
import scandy_pfc.models.LocationModel as locationmodel_module
import scandy_pfc.models.model as base_model
import scandy_pfc.utils.dataclass as dataclass_module
import scandy_pfc.utils.functions as uf


dataframe_files = {
    "young": 'df_all_fovs_psycsci.csv',
    "old":  'df_all_fovs_controls.csv.gz',
    "pd_on": 'df_all_fovs_parkinson_with.csv.gz',
    "pd_off": 'df_all_fovs_parkinson_without.csv.gz'

}
folder_path = "PictureExample/PictureExample_Resized/picture"


Dataset = dataclass_module.Dataset
ObjectModel = objectmodel_module.ObjectModel
LocationModel = locationmodel_module.LocationModel

files = [os.path.splitext(f)[0] for f in os.listdir(folder_path) 
         if os.path.isfile(os.path.join(folder_path, f)) and f.endswith(".png")]

# full imagelist
imglist = sorted(files)
#imglist = sorted(['shoebill', 'fish', 'pedestrian', 'toaster', 'waterbottle', 'sealion', 'elevatorEmpty', 'watercooler', 'fly', 'chimpanzee', 'trinkvogel2'])
FEATURESET = "DG2E_cb"#"bottom_up"#"DG2E_cb"#"bottom_up"  #"DG2E_cb"

for test_group in ["young"]:
    evolution_name_dill = f"test_loc/evol_relpx_s10_loc_hl_full_{test_group}.dill"
    evolution_name_hdf = f"test_loc/evol_relpx_s10_loc_hl_full_{test_group}.hdf"
    simulate_with_best_params = f"test_loc/best_relpx_s10_loc_hl_full_{test_group}"


    random.seed(12345)
    trainlist = sorted(random.sample(imglist, 60))
    testlist = sorted([vidname for vidname in imglist if vidname not in trainlist])
    
    datadict = {
        "PATH": "PictureExample/PictureExample_Resized/",  # path to the dataset
        'FPS' : 30,
        'PX_TO_DVA' : ((47.7 * 0.8) / 1920)*4, #For faster processing during testing, resize the videos to smaller dimensions
        'FRAMES_ALL_VIDS' : 150,
        'VID_EXTENSION' : 'png',
        'gt_foveation_df' : dataframe_files[test_group],
        'dataformat': 'picture', #'video', 'both' #<-- filter for video/picture
        'NAME_COL' : 'scene', #<-- changed from 'video' to 'scene' from ScanDy to ScanDy with Pfc
        'trainset' : trainlist,
        'testset' : testlist,
        'used_videos': imglist
    }
    VidCom = Dataset(datadict)

    # Ground truth 
    gt_amp_dva = VidCom.gt_foveation_df["sac_amp_dva"].dropna().values
    gt_amp_dva = gt_amp_dva[gt_amp_dva > 0.5] # only saccades larger than 0.5 dva
    gt_dur_ms = VidCom.gt_foveation_df["duration_ms"].dropna().values


    # ## Evolution Algorithm

    def optimize_me(traj):
        ind = evolution.getIndividualFromTraj(traj)
      
        model = LocationModel(VidCom)
        #model.params["centerbias"] = "anisotropic_default"
        model.params["featuretype"] = FEATURESET
        # free model parameters, varied in evolution
        model.params["ddm_thres"] = ind.ddm_thres
        model.params["ddm_sig"] = ind.ddm_sig
        model.params["att_dva"] = ind.att_dva
        model.params["ior_decay"] = ind.ior_decay
        # IOR parameters depend on the model...
        model.params["ior_dva"] = ind.ior_dva
        
        model.run("train", seeds=[s for s in range(1,10)], overwrite_old=True)

        model.evaluate_all_to_df()  # creates model.result_df
        sim_dur_ms = model.result_df["duration_ms"].dropna().values
        sim_amp_dva = model.result_df["sac_amp_dva"].dropna().values

        # evaluate fitness
        ks_amp, _ = stats.ks_2samp(gt_amp_dva, sim_amp_dva)
        ks_dur, _ = stats.ks_2samp(gt_dur_ms, sim_dur_ms)
        fitness_tuple = (ks_dur, ks_amp)

        # we can store more information in the HDF file by returning a dictionary
        res_dict = model.get_fovcat_ratio()
        
        return fitness_tuple, res_dict

    obj_pars = ParameterSpace(
        ["ddm_thres", "ddm_sig", "att_dva", "ior_decay", "ior_dva"],
       #[[1.0, 3.0], [0.05, 0.25], [4.0, 6.0], [30, 300], [0.4, 1.0]],
       [[0.2, 2], [0.01, 0.1], [3, 15], [30, 300], [0.5, 1.0]],
    )
    # Nico Location: 'ddm_thres': [0.2, 2], 'ddm_sig': [0.01, 0.1], 'att_dva': [3, 15], 'ior_decay': [30, 300], 'ior_dva': [0.5, 10]}
    # Nico Object:  'ddm_thres': [1.0, 3.0], 'ddm_sig': [0.05, 0.25], 'att_dva': [5, 20], 'ior_decay': [30, 300], 'ior_inobj': [0.4, 1.0]}


    
    evolution = Evolution(
        optimize_me,
        obj_pars,
        weightList=[-1.0, -1.0], #weights foveation duration and saccade amplitude equally
        filename= evolution_name_hdf,
        POP_INIT_SIZE= 32, #64, #6 better: 64 candidate solutions created in the first generation
        POP_SIZE= 16, #32, #6 better: 32 individuals are kept or generated after selection and mutation in each generation
        NGEN=30, #50 #5 better: 50 for one video 30 seem enough
    )


    
    # # verbose means it creates multiple plots for each generation
    evolution.run(verbose = False)

    evolution.saveEvolution(f"results/{evolution_name_dill}")


    # -------------------


    evol = Evolution(lambda x: x, ParameterSpace(['mock'], [[0, 1]]))
    evolution = evol.loadEvolution(f"results/{evolution_name_dill}")


    # ## Simulate model with top parameter

    #model = ObjectModel(VidCom)
    model = LocationModel(VidCom)
    parameters = ["ddm_thres", "ddm_sig", "att_dva", "ior_decay", "ior_dva"]
    model.params["featuretype"] = FEATURESET
    #model.params["centerbias"] = "anisotropic_default"

    df_evol = evolution.dfEvolution(outputs=True).copy()

    for i in range(2):
        for par in parameters:
                    model.params[par] = df_evol.sort_values("score", ascending=False).iloc[i][
                        par
                    ] 
        filename = f"results/{simulate_with_best_params}_{i}.pkl"
        if os.path.isfile(filename) == False:
            print(f"Running model for top{i} parameter set...")
            model.run('all', seeds=[s for s in range(1,10)], overwrite_old=True) #range(1,13)
            with open(filename, 'wb') as file:
                pickle.dump(model.result_dict, file)
            df_filename = f"results/{simulate_with_best_params}_{i}.csv.gz"
            if os.path.isfile(df_filename) == False:
                model.evaluate_all_to_df()
                model.result_df.to_csv(
                    df_filename, compression="gzip", index=False
                )

