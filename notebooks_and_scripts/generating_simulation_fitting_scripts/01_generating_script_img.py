import os
if os.getcwd().split("/")[-1] == "examples":
    os.chdir('..')
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

import scandy_pfc.models.ObjectModel as objectmodel_module
import scandy_pfc.models.model as base_model
import scandy_pfc.utils.dataclass as dataclass_module
import scandy_pfc.utils.functions as uf

import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run evolution optimization')
parser.add_argument('--featureset', type=str, choices=["bottom_up", "DG2E_cb", 'None', None], default="bottom_up", help='Feature set to use ("bottom_up", "DG2E_cb")')
parser.add_argument('--centerbias', type=str, choices=["anisotropic_default", 'None', None], default="anisotropic_default", help='Center bias method or None')
parser.add_argument('--imglist', type=str, default="animated", choices=["full", "seed3", "seed10", "animated", "non-animated"], help='Image list choice')
parser.add_argument('--std', type=float, nargs='+', default=[0.5, 1], help='List of std values without comma')
parser.add_argument('--trainlist_size', type=int, default=32, help='Training list size')
parser.add_argument('--frames', type=int, default=150, choices=[15, 30, 90, 150], help='Number of frames (with 30 FPS)')
parser.add_argument('--best_range', type=int, default=5, help='Number of best parameters to simulate')
parser.add_argument('--max_seed', type=int, default=13, help='Maximum seed value')
parser.add_argument('--evolution_name', type=str, default="bigger_trainset_scale4/evol_par_rand_", help='Evolution name prefix')
parser.add_argument('--simulate_name', type=str, default="bigger_trainset_scale4/best_par_rand_", help='Simulation name prefix')
parser.add_argument('--resize', type=bool, default=True , help='Resize images for faster processing.')

args = parser.parse_args()

## Params start
FEATURESET = args.featureset if args.featureset != "None" else None
CENTERBIAS = args.centerbias if args.centerbias != "None" else None
imglist_choice = args.imglist
std_list = args.std
evolution_name = args.evolution_name
simulate_name = args.simulate_name
trainlist_int = args.trainlist_size
frames = args.frames
best_range = args.best_range
max_seed = args.max_seed
resized = args.resize

if FEATURESET == "bottom_up":
    evolution_name += "obj_ll_"
    simulate_name += "obj_ll_"
elif FEATURESET == "DG2E_cb":
    evolution_name += "obj_hl_"
    simulate_name += "obj_hl_"
if CENTERBIAS == "anisotropic_default":
    evolution_name += "cb_"
    simulate_name += "cb_"
if resized == False:
    evolution_name += "fullsize_"
    simulate_name += "fullsize_"
if max_seed is not None:
    evolution_name += f"s{max_seed}_"
    simulate_name += f"s{max_seed}_"
if frames != 150:
    evolution_name += f"{frames/30}s_"
    simulate_name += f"{frames/30}s_"

dataframe_files = {
    "young": 'parkinson_data/df_img_fovs_psycsci.csv',
    "old":  'parkinson_data/df_img_fovs_controls.csv',
    "pd_on": 'parkinson_data/df_img_fovs_parkinson_with.csv',
    "pd_off": 'parkinson_data/df_img_fovs_parkinson_without.csv'
    }

if resized: 
    folder_path = "PictureExample/PictureExample_Resized/"
    pic_path = "PictureExample/PictureExample_Resized/picture"
    px_to_dva_factor = 4
else: 
    folder_path = "PictureExample/PictureExample/"
    pic_path = "PictureExample/PictureExample/picture"
    px_to_dva_factor = 1

Dataset = dataclass_module.Dataset
ObjectModel = objectmodel_module.ObjectModel

files = [os.path.splitext(f)[0] for f in os.listdir(pic_path) 
         if os.path.isfile(os.path.join(pic_path, f)) and f.endswith(".png")]


imglist_dict = {"full": sorted(files), # full imagelist
"seed3": sorted(['shoebill', 'fish', 'pedestrian', 'toaster', 'waterbottle', 'sealion', 'elevatorEmpty', 'watercooler', 'fly', 'chimpanzee', 'trinkvogel2']),
"seed10": ['plank', 'fingerTapping', 'statues', 'work2', 'pedestrian', 'chessBoard', 'youtube', 'axolotl', 'reindeers', 'ballBalance', 'bench', 'birdFalling', 'elevatorEmpty', 'mokaPot', 'toytrainHouses', 'ventilator', 'penDrawing', 'dino', 'trafficLight'],
"animated": ["crowBall", "fingerTapping", "statues", "work2", "pedestrian", "fish", "chessBoard", "youtube", "axolotl", "lizard",
    "coffeeOnSofa", "reindeers", "openDoorOutside", "throw", "shoebill", "chimpanzee", "bird", "mail", "plank", "shoot", "heron",
    "monitorLizard", "bench", "work", "bigAnimalBackground", "conversation", "watering", "robot2", "giraffe", "crow", "bikeUnlocking",
    "monkey2", "fly", "trash", "selfie", "catcafe", "yoga", "elevatorWrongSide", "lake", "construction"],
"non-animated": ["blueBoiler", "watercooler", "birdFalling", "sealion", "sparkling2", "ventilator", "kettle", "waterbottle", "clock", "penDrawing",
           "gondolaUp", "laundry", "receipt", "bottleString", "stapler", "elevatorEmpty", "waterHose", "carStart", "billboard", "teabagOut",
           "snip", "disinfectant", "mokaPot", "trinkvogel2", "ballBalance", "rippingPaper", "phone", "toytrainHouses", "candle", "bed", "robot",
           "espresso", "trafficLight2", "skiLift", "dino", "toytrain", "openDoorInside", "trafficLight", "whiteBoard", "toaster"],
}

imglist = imglist_dict[imglist_choice]

for std_value in std_list:
    for test_group in ["young", "pd_on", "pd_off"]:
        evolution_name_dill = f"{evolution_name}_std{std_value}_img{imglist_choice}_{test_group}.dill"
        evolution_name_hdf = f"{evolution_name}_std{std_value}_img{imglist_choice}_{test_group}.hdf"
        simulate_with_best_params = f"{simulate_name}_std{std_value}_img{imglist_choice}_{test_group}"
        dva = 47.7 if test_group == "young" else 40.45
        random.seed(12345)
        trainlist = sorted(random.sample(imglist, trainlist_int))  
        testlist = sorted([vidname for vidname in imglist if vidname not in trainlist])
        datadict = {
            "PATH": folder_path,  
            'FPS' : 30,
            'PX_TO_DVA' : ((dva * 0.8) / 1920)*px_to_dva_factor,  
            'FRAMES_ALL_VIDS' : frames,
            'VID_EXTENSION' : 'png',
            'gt_foveation_df' : dataframe_files[test_group],
            'dataformat': 'picture', #'video', 'both' #<-- filter for video/picture
            'NAME_COL' : 'scene', #<-- changed from 'video' to 'scene' from ScanDy to ScanDy with Pfc
            'trainset' : trainlist,
            'testset' : testlist,
            'used_videos': imglist, 
            'startpos': 'random'
}
        VidCom = Dataset(datadict)
        print(f'PX_TO_DVA: {VidCom.PX_TO_DVA}, Number of training images: {len(VidCom.trainset)}, Number of testing images: {len(VidCom.testset)}')
        # Ground truth 
        if frames == 150:
            gt_amp_dva = VidCom.gt_foveation_df["sac_amp_dva"].dropna().values
            gt_amp_dva = gt_amp_dva[gt_amp_dva > 0.5] # only saccades larger than 0.5 dva
            gt_dur_ms = VidCom.gt_foveation_df["duration_ms"].dropna().values
        else: 
            MAX_TIME_MS = frames/30 * 1000 
            gt_foveation_filtered = VidCom.gt_foveation_df[
                VidCom.gt_foveation_df["t_end"] <= MAX_TIME_MS  
            ].copy()
            gt_amp_dva = gt_foveation_filtered["sac_amp_dva"].dropna().values
            gt_amp_dva = gt_amp_dva[gt_amp_dva > 0.5] # only saccades larger than 0.5 dva
            gt_dur_ms = gt_foveation_filtered["duration_ms"].dropna().values
            
        # Evolution Algorithm
        def optimize_me(traj):
            ind = evolution.getIndividualFromTraj(traj)

            model = ObjectModel(VidCom)
            if CENTERBIAS is not None:
                model.params["centerbias"] = CENTERBIAS
            if FEATURESET is not None:
                model.params["featuretype"] = FEATURESET
            model.params["feature_std"] = std_value
            #model.params["startpos"] = "human_sample"
            model.params["startpos"] = "random"
            # free model parameters, varied in evolution
            model.params["ddm_thres"] = ind.ddm_thres
            model.params["ddm_sig"] = ind.ddm_sig
            model.params["att_dva"] = ind.att_dva
            model.params["ior_decay"] = ind.ior_decay
            model.params["ior_inobj"] = ind.ior_inobj
            #model.params["feature_fade"] = True
            model.run("train", seeds=[s for s in range(1,max_seed)], overwrite_old=True)

            model.evaluate_all_to_df()  # creates model.result_df
            sim_dur_ms = model.result_df["duration_ms"].dropna().values
            sim_amp_dva = model.result_df["sac_amp_dva"].dropna().values
            sim_amp_dva = sim_amp_dva[sim_amp_dva > 0.5]  # Apply same filter as for GT
             # DEBUGGING - print first evaluation
            if not hasattr(optimize_me, '_debug_printed'):
                print(f"\n=== DEBUGGING FIRST EVALUATION ===")
                print(f"GT amplitudes: n={len(gt_amp_dva)}, mean={gt_amp_dva.mean():.3f}, range=[{gt_amp_dva.min():.3f}, {gt_amp_dva.max():.3f}]")
                print(f"Sim amplitudes: n={len(sim_amp_dva)}, mean={sim_amp_dva.mean():.3f}, range=[{sim_amp_dva.min():.3f}, {sim_amp_dva.max():.3f}]")
                print(f"GT durations: n={len(gt_dur_ms)}, mean={gt_dur_ms.mean():.3f}")
                print(f"Sim durations: n={len(sim_dur_ms)}, mean={sim_dur_ms.mean():.3f}")
                optimize_me._debug_printed = True


            # evaluate fitness
            ks_amp, _ = stats.ks_2samp(gt_amp_dva, sim_amp_dva)
            ks_dur, _ = stats.ks_2samp(gt_dur_ms, sim_dur_ms)
            fitness_tuple = (ks_dur, ks_amp)
            res_dict = model.get_fovcat_ratio()
            
            return fitness_tuple, res_dict

        #Original Parameter Space
        # obj_pars = ParameterSpace(
        #    ["ddm_thres", "ddm_sig", "att_dva", "ior_decay", "ior_inobj"],
        #    [[1.0, 3.0], [0.05, 0.25], [4.0, 6.0], [30, 300], [0.4, 1.0]],
        # )

        # Adjusted Parameter Space (par)
        obj_pars = ParameterSpace(
           ["ddm_thres", "ddm_sig", "att_dva", "ior_decay", "ior_inobj"],
           [ [0.5, 5.0], [0.05, 0.45], [4.0, 10.0],[30, 200], [0.4, 1.0] ])

        evolution = Evolution(
            optimize_me,
            obj_pars,
            weightList=[-1.0, -1.0], 
            filename= evolution_name_hdf,
            POP_INIT_SIZE= 64,#64,#32,
            POP_SIZE= 32,#16,#32, #16
            NGEN= 50,#50,#40,
            ncores= 50#30, 
        )

        evolution.run(verbose = False)
        evolution.saveEvolution(f"sorted_results/{evolution_name_dill}")
      

        # ------- Simulate model with top parameter ------------ 

        model = ObjectModel(VidCom)
        parameters = ["ddm_thres", "ddm_sig", "att_dva", "ior_decay", "ior_inobj"]
        if FEATURESET is not None:
                model.params["featuretype"] = FEATURESET
        model.params["feature_std"] = std_value
        #model.params["feature_fade"] = True
        if CENTERBIAS is not None:
            model.params["centerbias"] = CENTERBIAS
        #model.params["startpos"] = "human_sample"
        model.params["startpos"] = "random"
        df_evol = evolution.dfEvolution(outputs=True).copy()

        for rang in range(best_range):
            for par in parameters:
                        model.params[par] = df_evol.sort_values("score", ascending=False).iloc[rang][
                            par
                        ] 
            filename = f"sorted_results/{simulate_with_best_params}_{rang}.pkl"
            if os.path.isfile(filename) == False:
                model.run('all', seeds=[s for s in range(1,max_seed)], overwrite_old=True)
                with open(filename, 'wb') as file:
                    pickle.dump(model.result_dict, file)
                df_filename = f"sorted_results/{simulate_with_best_params}_{rang}.csv.gz"
                if os.path.isfile(df_filename) == False:
                    model.evaluate_all_to_df()
                    model.result_df.to_csv(
                        df_filename, compression="gzip", index=False
                    )
