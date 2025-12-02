import os
import glob

import numpy as np
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from skimage.transform import resize

d_animate = {
    "axolotl": 1,
    "ballBalance": 0,
    "bed": 0,
    "bench": 1,
    "bigAnimalBackground": 1,
    "bikeUnlocking": 1,
    "billboard": 0,
    "bird": 1,
    "birdFalling": 0,
    "blueBoiler": 0,
    "bottleString": 0,
    "candle": 0,
    "carStart": 0,
    "catcafe": 1,
    "chessBoard": 1,
    "chimpanzee": 1,
    "clock": 0,
    "coffeeOnSofa": 1,
    "construction": 1,
    "conversation": 1,
    "crow": 1,
    "crowBall": 1,
    "dino": 0,
    "disinfectant": 0,
    "elevatorEmpty": 0,
    "elevatorWrongSide": 1,
    "espresso": 0,
    "fingerTapping": 1,
    "fish": 1,
    "fly": 1,
    "giraffe": 1,
    "gondolaUp": 0,
    "heron": 1,
    "kettle": 0,
    "lake": 1,
    "laundry": 0,
    "lizard": 1,
    "mail": 1,
    "mokaPot": 0,
    "monitorLizard": 1,
    "monkey2": 1,
    "openDoorInside": 0,
    "openDoorOutside": 1,
    "pedestrian": 1,
    "penDrawing": 0,
    "phone": 0,
    "plank": 1,
    "receipt": 0,
    "reindeers": 1,
    "rippingPaper": 0,
    "robot": 0,
    "robot2": 1,
    "sealion": 0,
    "selfie": 1,
    "shoebill": 1,
    "shoot": 1,
    "skiLift": 0,
    "snip": 0,
    "sparkling2": 0,
    "stapler": 0,
    "statues": 1,
    "teabagOut": 0,
    "throw": 1,
    "toaster": 0,
    "toytrain": 0,
    "toytrainHouses": 0,
    "trafficLight": 0,
    "trafficLight2": 0,
    "trash": 1,
    "trinkvogel2": 0,
    "ventilator": 0,
    "waterHose": 0,
    "waterbottle": 0,
    "watercooler": 0,
    "watering": 1,
    "whiteBoard": 0,
    "work": 1,
    "work2": 1,
    "yoga": 1,
    "youtube": 1,
}


# create PfC rating masks, taken from LPA_experimental_code/evaluate_metrics_psychsci_hpc.py
def create_expert_df(path):
    expert_list = glob.glob(f"{path}*.csv")
    df_pfa = pd.DataFrame()
    for expert_str in expert_list:
        expert_name = expert_str.split("/")[-1][:-12]
        print(expert_name, expert_str)
        df_temp = pd.read_csv(
            expert_str, header=None, names=["scene", "r0", "r1", "r2", "r3"]
        )
        df_temp["expert"] = expert_name
        df_pfa = pd.concat([df_pfa, df_temp], ignore_index=True)
    return df_pfa


def get_potential_for_action(df_pfa, scene):
    pfa_mask = np.zeros((1080, 1920))
    df_temp = df_pfa[df_pfa["scene"] == scene]
    for i in range(len(df_temp)):
        r = [
            int(df_temp.iloc[i][1]),
            int(df_temp.iloc[i][2]),
            int(df_temp.iloc[i][3]),
            int(df_temp.iloc[i][4]),
        ]
        pfa_mask[r[1] : r[1] + r[3], r[0] : r[0] + r[2]] += 1
    pfa_mask /= np.std(pfa_mask)
    pfa_mask -= np.mean(pfa_mask)
    return pfa_mask


def avrg_measure_along_t(
    df,
    measure,
    fov=True,
    seen=False,
    unseen=False,
    animate=False,
    inanimate=False,
    maxtime=5000,
):
    vid_t = np.ones(maxtime) * np.nan
    img_t = np.ones(maxtime) * np.nan
    df = df[df["t"] < maxtime]
    condition = [True for i in range(len(df))]
    if fov:
        condition = condition * (df["em_rv"] == "FOV")
    if seen:
        condition = condition * (df["seen"] == 1)
    if unseen:
        condition = condition * (df["seen"] == 0)
    if animate:
        condition = condition * (df["animate"] == 1)
    if inanimate:
        condition = condition * (df["animate"] == 0)
    vid_res = df[condition * (df.video == 1)].groupby("t")[measure].mean()
    vid_t[: len(vid_res)] = vid_res
    img_res = df[condition * (df.video == 0)].groupby("t")[measure].mean()
    img_t[: len(img_res)] = img_res
    return vid_t, img_t  # , condition


def avrg_measure_along_f(df, measure="pfa", maxtime=150):
    vid_t = np.ones(maxtime) * np.nan
    img_t = np.ones(maxtime) * np.nan
    df = df[df["t"] < maxtime]
    condition = [True for i in range(len(df))]
    vid_res = df[condition * (df.video == 1)].groupby("t")[measure].mean()
    vid_t[: len(vid_res)] = vid_res
    img_res = df[condition * (df.video == 0)].groupby("t")[measure].mean()
    img_t[: len(img_res)] = img_res
    return vid_t, img_t  # , condition


def eval_sim_res_pfa(
    DF_PFA, df_sim_gaze_vid, df_sim_gaze_img, eval_animacy=False, videoset=None
):
    df_sim_res = pd.DataFrame()
    if videoset is None:
        videoset = sorted(df_sim_gaze_vid["scene"].unique())
    for scene in videoset:
        animate_scene = d_animate[scene]
        pfa = get_potential_for_action(DF_PFA, scene)

        df_v = pd.DataFrame()
        df_v["subj_id"] = df_sim_gaze_vid[df_sim_gaze_vid["scene"] == scene]["subject"]
        df_v["t"] = df_sim_gaze_vid[df_sim_gaze_vid["scene"] == scene]["frame"]
        df_v["x"] = df_sim_gaze_vid[df_sim_gaze_vid["scene"] == scene]["x"]
        df_v["y"] = df_sim_gaze_vid[df_sim_gaze_vid["scene"] == scene]["y"]
        df_v["pfa"] = df_v.apply(lambda row: pfa[row["y"], row["x"]], axis=1)
        df_v.insert(0, "scene", scene)
        df_v.insert(1, "video", 1)
        df_v.insert(2, "animate", animate_scene)
        df_sim_res = pd.concat([df_sim_res, df_v], ignore_index=True)

        df_i = pd.DataFrame()
        df_i["subj_id"] = df_sim_gaze_img[df_sim_gaze_img["scene"] == scene]["subject"]
        df_i["t"] = df_sim_gaze_img[df_sim_gaze_img["scene"] == scene]["frame"]
        df_i["x"] = df_sim_gaze_img[df_sim_gaze_img["scene"] == scene]["x"]
        df_i["y"] = df_sim_gaze_img[df_sim_gaze_img["scene"] == scene]["y"]
        df_i["pfa"] = df_i.apply(lambda row: pfa[row["y"], row["x"]], axis=1)
        df_i.insert(0, "scene", scene)
        df_i.insert(1, "video", 0)
        df_i.insert(2, "animate", animate_scene)
        df_sim_res = pd.concat([df_sim_res, df_i], ignore_index=True)

    d_sim = {"pfa": {"vid": {}, "img": {}}}
    # if eval_animacy, add pfa_ani and pfa_ina to d_sim
    if eval_animacy:
        d_sim["pfa_ani"] = {"vid": {}, "img": {}}
        d_sim["pfa_ina"] = {"vid": {}, "img": {}}
    for s_id in sorted(df_sim_res["subj_id"].unique()):
        d_sim["pfa"]["vid"][s_id], d_sim["pfa"]["img"][s_id] = avrg_measure_along_f(
            df_sim_res[df_sim_res["subj_id"] == s_id]
        )
        if eval_animacy:
            d_sim["pfa_ani"]["vid"][s_id], d_sim["pfa_ani"]["img"][s_id] = (
                avrg_measure_along_f(
                    df_sim_res[
                        (df_sim_res["subj_id"] == s_id) & (df_sim_res["animate"] == 1)
                    ]
                )
            )
            d_sim["pfa_ina"]["vid"][s_id], d_sim["pfa_ina"]["img"][s_id] = (
                avrg_measure_along_f(
                    df_sim_res[
                        (df_sim_res["subj_id"] == s_id) & (df_sim_res["animate"] == 0)
                    ]
                )
            )

    for cond in d_sim.keys():
        for mode in ["vid", "img"]:
            d_sim[cond][mode] = pd.DataFrame(d_sim[cond][mode]).T
    return d_sim


def get_saliency_dg(path, scene):
    sal = np.exp(np.load(path + f"dg2e_{scene}_cb.npy"))
    sal = resize(sal, (1080, 1920), order=3)
    sal = (sal - np.mean(sal)) / np.std(sal)
    # sal_ncb = np.exp(np.load(path + f"dg2e_{scene}_nocb.npy"))
    # sal_ncb = resize(sal_ncb, (1080, 1920))
    # sal_ncb = (sal_ncb - np.mean(sal_ncb)) / np.std(sal_ncb)
    return sal  # , sal_ncb


def eval_sim_res_sal(
    df_sim_gaze_vid,
    df_sim_gaze_img,
    eval_animacy=False,
    videoset=None,
    SAL_PATH="/home/nico/project_code/LPA_study/Saliency/",
):
    df_sim_res = pd.DataFrame()
    if videoset is None:
        videoset = sorted(df_sim_gaze_vid["scene"].unique())
    for scene in videoset:
        animate_scene = d_animate[scene]
        sal = get_saliency_dg(SAL_PATH, scene)

        df_v = pd.DataFrame()
        df_v["subj_id"] = df_sim_gaze_vid[df_sim_gaze_vid["scene"] == scene]["subject"]
        df_v["t"] = df_sim_gaze_vid[df_sim_gaze_vid["scene"] == scene]["frame"]
        df_v["x"] = df_sim_gaze_vid[df_sim_gaze_vid["scene"] == scene]["x"]
        df_v["y"] = df_sim_gaze_vid[df_sim_gaze_vid["scene"] == scene]["y"]
        df_v["sal"] = df_v.apply(lambda row: sal[row["y"], row["x"]], axis=1)
        df_v.insert(0, "scene", scene)
        df_v.insert(1, "video", 1)
        df_v.insert(2, "animate", animate_scene)
        df_sim_res = pd.concat([df_sim_res, df_v], ignore_index=True)

        df_i = pd.DataFrame()
        df_i["subj_id"] = df_sim_gaze_img[df_sim_gaze_img["scene"] == scene]["subject"]
        df_i["t"] = df_sim_gaze_img[df_sim_gaze_img["scene"] == scene]["frame"]
        df_i["x"] = df_sim_gaze_img[df_sim_gaze_img["scene"] == scene]["x"]
        df_i["y"] = df_sim_gaze_img[df_sim_gaze_img["scene"] == scene]["y"]
        df_i["sal"] = df_i.apply(lambda row: sal[row["y"], row["x"]], axis=1)
        df_i.insert(0, "scene", scene)
        df_i.insert(1, "video", 0)
        df_i.insert(2, "animate", animate_scene)
        df_sim_res = pd.concat([df_sim_res, df_i], ignore_index=True)

    d_sim = {"sal": {"vid": {}, "img": {}}}
    # if eval_animacy, add pfa_ani and pfa_ina to d_sim
    if eval_animacy:
        d_sim["sal_ani"] = {"vid": {}, "img": {}}
        d_sim["sal_ina"] = {"vid": {}, "img": {}}
    for s_id in sorted(df_sim_res["subj_id"].unique()):
        d_sim["sal"]["vid"][s_id], d_sim["sal"]["img"][s_id] = avrg_measure_along_f(
            df_sim_res[df_sim_res["subj_id"] == s_id], measure="sal"
        )
        if eval_animacy:
            d_sim["sal_ani"]["vid"][s_id], d_sim["sal_ani"]["img"][s_id] = (
                avrg_measure_along_f(
                    df_sim_res[
                        (df_sim_res["subj_id"] == s_id) & (df_sim_res["animate"] == 1)
                    ],
                    measure="sal",
                )
            )
            d_sim["sal_ina"]["vid"][s_id], d_sim["sal_ina"]["img"][s_id] = (
                avrg_measure_along_f(
                    df_sim_res[
                        (df_sim_res["subj_id"] == s_id) & (df_sim_res["animate"] == 0)
                    ],
                    measure="sal",
                )
            )

    for cond in d_sim.keys():
        for mode in ["vid", "img"]:
            d_sim[cond][mode] = pd.DataFrame(d_sim[cond][mode]).T
    return d_sim


def eval_gt_res(EVAL_PATH, measure="pfa", eval_animacy=False, subj_ids=None):
    d_res = {measure: {"vid": {}, "img": {}}}
    if eval_animacy:
        d_res[f"{measure}_ani"] = {"vid": {}, "img": {}}
        d_res[f"{measure}_ina"] = {"vid": {}, "img": {}}
    if subj_ids is None:
        subj_ids = sorted(
            [f[7:9] for f in os.listdir(EVAL_PATH) if "_all_hpc.csv.gz" in f]
        )
    for s_id in sorted(subj_ids):
        # find s_id_file in EVAL_PATH which ends with .csv.gz and includes s_id
        s_id_file = [
            f
            for f in os.listdir(EVAL_PATH)
            if f.endswith(".csv.gz") and f"_{s_id}_" in f
        ]
        assert len(s_id_file) == 1, f"More than one file found for {s_id}"
        df_eval = pd.read_csv(f"{EVAL_PATH}{s_id_file[0]}", compression="gzip")
        print(f"Loaded {s_id_file[0]}")
        d_res[measure]["vid"][s_id], d_res[measure]["img"][s_id] = avrg_measure_along_t(
            df_eval, measure
        )
        if eval_animacy:
            d_res[f"{measure}_ani"]["vid"][s_id], d_res[f"{measure}_ani"]["img"][s_id] = (
                avrg_measure_along_t(df_eval, measure, animate=True)
            )
            d_res[f"{measure}_ina"]["vid"][s_id], d_res[f"{measure}_ina"]["img"][s_id] = (
                avrg_measure_along_t(df_eval, measure, inanimate=True)
            )

    for cond in d_res.keys():
        for mode in ["vid", "img"]:
            d_res[cond][mode] = pd.DataFrame(d_res[cond][mode]).T
    return d_res


def plot_img_vs_vid(
    d_res,
    measure,
    ax,
    tmax=150,
    subj_ids=None,
    img_color="xkcd:blue",
    vid_color="xkcd:red",
    xlabel="Time (ms)",
    ylabel="Potential for change (a.u.)",
    ltitle="",
    limg="",
    lvid=""
):
    if subj_ids is not None:
        for i in range(len(subj_ids)):
            ax.plot(
                d_res[measure]["img"].iloc[i], label="Image", color=img_color, lw=0.1
            )
            ax.plot(
                d_res[measure]["vid"].iloc[i], label="Video", color=vid_color, lw=0.1
            )
    img_mean = d_res[measure]["img"].mean()
    img_sem = d_res[measure]["img"].sem()
    ax.plot(img_mean, color=img_color, lw=1.5, label=limg)
    ax.fill_between(
        img_mean.index,
        img_mean - img_sem,
        img_mean + img_sem,
        alpha=0.5,
        color=img_color,
    )
    vid_mean = d_res[measure]["vid"].mean()
    vid_sem = d_res[measure]["vid"].sem()
    ax.plot(vid_mean, color=vid_color, lw=1.5, label=lvid)
    ax.fill_between(
        vid_mean.index,
        vid_mean - vid_sem,
        vid_mean + vid_sem,
        alpha=0.5,
        color=vid_color,
    )
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_xlim(0, tmax)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    if tmax == 150:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xticks([i * 30 for i in range(int(tmax / 30) + 1)])
        ax.set_xticklabels([str(int(i * 1000)) for i in range(int(tmax / 30) + 1)])
    if ltitle != "":
        legend = ax.legend(title=ltitle, loc="lower right", frameon=False, fontsize=9)
        legend._legend_box.align = "left"

