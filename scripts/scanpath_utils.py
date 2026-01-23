import glob
import os
import pickle
import numpy as np
import pandas as pd
import cv2
from collections import Counter

def angle_limits(angle):
    if -180 < angle <= 180:
        return angle
    elif angle > 180:
        return angle - 360
    else:
        return angle + 360


def calc_px2dva(image_size, display_size=(1080, 1920), display_w_dva=47.7, max_scaling=0.8):
    px2dva_unscaled = display_w_dva / display_size[1]
    if display_size[1] / display_size[0] > image_size[1] / image_size[0]:
        # display is wider than image -> limit by height
        movie_scale_factor = display_size[0] * max_scaling / image_size[0]
    else:
        # display is taller than image -> limit by width
        movie_scale_factor = display_size[1] * max_scaling / image_size[1]
    return px2dva_unscaled * movie_scale_factor


# determine dominant object in a trial
def dominant_obj(entry):
    # clean up object column (sometimes multiple objects are seen, shown as "1, 2")
    entry_copy = entry.copy()  # Avoid SettingWithCopyWarning
    entry_copy["obj_clean"] = (
        entry_copy["obj"]
        .astype(str)
        .str.split(",")
        .apply(lambda xs: [x.strip() for x in xs])
    )

    # explode to count individually
    obj_series = entry_copy.explode("obj_clean")["obj_clean"]
    m = obj_series.mode()
    if m.empty:
        return np.nan
    obj = m.iloc[0]

    # if object is numeric (e.g. "1", "2"), rename using scene
    scene = entry_copy["scene"].iloc[0]

    # try to detect pure numbers (or numeric strings)
    if isinstance(obj, str) and obj.isdigit():  # Only pure digits like "1", "2", "123"
        return f"{scene}_{obj}"
    else:
        return obj 

   
def fov_cat(df):
    seen = set()
    prev_obj = None
    out = []

    for obj in df["obj"]:
        if obj == "B":
            out.append("B")
            prev_obj = obj
            continue

        # first time detection
        if obj not in seen:
            out.append("D")
            seen.add(obj)
            prev_obj = obj
            continue

        # object seen before
        if obj == prev_obj:
            out.append("I")  # continuous inspection
        else:
            out.append("R")  # revisit

        prev_obj = obj

    return out
