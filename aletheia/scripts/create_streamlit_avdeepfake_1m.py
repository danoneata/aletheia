from pathlib import Path
import pdb

import streamlit as st
import os
import json
import pandas as pd
import numpy as np

st.title("AV-Deepfake-1M")
BASE_DIR = Path("/mnt/private-share/speechDatabases/av-deepfake-1m")


@st.cache_data
def load_data(nrows):
    with open(BASE_DIR / "val_metadata.json") as f:
        a = json.load(f)

    meta = {}
    for k in range(len(a)):
        meta[a[k]["file"]] = a[k]

    avail = [k.strip() for k in open(BASE_DIR / "available_files.txt").readlines()]

    data = {}
    for k in avail[:nrows]:
        data[k] = meta[k]

    return data, len(avail)


def filter_data(meta, modification, operation, nsamples):
    data = {}
    c = 0
    for k in meta:
        if meta[k]["modify_type"] == modification:
            c += 1
            f = open(BASE_DIR / "val_metadata" / k.replace("mp4", "json"))
            b = json.load(f)
            if modification == "real":
                data[k] = {}
                data[k]["fs"] = []
                data[k]["info"] = []
            else:
                for m in b["operations"]:
                    if (operation == "any") or (m["operation"] == operation):
                        data[k] = {}
                        data[k]["fs"] = meta[k]["fake_segments"]
                        data[k]["info"] = []
                        for n in b["operations"]:
                            data[k]["info"].append(
                                [
                                    n["start"],
                                    n["end"],
                                    n["old_word"],
                                    n["new_word"],
                                    n["operation"],
                                ]
                            )
                        break
        if c == nsamples:
            break
    return data


#### LOAD DATA
data, navail = load_data(1000)

#### FILTER DATA
with st.sidebar:
    modification = st.selectbox(
        "Modification", ("both_modified", "audio_modified", "visual_modified", "real")
    )
    operation = st.selectbox("Operation", ("any", "insert", "replace", "delete"))
    nsamples = st.slider("Max samples", 10, 100)

dd = filter_data(data, modification, operation, nsamples)
st.write("Available files:", navail)
st.write(
    f"Found ",
    len(dd),
    " files with the specified operation ",
    {operation},
    " from a max of ",
    nsamples,
    " samples with the specified modification ",
    {modification},
)


##### SHOW DATA
for line in range(len(dd) // 2):
    for i, col in enumerate(st.columns(2)):
        choice = list(dd.keys())[line * 2 + i]
        with open(os.path.join("val", choice), "rb") as v:
            video_bytes = v.read()
        col.video(video_bytes)
        col.markdown("**\[:orange[File:] " + os.path.split(choice)[0] + "**\]")
        col.markdown(":blue[Fake segments:] ")
        for j, times in enumerate(dd[choice]["fs"]):
            info = str(times[0]) + "-" + str(times[1]) + " | "
            info += dd[choice]["info"][j][-1].upper().ljust(7) + " | "
            info += ":red[" + str(dd[choice]["info"][j][-3]) + "] -> "
            info += ":green[" + str(dd[choice]["info"][j][-2]) + "]"

            col.markdown(info)

if len(dd) % 2:
    col, col1 = st.columns(2)
    choice = list(dd.keys())[-1]
    with open(os.path.join("val", choice), "rb") as v:
        video_bytes = v.read()
    col.video(video_bytes)
    col.markdown("**\[:orange[File:] " + os.path.split(choice)[0] + "**\]")
    col.markdown(":blue[Fake segments:] ")
    for j, times in enumerate(dd[choice]["fs"]):
        info = str(times[0]) + "-" + str(times[1]) + " | "
        info += dd[choice]["info"][j][-1].upper().ljust(7) + " | "
        info += ":red[" + str(dd[choice]["info"][j][-3]) + "] -> "
        info += ":green[" + str(dd[choice]["info"][j][-2]) + "]"
        col.markdown(info)
