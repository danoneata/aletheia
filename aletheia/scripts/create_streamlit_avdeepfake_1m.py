import streamlit as st
import os
import json
import pandas as pd
import numpy as np
import random

st.set_page_config(layout="wide")
st.title('AV-Deepfake-1M')

datadir = "/mnt/private-share/speechDatabases/av-deepfake-1m/"


@st.cache_data
def load_data(nrows):
    with open(os.path.join(datadir,"val_metadata.json")) as f:
        a = json.load(f) 
    meta = {}
    for k in range(len(a)):
        meta[a[k]['file']] = a[k]
            
    avail = [k.strip() for k in open(os.path.join(datadir,'available_files.txt')).readlines()]
    data = {}
    for k in avail[:nrows]:
        data[k]  = meta[k]
        
    return data, len(avail)



def filter_data(meta, modification, operation, nsamples):
    data = {}
    c = 0
    keys = list(meta.keys())
    random.shuffle(keys)
    
    for k in keys:
        if meta[k]['modify_type']== modification: 
            f = open(os.path.join(datadir+"/val_metadata", k.replace('mp4', 'json')))
            b = json.load(f)
            if modification =='real':
                data[k] = {}
                data[k]['fs'] = []
                data[k]['info'] = []
                c+=1
            else:
                for m in  b['operations']:
                     if (operation=='any') or (m['operation'] == operation):
                        data[k] = {}
                        data[k]['fs'] = meta[k]['fake_segments']
                        data[k]['info'] = []
                        data[k]['tsc'] = ' '.join([x['word'] for x in b['transcripts']])
                        for n in b['operations']:
                            data[k]['info'].append([n['start'], n['end'], n['old_word'], n['new_word'], n['operation']])
                        
                        c+=1
                        break
                        
        if c == nsamples:           
            break           
    return data
            

#### LOAD DATA
data, navail = load_data(1000)

#### FILTER DATA
with st.sidebar:
    modification = st.selectbox(
        '**:blue[Modification]**',
        ('both_modified', 'audio_modified', 'visual_modified', 'real'))
    operation = st.selectbox(
        '**:orange[Operation]**',
        ('any', 'insert', 'replace', 'delete'))  

    st.divider()
    nsamples = st.slider('Max samples to display', 10, 100)
    ncols = st.slider("Number of columns for display", 3, 8)
    
dd = filter_data(data, modification, operation, nsamples)
st.write("Available files:", navail)


st.markdown(f"**Selected modification: :blue["+modification+"]**")
st.markdown(f"**Selected operation: :orange[{operation}]**")
#st.write(f"Found ",len(dd), " files with the specified operation ",{operation}," from a max of ",nsamples," samples with the specified modification ",{modification})
st.write("Found ", len(dd), " samples")

##### SHOW DATA
#ncols = 5
for line in range(len(dd)//ncols):
    for i, col in enumerate(st.columns(ncols)):
        choice = list(dd.keys())[line*ncols+i] 
        with open(os.path.join(datadir+'/val', choice), 'rb') as v:
            video_bytes = v.read()
        col.video(video_bytes)
        col.markdown("**\[:orange[File:] "+os.path.split(choice)[0] + "**\]")
        col.markdown("**:blue[Fake segments]**")
        for j, times in enumerate(dd[choice]['fs']):
            info = "- "+str(times[0])+"-"+str(times[1])+" | "
            info += ":orange["+dd[choice]['info'][j][-1].upper()+"] | "
            info += ":red["+str(dd[choice]['info'][j][-3]) +"] -> "
            info += ":green["+str(dd[choice]['info'][j][-2]) +"]"
            col.markdown(info)
        col.markdown(dd[choice]['tsc'])
    st.divider()    
    
if len(dd)%ncols:
    for i, col in enumerate(st.columns(ncols)):

        choice = list(dd.keys())[-1-i] 
        with open(os.path.join(datadir+'/val', choice), 'rb') as v:
            video_bytes = v.read()
        col.video(video_bytes)
        col.markdown("**\[:orange[File:] "+os.path.split(choice)[0] + "**\]")
        col.markdown("**:blue[Fake segments]**")
        for j, times in enumerate(dd[choice]['fs']):
            info = "- "+str(times[0])+"-"+str(times[1])+" | "
            info += ":orange["+dd[choice]['info'][j][-1].upper()+"] | "
            info += ":red["+str(dd[choice]['info'][j][-3]) +"] -> "
            info += ":green["+str(dd[choice]['info'][j][-2]) +"]"
            col.markdown(info)
        col.markdown(dd[choice]['tsc'])
        
        if i+1 == len(dd)%ncols:
            break
        