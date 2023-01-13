#import plotly.express as px
import streamlit as st
import pandas as pd
from modules.functionforDownloadButtons import download_button
from modules.NN_train import train_NN
from modules.NN_test import test_NN
import plotly.express as px
import matplotlib.pyplot as plt
import streamlit_toggle as tog
import datetime
import asyncio
import time
import numpy as np
import os
import subprocess
from st_aggrid import AgGrid, GridUpdateMode, JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder
import pydoocs

SA1_datastream_toggle = 'XFEL.DAQ/DISTRIBUTOR/DAQ.STREAM.5/EVB.EV.MASK'
SA2_datastream_toggle = 'XFEL.DAQ/DISTRIBUTOR/DAQ.STREAM.6/EVB.EV.MASK'
SA3_datastream_toggle = 'XFEL.DAQ/DISTRIBUTOR/DAQ.STREAM.7/EVB.EV.MASK'
datastream_off_value = 0#2147483603
datastream_on_value = 1
csv_location = 'records/daq_records.csv'

st.set_page_config(page_title="Virtual Diagnostics", layout="wide")

st.title("Virtual Diagnostics")
"""This demo demonstrates virtual diagnostics using DAQ datastreams."""

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

@st.cache
def convert_df(df):
    # Cache the conversion to prevent computation on every rerun
    df.to_csv(csv_location, index=False)

async def watch(records):
    secs = 0
    c = st.info('Starting DAQ')
    while True:
        mm, ss = secs // 60, secs % 60
        c.info(f"DAQ running: {mm:02d}:{ss:02d}")
        r = await asyncio.sleep(1)
        secs = secs + 1

        #if stop_button:
        #    break
        
async def convert(command):
    v = grid_table['selected_rows']
    if v:
        st.write('Converting the following runs')
        st.dataframe(v)

        try:
            with st.spinner('Converting, please wait...'):
                proc1 = subprocess.run(command, shell=True, check=True)
        except FileNotFoundError as exc:
            st.info(f"Process failed because the executable could not be found.\n{exc}")
            return
        except subprocess.CalledProcessError as exc:
            st.info(f"Process failed because did not return a successful return code. " f"Returned {exc.returncode}\n{exc}")
            return
        
        if stop_convert_button:
            proc1.kill()
            return 
        
        if proc1.returncode == 0:
            st.success('Converted successfully!', icon="✅")
            return
    else:
        st.exception('No runs selected. Select run(s) from table.')
    
    
if 'start_daq' not in st.session_state:
    st.session_state['start_daq'] = False
if 'starttime' not in st.session_state:
    st.session_state['starttime'] = None

tab1, tab2, tab3, tab4 = st.tabs(["DAQ", "Gridsearch", "Train", "Test"])

with tab1:
    st.header("DAQ")
    records = pd.read_csv(csv_location) # Read csv with all DAQ records
    ce, c1, ce, c2, ce = st.columns([0.07, 1, 0.07, 2, 0.07])
    with c1:
        sa1_daq_button = tog.st_toggle_switch(label="SA1 Datastream", 
                    key="Key1",  default_value=False, label_after = True, 
                    inactive_color = '#D3D3D3', active_color="#11567f", track_color="#29B5E8")

    
        xmldfile = st.text_input('XML description file path:', '/daq/xfel/admtemp/2022/linac/main/run1982')
        #try:
        #    xmldfile = file_selector(xmlfolder)
        #except:
        #    st.write('Folder not found. Listing files in current folder.')
        #    xmldfile = file_selector()
        st.write('You selected `%s`' % xmldfile)
        apply_filter = st.checkbox('Apply filter by destination')

        st.markdown('Convert RAW files to HDF5:')
        datastream = 'SASE 1'
        if apply_filter:
            if datastream == 'SASE 1':
                bunchfilter = 'SA1'
            if datastream == 'SASE 2':
                bunchfilter = 'SA2'
            if datastream == 'SASE 3':
                bunchfilter = 'SA3'
        else:
            bunchfilter = 'all'
            
        startstring = '2021-11-17T15:02:05'
        stopstring = '2021-11-17T15:02:05'
        #command = "python3 modules/level0.py --start %s --stop %s --xmldfile %s --dest %s" % (startstring, stopstring, xmldfile, bunchfilter)
        command = "python3 modules/hello.py"

        convert_button = st.empty()
        start_convert_button = convert_button.button('Start Conversion')
    
    with c2:
        gd = GridOptionsBuilder.from_dataframe(records)
        gd.configure_pagination(enabled=True)
        gd.configure_default_column(groupable=True, enablePivot=True, enableValue=True, enableRowGroup=True)
        gd.configure_side_bar()
        gd.configure_selection(selection_mode='multiple', use_checkbox=True)
        gridOptions=gd.build()
        tab = st.empty()
        grid_table = AgGrid(records, gridOptions=gridOptions, fit_columns_on_grid_load=True, height=400, width='90%', theme='streamlit',
                            update_mode=GridUpdateMode.GRID_CHANGED, reload_data=False, allow_unsafe_jscode=True, editable=True)

## GRIDSEARCH
with tab2:
    st.header("Gridsearch")
    st.write("Select the parameter space to consider for the search (max 10 values per parameter)")
    with st.form(key="gridsearch_form"):
        numlist = np.arange(1,101)
        epochlist = np.arange(1,10001)
        lrlist = np.linspace(0.0, 1.0, num=20, endpoint=False)

        ce, c1, ce, c2, ce, c4, ce, c5 = st.columns([0.07, 1, 0.07, 1, 0.07, 1, 0.07, 1])
        with c1:
            hidden_nodes = st.multiselect('Hidden nodes:', numlist, max_selections=10)
            epochs = st.multiselect('Epochs:', epochlist, max_selections=10)
        with c4:
            hidden_layers = st.multiselect('Hidden layers:', numlist, max_selections=10)
            learning_rate = st.multiselect('Learning rate:', lrlist, max_selections=10)
            search_button = st.form_submit_button(label="✨ Search")

        if search_button:
            st.info('Running Gridsearch ...')
            try:
                #valid_score = train_NN(option)
                #score = float(valid_score)
                st.success('Placeholder', icon="✅")
            except:
                e = RuntimeError('This is an exception of type RuntimeError')
                st.exception(e)

## TRAINING 
with tab3:
   st.header("Train")
   with st.form(key="train_form"):

    ce, c1, ce, c2, c3 = st.columns([0.07, 1, 0.07, 1, 0.07])
    with c1:
        option = st.selectbox('Select run',('6', '7', '9','10','11','12','13','14','15'))
        train_button = st.form_submit_button(label="✨ Train")

    if train_button:
        st.info('Training NN ...')
        try:
            valid_score = train_NN(option)
            score = float(valid_score)
            st.success('NN trained successfully with validation_score: ' +  str(np.round(score,3)), icon="✅")
        except:
            e = RuntimeError('This is an exception of type RuntimeError')
            st.exception(e)
    
## TESTING 
with tab4:
    st.header("Test")
    n=1
    run_7 = np.arange(1441789915, 1441796858)
    df = pd.read_hdf("data/merged.h5", key='df')
    df = df.filter(items=run_7, axis=0)
    df = df.drop('date', axis=1)
    df = df.iloc[::n, :]
    
    fig = px.scatter(
        df,
        x="com_fel_x",
        y="com_fel_y",
        color="/XFEL.FEL/XGM/XGM.2643.T9/INTENSITY.TD",
        color_continuous_scale="viridis")
 
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

if sa1_daq_button == True:
    starttime = datetime.datetime.now().replace(microsecond=0).isoformat()
    print('Start_time', starttime) 
    st.session_state['starttime'] = starttime
    st.session_state['start_daq'] = True
    asyncio.run(watch(records))

if sa1_daq_button == False and st.session_state.start_daq == True:
    stoptime = datetime.datetime.now().replace(microsecond=0).isoformat() 
    print('Stop_time', stoptime)  
    new_row = {'Datastream': 'SA1', 'Start': st.session_state['starttime'], 'Stop': stoptime}
    record_update = records.append(new_row, ignore_index=True)
    convert_df(record_update)
    st.session_state['start_daq'] = False
    #show_table(record_update)
    #print(records)


if start_convert_button:
    stop_convert_button = convert_button.button('Stop Conversion')
    asyncio.run(convert(command))
    
    #start_convert_button = convert_button.button('Start Conversion')



#d = st.date_input(
#    "Start time",
#    datetime.date(2019, 7, 6))
#st.write('Your birthday is:', d)
#st.markdown("## Download results")

#cs, c1, c2, c3, cLast = st.columns([2, 1.5, 1.5, 1.5, 2])

#with c1:
#    CSVButton2 = download_button(
#        {'test': '1'}, "Data.csv", "📥 Download (.csv)")
#with c2:
#    CSVButton2 = download_button(
#        {'test': '1'}, "Data.txt", "📥 Download (.txt)")
#with c3:
#    CSVButton2 = download_button(
#        {'test': '1'}, "Data.json", "📥 Download (.json)")