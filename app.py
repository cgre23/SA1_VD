#import plotly.express as px
import streamlit as st
import pandas as pd
from modules.functionforDownloadButtons import download_button
from modules.NN_train import train_NN
from modules.NN_test import test_NN
import plotly.express as px
import matplotlib.pyplot as plt
import datetime
import numpy as np
#import pydoocs

st.set_page_config(
    page_title="SASE1 Virtual Diagnostics", layout="wide"
)

st.title("SASE1 Virtual Diagnostics")
"""This demo demonstrates SASE1 virtual diagnostics using DAQ datastreams."""


#st.sidebar.title("Settings")


    # Can be used wherever a "file-like" object is accepted:
    #st.write(dataframe)
# If the user doesn't want to select which features to control, these will be used.
#fac = "XFEL*"
#facility = pydoocs.names("XFEL*")
#fac = st.sidebar.selectbox('Facility filter', facility)
#device = pydoocs.names(str(fac)+'/*')
#dev = st.sidebar.selectbox('Device filter', device)
#location = pydoocs.names(str(fac)+'/'+ str(dev)+'/*')
#loc = st.sidebar.selectbox('Location filter', location)

#st.write('You selected: ', str(fac)+'/'+ str(dev)+'/'+str(loc))
#if st.sidebar.checkbox("Show advanced options"):
    # Let the user pick which features to control with sliders.
#    speed = st.sidebar.number_input('Set animation speed', min_value=1, max_value=500, value=50, step=5, format=None, key='speed', help='None')
#st.sidebar.write("""The animation speed is fixed at 10 ms per frame.""")
    #control_features = st.sidebar.multiselect(
    #    "Exclude which cells?",
    #    ['Cell 1', 'Cell 2', 'Cell 3', 'Cell 4', 'Cell 5', 'Cell 6', 'Cell 7', 'Cell 8', 'Cell 9', 'Cell 10'],
   #     ['Cell 1'], help='This is still work in progress.'
    #)
#else:
    # Don't let the user pick feature values to control.
    #control_features = default_control_features
#    speed = 100
#st.sidebar.slider('Test', 0, 100, 50, 5)
#st.sidebar.title("Note")
#st.sidebar.write(
#    """The app is still in development.
#     """
#)
#st.sidebar.caption("Developed by: Christian Grech (DESY, MXL)")
#st.sidebar.caption(f"Streamlit version `{st.__version__}`")

tab1, tab2, tab3, tab4 = st.tabs(["DAQ", "Gridsearch", "Train", "Test"])

with tab1:
   st.header("DAQ")
   #st.image("https://static.streamlit.io/examples/cat.jpg", width=200)
   with st.form(key="DAQ_form"):

    ce, c1, ce, c2, ce, c4, ce, c5 = st.columns([0.07, 1, 0.07, 1, 0.07, 1, 0.07, 1])
    with c1:
        d1 = st.date_input("Start date", datetime.datetime.now())
        d2 = st.date_input("Stop date", datetime.datetime.now())

    with c2:
        start = "00:00"
        end = "23:59"
        times = []
        start = now = datetime.datetime.strptime(start, "%H:%M")
        end = datetime.datetime.strptime(end, "%H:%M")
        while now != end:
            times.append(str(now.strftime("%H:%M")))
            now += datetime.timedelta(minutes=1)
        times.append(end.strftime("%H:%M"))
        t1 = st.selectbox('Start time:',times)

        t2 = st.selectbox('Stop time:',times)
        
        #t1 = st.time_input("Start time", datetime.datetime.now())
        #t2 = st.time_input("Stop time", datetime.datetime.now())
        #words = st.text_area("Enter search words:", height=10)
        #words = st.multiselect('Choose a Keyword Tag:', df, max_selections=1)
    
    with c2:
        daq_button = st.form_submit_button(label="✨ DAQ")

#if not daq_button:
#    st.stop()

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

with tab3:
   st.header("Train")
   #st.image("https://static.streamlit.io/examples/owl.jpg", width=200)
   with st.form(key="train_form"):

    ce, c1, ce, c2, c3 = st.columns([0.07, 1, 0.07, 1, 0.07])
    with c1:
        option = st.selectbox('Select run',('6', '7', '9','10','11','12','13','14','15'))

    #with c2:
        
        #words = st.text_area("Enter search words:", height=10)
        #words = st.multiselect('Choose a Keyword Tag:', df, max_selections=1)
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
    #df = px.data.iris()
    #fig = px.scatter(
    #     df,
    #     x="sepal_width",
    #     color="sepal_length",
    #     color_continuous_scale="reds")
    #     y="sepal_length",)
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)



#if not submit_button:
#    st.stop()
#st.button('Get the data')



#d = st.date_input(
#    "Start time",
#    datetime.date(2019, 7, 6))
#st.write('Your birthday is:', d)

#dfm = pd.read_csv('data/merged.csv')

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