import plotly.express as px
import streamlit as st
import pandas as pd


#import the data available in plotly.express
gapminder_df = px.data.gapminder()

st.set_page_config(
    page_title="SASE1 Virtual Diagnostics", layout="wide"
)

st.title("SASE1 Orbit Viewer")
"""This demo demonstrates SASE2 orbit visualization using DOOCS HIST data for 21.10.2022, local time."""


st.sidebar.title("Settings")


    # Can be used wherever a "file-like" object is accepted:
    #st.write(dataframe)
# If the user doesn't want to select which features to control, these will be used.


if st.sidebar.checkbox("Show advanced options"):
    # Let the user pick which features to control with sliders.
    speed = st.sidebar.number_input('Set animation speed', min_value=1, max_value=500, value=50, step=5, format=None, key='speed', help='None', label_visibility="visible")
#st.sidebar.write("""The animation speed is fixed at 10 ms per frame.""")
    #control_features = st.sidebar.multiselect(
    #    "Exclude which cells?",
    #    ['Cell 1', 'Cell 2', 'Cell 3', 'Cell 4', 'Cell 5', 'Cell 6', 'Cell 7', 'Cell 8', 'Cell 9', 'Cell 10'],
   #     ['Cell 1'], help='This is still work in progress.'
    #)
else:
    # Don't let the user pick feature values to control.
    #control_features = default_control_features
    speed = 100

# Insert user-controlled values from sliders into the feature vector.

st.sidebar.slider('Time', 0, 100, 50, 5)



st.sidebar.title("Note")
st.sidebar.write(
    """The app is still in development.
     """
)

st.sidebar.caption("Developed by: Christian Grech (DESY, MXL)")
st.sidebar.caption(f"Streamlit version `{st.__version__}`")


dfm = pd.read_pickle('merged.pkl')


# Animation year by year basis


