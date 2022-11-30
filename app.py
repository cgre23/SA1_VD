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


#if st.sidebar.checkbox("Show advanced options"):
    # Let the user pick which features to control with sliders.
    #speed = st.sidebar.number_input('Set animation speed', min_value=1, max_value=500, value=50, step=5, format=None, key='speed', help='None', label_visibility="visible")
st.sidebar.write("""The animation speed is fixed at 10 ms per frame.""")
    #control_features = st.sidebar.multiselect(
    #    "Exclude which cells?",
    #    ['Cell 1', 'Cell 2', 'Cell 3', 'Cell 4', 'Cell 5', 'Cell 6', 'Cell 7', 'Cell 8', 'Cell 9', 'Cell 10'],
   #     ['Cell 1'], help='This is still work in progress.'
    #)
#else:
    # Don't let the user pick feature values to control.
    #control_features = default_control_features
    #speed = 100

# Insert user-controlled values from sliders into the feature vector.

#st.sidebar.slider('Time', 0, 100, 50, 5)



st.sidebar.title("Note")
st.sidebar.write(
    """The app is still in development.
     """
)

st.sidebar.caption("Developed by: Christian Grech (DESY, MXL)")
st.sidebar.caption(f"Streamlit version `{st.__version__}`")


dfm = pd.read_pickle('doocs_orbit_20221021_viewer.pkl')
minCell = dfm['Cell No'].min()
maxCell = dfm['Cell No'].max()
minVal = min(dfm[['ValueX','ValueY']].min())
maxVal = max(dfm[['ValueX','ValueY']].max())
dfm['Time'] = dfm.index.strftime("%H:%M:%S.%f").str[:-5]

# Animation year by year basis

animation = px.line(data_frame=dfm,
          x= 'Cell No',
          y = ['ValueX', 'ValueY'],
          hover_name = 'Cell No',
          color_discrete_sequence=px.colors.qualitative.G10,
          markers=True,
          range_x = [minCell-0.2, maxCell+0.05],
          range_y = [minVal-0.2, maxVal+0.05],
          animation_frame='Time',
          height=650)
animation.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 10
animation.update_layout(
    {'plot_bgcolor': 'rgba(0, 0, 0, 0)'},
    margin=dict(l=50, r=50, t=40, b=20),
    legend_title_text='Variable',
    yaxis_title='Position',
    xaxis_title='Cell'
)
animation.update_yaxes(
    showline=True, linewidth=2, linecolor='black', showgrid=True
)
animation.update_xaxes(
    showline=True, linewidth=1, linecolor='black'
)
newnames = {'ValueX':'X.SA2.HIST', 'ValueY': 'Y.SA2.HIST'} # From the other post
animation.for_each_trace(lambda t: t.update(name = newnames[t.name]))
st.plotly_chart(animation, use_container_width=True)
