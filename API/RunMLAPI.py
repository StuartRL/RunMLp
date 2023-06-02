""" Starting basic web page API UI for RunML testing
CLI "streamlit run ./API/RunMLAPI.py"
ToDo:
1) Remove index column on results table
2) add all slider reset or restart app with defaults
3) Right justify title "RunML - date and time", text
4) Improve the model
"""

import streamlit as st
from datetime import datetime
from joblib import load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

header = st.container()
model_prediction = st.container()
dataset = st.container()
# features = st.container(df.set_index('Index'))

with header:
    st.title("COURSE TYPE PREDICTION")
    st.text(f"RunML - {datetime.now().strftime('%d/%m/%y %H:%M:%S')}")
    st.text(
        "Testing new data against a trained RFC model and historic Garmin running data"
    )
    st.text("ref. (10k=6.2 miles , Half marathon=13.1 miles, Full marathon=26.2 miles)")

rfc_model = load("../models/rfc.joblib")

st.sidebar.markdown(
    """
    **Model:**
    Feb-23; 767 running sessions; 13,039 data points; 9 course types; 17 features.
    """
)

# with model_training:
with st.sidebar:
    st.header("Session data")
    usr_Dist = st.slider(
        "1\) DISTANCE run (miles)", min_value=0.5, max_value=13.1, value=3.0, step=0.1
    )
    usr_Cal = st.slider(
        "2\) CALORIES consumed (c)", min_value=100, max_value=2000, value=300, step=50
    )
    usr_Time = st.slider(
        "3\) TIME taken (hrs)", min_value=0.5, max_value=2.5, value=0.5, step=0.1
    )
    usr_Seconds = usr_Time * 3600
    usr_AvgHR = st.slider(
        "4\) Average heart rate (bpm)", min_value=70, max_value=200, value=147, step=10
    )
    usr_MaxHR = st.slider(
        "5\) Maximum heart rate (bpm)", min_value=70, max_value=223, value=173, step=10
    )
    usr_AerobicTE = st.slider(
        "6\) Aerobic training effect (#)",
        min_value=0.0,
        max_value=5.0,
        value=1.5,
        step=0.1,
    )
    usr_AvgRunCadence = st.slider(
        "7\) Average Cadence (spm)", min_value=0, max_value=174, value=65, step=10
    )
    usr_MaxRunCadence = st.slider(
        "8\) Maximum Cadence (spm)", min_value=0, max_value=250, value=75, step=10
    )
    usr_AvgPace = st.slider(
        "9\) Average Pace (min/mile)",
        min_value=0.0,
        max_value=18.4,
        value=10.8,
        step=0.1,
    )
    usr_Seconds_AP = usr_AvgPace * 60
    usr_BestPace = st.slider(
        "10\) BEST PACE (min/mile)", min_value=0.0, max_value=22.8, value=7.8, step=0.1
    )
    usr_Seconds_BP = usr_BestPace * 60
    usr_TotalAscent = st.slider(
        "11\) Total Assent (m)", min_value=40, max_value=638, value=107, step=10
    )
    usr_TotalDescent = st.slider(
        "12\) TOTAL DESCENT (m)", min_value=30, max_value=643, value=110, step=10
    )
    usr_NumberofLaps = st.slider(
        "13\) Numer of laps (#)", min_value=1, max_value=20, value=3, step=1
    )
    usr_MovingTime = st.slider(
        "14\) Moving time (hrs)", min_value=0.5, max_value=2.3, value=0.35, step=0.1
    )
    usr_Seconds_MT = usr_MovingTime * 3600
    usr_ElapsedTime = st.slider(
        "15\) Elapsed time (hrs)", min_value=0.5, max_value=2.4, value=0.35, step=0.1
    )
    usr_Seconds_ET = usr_ElapsedTime * 3600
    usr_MinElevation = st.slider(
        "16\) Minimum elevation (m)", min_value=0, max_value=1000, value=11, step=10
    )
    usr_MaxElevation = st.slider(
        "17\) Maximum elevation (m)", min_value=0, max_value=1000, value=28, step=10
    )

# Results from model
with model_prediction:
    col1, col2 = st.columns(2)
    col1.markdown("**Course Prediction** (model)")
    pred_rfc = rfc_model.predict(
        [
            [
                usr_Dist,
                usr_Cal,
                usr_Seconds,
                usr_AvgHR,
                usr_MaxHR,
                usr_AerobicTE,
                usr_AvgRunCadence,
                usr_MaxRunCadence,
                usr_Seconds_AP,
                usr_Seconds_BP,
                usr_TotalAscent,
                usr_TotalDescent,
                usr_NumberofLaps,
                usr_Seconds_MT,
                usr_Seconds_ET,
                usr_MinElevation,
                usr_MaxElevation,
            ]
        ]
    )

    Courses = pd.DataFrame(
        {
            "Type of course": [
                "Fast",
                "Fast Hills",
                "Fast Nice Flat",
                "Okay",
                "Okay Flat",
                "Okay Hills",
                "Slow",
                "Slow Dangerous Hills",
                "Slow Flat",
            ]
        }
    )
    df = pd.concat([Courses, pd.DataFrame({"Prediction": pred_rfc[0, :]})], axis=1)
    df.index = np.arange(1, len(df) + 1)  # sequence index from 1
    # col1.dataframe(df)
    col1.header(
        ":red['" + str(df.loc[df["Prediction"] == 1, "Type of course"].iloc[0]) + "']"
    )

    # Most important features
    importances = rfc_model.feature_importances_
    df = pd.DataFrame(
        {
            "Score": importances,
            "Item": [
                "Distance",
                "Calories",
                "Time",
                "Average Heart Rate",
                "Maximum Heart Rate",
                "Aerobic TE",
                "Average Cadence",
                "Maximum Cadence",
                "Average Pace",
                "Best Pace",
                "Ascent",
                "Descent",
                "Laps",
                "Moving Time",
                "Elapsed Time",
                "Minimum Elevation",
                "Maximum Elevation",
            ],
        }
    )
    df.sort_values(by="Score", ascending=False, inplace=True)
    df.index = np.arange(1, len(df) + 1)  # sequence index from 1
    col1.markdown("**Feature importance** (model)")
    col1.dataframe(df, height=632)

    # Input values (from sliders)
    col2.markdown("**Input values** (sliders)")
    df = pd.DataFrame(
        {
            "Items": [
                "Distance",
                "Calories",
                "Time",
                "Average Heart Rate",
                "Maximum Heart Rate",
                "Aerobic TE",
                "Average Cadence",
                "Maximum Cadence",
                "Average Pace",
                "Best Pace",
                "Ascent",
                "Descent",
                "Laps",
                "Moving Time",
                "Elapsed Time",
                "Minimum Elevation",
                "Maximum Elevation",
            ],
            "Inputs": [
                usr_Dist,
                usr_Cal,
                usr_Seconds / 3600,
                usr_AvgHR,
                usr_MaxHR,
                usr_AerobicTE,
                usr_AvgRunCadence,
                usr_MaxRunCadence,
                usr_Seconds_AP / 60,
                usr_Seconds_BP / 60,
                usr_TotalAscent,
                usr_TotalDescent,
                usr_NumberofLaps,
                usr_Seconds_MT / 3600,
                usr_Seconds_ET / 3600,
                usr_MinElevation,
                usr_MaxElevation,
            ],
            "Units": [
                "miles",
                "c",
                "hrs",
                "bps",
                "bps",
                "#",
                "spm",
                "spm",
                "min/mile",
                "min/mile",
                "m",
                "m",
                "#;[miles]",
                "hrs",
                "hrs",
                "m",
                "m",
            ],
        }
    )
    df.index = np.arange(1, len(df) + 1)  # sequence index from 1
    col2.dataframe(df, height=632)  # bodge

# Supporting and other information
with dataset:
    st.header("Dataset (sample 4x48)")
    RunData = pd.read_csv("./Garmin/All Activities with Labels.csv")
    st.write(RunData.head(4))

    st.subheader("Popular Distances (App)")
    # st.bar_chart(pd.DataFrame(RunData["Distance"].value_counts()))

    fig, ax = plt.subplots()
    xxx = pd.DataFrame(RunData["Distance"])
    ax.hist(xxx, bins=20)
    plt.title("Histogram of Session Distances")
    plt.xlabel("Miles per Session")
    plt.ylabel("Number of sessions")
    plt.grid(linestyle="--")
    plt.text(12, 150, "N=" + str(len(RunData.index)))
    st.pyplot(fig)

st.markdown(
    """Notes
* Course classification obtained from new performance data
* 'Best Pace' has good effect
* Model is not great!
"""
)
