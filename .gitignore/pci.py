# import os
import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as sts
import plotly.figure_factory as ff

# import math
import matplotlib.pyplot as plt
# import matplotlib
# import matplotlib.ticker as ticker
# import matplotlib.transforms as ts

# Title
st.title("Process Capability Index Visualization")

# Data Upload
st.header("Dataset Upload")
data = st.file_uploader("", type=["csv"])
if data is not None:
    df = pd.read_csv(data)

# PCI Graph Property Setting
st.header("Process Capability Index")

# plot setting
st.sidebar.subheader("PCI GRAPH SETTING")
USL = st.sidebar.number_input('input USL', value=6.77)
LSL = st.sidebar.number_input('input LSL', value=-6.77)
dpi_value = st.sidebar.number_input(
    'input fig_dpi', min_value=96.0, max_value=384.0, value=96.0, step=96.0)
k = st.sidebar.slider('Select 区間の幅', 0.1, 10.0, 2.0)
histbins = st.sidebar.slider('Select 階級数', 1, 10, 5)
plotlyhistbins = st.sidebar.slider('Select interactive 階級数', 0.01, 2.00, 0.10)
lh = st.sidebar.slider('Select line height', 1.0, 1000.0, 1000.0)
figwidth = st.sidebar.slider('Select fig width（pix）', 0.1, 5.0, 2.5)
figheight = st.sidebar.slider('Select fig height（pix）', 0.1, 5.0, 1.0)
histwidth = st.sidebar.slider('Select hist width（pix）', 0.1, 10.0, 2.0)
Curvewidth = st.sidebar.slider('Select Curve width（pix）', 0.1, 3.0, 2.0)
SLwidth = st.sidebar.slider('Select Spec Limits width（pix）', 0.1, 3.0, 1.0)
ALwidth = st.sidebar.slider('Select Action Limits width（pix）', 0.1, 3.0, 1.0)
histcolor = st.sidebar.selectbox(
    'Select hist Color', ["blue", "tab:blue", "red", "green"])
Curvecolor = st.sidebar.selectbox(
    'Select Curve Color', ["tab:blue", "red", "green"])
SLcolor = st.sidebar.selectbox('Select Spec Limits Color', [
                               "red", "green", "tab:blue"])
ALcolor = st.sidebar.selectbox('Select Action Limits Color', [
                               "green", "tab:blue", "red"])
Curvestyle = st.sidebar.selectbox('Select Curve style', ['-', '--', '-.', ':'])
SLstyle = st.sidebar.selectbox(
    'Select Spec Limits style', ['-', '--', '-.', ':'])
ALstyle = st.sidebar.selectbox(
    'Select Action Limits style', ['--', '-', '-.', ':'])


plt.figure(dpi=dpi_value, figsize=[figwidth, figheight])

# Data Process
if st.checkbox("Sellect Column to Show PCI Graph"):
    PCIall_columns_names = df.columns.tolist()
    PCIselected_columns_names = st.selectbox(
        "Select Columns To Plot PCI", PCIall_columns_names)
    PCI_data = df[PCIselected_columns_names]
    group_labels = [PCIselected_columns_names]
    # 統計処理
    n = len(df[PCIselected_columns_names])
    mu = df[PCIselected_columns_names].mean()
    sig = PCI_data.std(ddof=1)
    LAL = mu-9*sig
    UAL = mu+9*sig
    CPU = (USL - mu) / (3*sig)
    CPL = (mu - LSL) / (3*sig)
    Cpk = min(CPU, CPL)
    st.write(
        f'■ 個数：{n:.0f}、平均：{mu:.2f}、標準偏差：{sig:.2f}、Cpk値：{Cpk:.2f}、UAL：{UAL:.2f}、LAL：{LAL:.2f}')
    plt.figure(dpi=dpi_value, figsize=[figwidth, figheight])
    # plt.xlim(x_min,x_max)
    plt.yticks([])
    plt.xticks([])
    nx = np.linspace(mu-5.2*sig, mu+5.2*sig, 1000)
    ny = sts.norm.pdf(nx, mu, sig) * k * len(df[PCIselected_columns_names])
    plotlyfig = ff.create_distplot([df[PCIselected_columns_names]],
                                   group_labels, bin_size=plotlyhistbins, curve_type='normal')
    st.plotly_chart(plotlyfig, use_container_width=True)
    plt.hist(df[PCIselected_columns_names], bins=histbins,
             color=histcolor, rwidth=histwidth)
    plt.plot(nx, ny, color=Curvecolor,
             linewidth=Curvewidth, linestyle=Curvestyle)
    # 規格線出力
    plt.vlines(USL, 0, lh, color=SLcolor, linewidth=SLwidth, linestyle=SLstyle)
    plt.vlines(LSL, 0, lh, color=SLcolor, linewidth=SLwidth, linestyle=SLstyle)
    plt.vlines(UAL, 0, lh, color=ALcolor, linewidth=ALwidth, linestyle=ALstyle)
    plt.vlines(LAL, 0, lh, color=ALcolor, linewidth=ALwidth, linestyle=ALstyle)
    st.pyplot()
if st.button("Save Graph"):
    plt.savefig(str(PCIselected_columns_names)+'.png',
                format='png', bbox_inches='tight', transparent=True)
    plt.close()

if st.checkbox("Show All PCI Graph"):
    PCIall_columns_names = df.columns.tolist()
# 統計処理
    for PCIall_columns_name in PCIall_columns_names:
        PCI_data = df[PCIall_columns_name]
        n = len(df[PCIall_columns_name])
        mu = df[PCIall_columns_name].mean()
        sig = PCI_data.std(ddof=1)
        LAL = mu-9*sig
        UAL = mu+9*sig
        CPU = (USL - mu) / (3*sig)
        CPL = (mu - LSL) / (3*sig)
        Cpk = min(CPU, CPL)
        st.write(
            f'■ 個数：{n:.0f}、平均：{mu:.2f}、標準偏差：{sig:.2f}、Cpk値：{Cpk:.2f}、UAL：{Cpk:.2f}、LAL：{Cpk:.2f}')
        plt.figure(dpi=dpi_value, figsize=[figwidth, figheight])
        # plt.xlim(x_min,x_max)
        plt.yticks([])
        plt.xticks([])
        nx = np.linspace(mu-5.2*sig, mu+5.2*sig, 1000)
        ny = sts.norm.pdf(nx, mu, sig) * k * len(df[PCIall_columns_name])

        plt.plot(nx, ny, color=Curvecolor,
                 linewidth=Curvewidth, linestyle=Curvestyle)
        # 規格線出力
        plt.vlines(USL, 0, lh, color=SLcolor,
                   linewidth=SLwidth, linestyle=SLstyle)
        plt.vlines(LSL, 0, lh, color=SLcolor,
                   linewidth=SLwidth, linestyle=SLstyle)
        plt.vlines(UAL, 0, lh, color=ALcolor,
                   linewidth=ALwidth, linestyle=ALstyle)
        plt.vlines(LAL, 0, lh, color=ALcolor,
                   linewidth=ALwidth, linestyle=ALstyle)
        st.pyplot()

# Explore Data
st.header("Explore Dataset")
st.subheader("Dataset Preview")
if st.checkbox("View All Dataset"):
    st.dataframe(df)

# Select Columns
    if st.checkbox("Select Dataset Columns"):
        all_columns = df.columns.tolist()
        selected_columns = st.multiselect("Select Columns", all_columns)
        new_df = df[selected_columns]
        st.dataframe(new_df)

# Plot and Visualization
st.subheader("Data Visualization")
# Customizable Plot
if st.checkbox("Show Data visualization"):
    all_columns_names = df.columns.tolist()
    type_of_plot = st.selectbox("Select Type of Plot", [
                                "bar", "line", "hist", "box"])
    selected_columns_names = st.multiselect(
        "Select Columns To Plot", all_columns_names)

    if st.button("Generate Plot"):
        st.success("Generating Customizable Plot of {} for {}".format(
            type_of_plot, selected_columns_names))

    # Plot By Streamlit
    if type_of_plot == 'bar':
        cust_data = df[selected_columns_names]
        st.bar_chart(cust_data)

    elif type_of_plot == 'line':
        cust_data = df[selected_columns_names]
        st.line_chart(cust_data)

    # Custom Plot
    elif type_of_plot:
        cust_plot = df[selected_columns_names].plot(kind=type_of_plot)
        st.write(cust_plot)
        st.pyplot()

# Graph x,y axis display
