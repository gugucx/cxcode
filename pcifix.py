# モジュールインポート
import os
import tkinter as tk
from tkinter import filedialog
import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

# 保存関数


def save_path():
    save_window = tk.Tk()
    save_window.withdraw()
    save_address = filedialog.askdirectory(parent=save_window, initialdir=os.getcwd(
    ), title="Please select a folder:")
    return save_address


# ツールを選択
st.sidebar.subheader("ツールを選択")
select_tool = st.sidebar.selectbox('', ["工程能力まとめ", "データ可視化"])
if select_tool == '工程能力まとめ':
    # 設定(サイドバー)

    # 初期設定(今後jasonで辞書化)
    fig_width = 2.5
    fig_height = 1.0
    dpi_value = 96.0
    sl_color = "red"
    sl_style = '-'
    sl_width = 1.0
    hist_color = "blue"
    hist_bins = 10
    curve_color = "tab:blue"
    curve_style = "-"
    curve_width = 2.0
    al_color = "green"
    al_style = "--"
    al_width = 1.0

    # グラフ設定
    st.sidebar.subheader("グラフ設定")
    # グラフサイズ
    fig_width = st.sidebar.slider('グラフ幅', 0.1, 5.0, fig_width)
    fig_height = st.sidebar.slider('グラフ高さ', 0.1, 5.0, fig_height)
    # グラフDPI
    dpi_value = st.sidebar.number_input(
        'グラフdpi', min_value=96.0, max_value=384.0, value=dpi_value, step=96.0)
    # グラフ背景色
    # グラフ軸値表示有無
    # グラフ枠表示有無
    # グラフ軸表示有無

    # プロット設定
    st.sidebar.subheader("プロット設定")
    # 線高さ
    k_lh = st.sidebar.slider('線高さ', 1.0, 5.0, 1.2)
    if st.sidebar.checkbox("プロット詳細設定"):
        # 色
        sl_color = st.sidebar.selectbox('規格線色', ["red", "green", "tab:blue"])
        # 範囲
        # 線タイプ
        sl_style = st.sidebar.selectbox('規格線タイプ', ['-', '--', '-.', ':'])
        # 線幅
        sl_width = st.sidebar.slider('規格線幅', 0.1, 3.0, sl_width)
        # 高さ
        # ヒストグラム関連設定
        # 度数
        # 階級
        hist_bins = st.sidebar.slider('階級', 1, 50, hist_bins)
        # 色
        hist_color = st.sidebar.selectbox(
            'ヒストグラム色', ["blue", "tab:blue", "red", "green"])
        # 範囲
        # 幅
        # 正規分布曲線
        # 色
        curve_color = st.sidebar.selectbox(
            '正規分布曲線色', ["tab:blue", "red", "green"])
        # 範囲
        # 線タイプ
        curve_style = st.sidebar.selectbox('正規分布曲線タイプ', ['-', '--', '-.', ':'])
        # 線幅
        curve_width = st.sidebar.slider('正規分布曲線幅', 0.1, 3.0, curve_width)
        # 中心線
        # 色
        # 範囲
        # 線タイプ
        # 線幅
        # アクションライン
        # 色
        al_color = st.sidebar.selectbox(
            'アクションライン色', ["green", "tab:blue", "red"])
        # 範囲
        # 線タイプ
        al_style = st.sidebar.selectbox('アクションラインタイプ', ['--', '-', '-.', ':'])
        # 線幅
        al_width = st.sidebar.slider('アクションライン幅', 0.1, 3.0, al_width)

    # 出力設定

    # データインプット
    st.title("工程能力まとめ ツール")
    st.subheader("データアップロード (CSVファイル)")
    upload_data = st.file_uploader("", type=["csv"])
    if upload_data is not None:
        df = pd.read_csv(upload_data)
        st.subheader("アップロードデータ表示")
        if st.checkbox("全データ表示"):
            st.dataframe(df)
        # 4.データプロセス
        column_names = df.columns.tolist()
        st.subheader("工程能力線図表示 (項目別)")
        # 単一項目選択表示
        select_column_names = st.selectbox('表示項目を選択', column_names)
        select_column_data = df[select_column_names]
        # class
        n = len(df[select_column_names])
        # k = int(math.sqrt(n))
        # max_x = df[select_column_names].max()
        # min_x = df[select_column_names].min()
        # hist_bins = int(abs((max_x-min_x) / k))
        mu = df[select_column_names].mean()
        sig = select_column_data.std(ddof=1)
        lal = mu-9*sig
        ual = mu+9*sig
        usl = ual*1.1
        lsl = lal*1.1
        cpu = (usl - mu) / (3*sig)
        cpl = (mu - lsl) / (3*sig)
        cpk = min(cpu, cpl)
        nx = np.linspace(mu-5.2*sig, mu+5.2*sig, 1000)
        ny = sts.norm.pdf(nx, mu, sig)
        lh = max(ny)*k_lh
        # 規格線
        usl = st.number_input('上限規格値', value=usl)
        lsl = st.number_input('下限規格値', value=lsl)
        # 工程能力指数
        st.write(
            f'■ 個数：{n:.0f}、平均：{mu:.2f}、標準偏差：{sig:.2f}、Cpk値：{cpk:.2f}、UAL：{ual:.2f}、LAL：{lal:.2f}')
        # 工程能力線図
        fig1 = plt.figure(dpi=dpi_value, figsize=[fig_width, fig_height])
        fig1.patch.set_alpha(0.0)
        ax = fig1.add_subplot()
        ax.patch.set_alpha(0.0)
        plt.yticks([])
        plt.xticks([])
        plt.hist(df[select_column_names], bins=hist_bins,
                 color=hist_color, rwidth=0.8, density=True, alpha=0.75)
        plt.plot(nx, ny, color=curve_color,
                 linewidth=curve_width, linestyle=curve_style,)
        plt.vlines(usl, 0, lh, color=sl_color,
                   linewidth=sl_width, linestyle=sl_style)
        plt.vlines(lsl, 0, lh, color=sl_color,
                   linewidth=sl_width, linestyle=sl_style)
        plt.vlines(ual, 0, lh, color=al_color,
                   linewidth=al_width, linestyle=al_style)
        plt.vlines(lal, 0, lh, color=al_color,
                   linewidth=al_width, linestyle=al_style)
        # 工程能力指数表生成（項目別）
        # データアウトプット
        # python打包到本地的zipfile,然后通过html herf来下载 参考test.py
        if st.button("画像を保存"):
            saved_path = save_path()
            fig1.savefig(saved_path+"/"+str(select_column_names)+'.png',
                         bbox_inches='tight', transparent=True)
            st.success("画像を　"+saved_path+"　に保存しました")
        st.pyplot()
        plt.close()
        # 全項目表示
        st.subheader("工程能力線図表示(全項目)")
        if st.checkbox("全項目表示"):
            # 上下限規格設定
            usl = st.number_input('全上限規格値', value=usl)
            lsl = st.number_input('全下限規格値', value=lsl)
            for column_name in column_names:
                column_data = df[column_name]
                # class
                n = len(df[column_name])
                # k = int(math.sqrt(n))
                # max_x = df[column_name].max()
                # min_x = df[column_name].min()
                # hist_bins = int(abs((max_x-min_x) / k))
                mu = df[column_name].mean()
                sig = column_data.std(ddof=1)
                lal = mu-9*sig
                ual = mu+9*sig
                cpu = (usl - mu) / (3*sig)
                cpl = (mu - lsl) / (3*sig)
                cpk = min(cpu, cpl)
                nx = np.linspace(mu-5.2*sig, mu+5.2*sig, 1000)
                ny = sts.norm.pdf(nx, mu, sig)
                lh = max(ny)*k_lh
                # 表示(中央画面)
                # 工程能力指数
                st.write(
                    f'■ 個数：{n:.0f}、平均：{mu:.2f}、標準偏差：{sig:.2f}、Cpk値：{cpk:.2f}、UAL：{ual:.2f}、LAL：{lal:.2f}')
                # 工程能力線図
                plt.figure(dpi=dpi_value, figsize=[fig_width, fig_height])
                plt.yticks([])
                plt.xticks([])
                plt.hist(df[column_name], bins=hist_bins,
                         color=hist_color, rwidth=0.8, density=True, alpha=0.75)
                plt.plot(nx, ny, color=curve_color,
                         linewidth=curve_width, linestyle=curve_style)
                plt.vlines(usl, 0, lh, color=sl_color,
                           linewidth=sl_width, linestyle=sl_style)
                plt.vlines(lsl, 0, lh, color=sl_color,
                           linewidth=sl_width, linestyle=sl_style)
                plt.vlines(ual, 0, lh, color=al_color,
                           linewidth=al_width, linestyle=al_style)
                plt.vlines(lal, 0, lh, color=al_color,
                           linewidth=al_width, linestyle=al_style)
                st.pyplot()
            # 全画像保存
            if st.button("全画像を保存"):
                saved_path = save_path()
                for column_name in column_names:
                    column_data = df[column_name]
                    # class
                    n = len(df[column_name])
                    # k = int(math.sqrt(n))
                    # max_x = df[column_name].max()
                    # min_x = df[column_name].min()
                    # hist_bins = int(abs((max_x-min_x) / k))
                    mu = df[column_name].mean()
                    sig = column_data.std(ddof=1)
                    lal = mu-9*sig
                    ual = mu+9*sig
                    cpu = (usl - mu) / (3*sig)
                    cpl = (mu - lsl) / (3*sig)
                    cpk = min(cpu, cpl)
                    nx = np.linspace(mu-5.2*sig, mu+5.2*sig, 1000)
                    ny = sts.norm.pdf(nx, mu, sig)
                    lh = max(ny)*k_lh
                    # 工程能力線図
                    plt.figure(dpi=dpi_value, figsize=[fig_width, fig_height])
                    plt.yticks([])
                    plt.xticks([])
                    plt.hist(df[column_name], bins=hist_bins,
                             color=hist_color, rwidth=0.8, density=True, alpha=0.75)
                    plt.plot(nx, ny, color=curve_color,
                             linewidth=curve_width, linestyle=curve_style)
                    plt.vlines(usl, 0, lh, color=sl_color,
                               linewidth=sl_width, linestyle=sl_style)
                    plt.vlines(lsl, 0, lh, color=sl_color,
                               linewidth=sl_width, linestyle=sl_style)
                    plt.vlines(ual, 0, lh, color=al_color,
                               linewidth=al_width, linestyle=al_style)
                    plt.vlines(lal, 0, lh, color=al_color,
                               linewidth=al_width, linestyle=al_style)
                    # データアウトプット
                    plt.savefig(saved_path+"/"+str(column_name)+'.png',
                                bbox_inches='tight', transparent=True)
                    plt.close()
                st.success("画像を　"+saved_path+"　に保存しました")
# if select_tool=='データ可視化':
if select_tool == 'データ可視化':
    # データインプット
    st.title("データ可視化 ツール")
    st.subheader("データアップロード (CSVファイル)")
    upload_data = st.file_uploader("", type=["csv"])
    if upload_data is not None:
        df = pd.read_csv(upload_data)
        st.subheader("アップロードデータ表示 ")
        if st.checkbox("全データ表示 "):
            st.dataframe(df)
        # データ可視化
        column_names = df.columns.tolist()
        st.subheader("グラフ表示")
        # 単一項目選択表示
        if st.checkbox("単一データ表示"):
            type_of_plot = st.selectbox("グラフタイプを選択", [
                                        "ヒストグラム", "棒図", "線図"])
            select_column_names = st.selectbox('データを選択', column_names)

            # Plot By Streamlit
            if type_of_plot == 'ヒストグラム':
                select_column_data = df[select_column_names]
                labels = [select_column_names]
                plotly_histbins = st.slider('階級数', 0.01, 2.00, 0.10)
                plotly_fig = ff.create_distplot([df[select_column_names]],
                                                labels, bin_size=plotly_histbins, curve_type='normal')
                st.plotly_chart(plotly_fig, use_container_width=True)

            if type_of_plot == '棒図':
                cust_data = df[select_column_names]
                st.bar_chart(cust_data)

            elif type_of_plot == '線図':
                cust_data = df[select_column_names]
                st.line_chart(cust_data)

        if st.checkbox("複数データ表示"):
            type_of_plot = st.selectbox("グラフタイプを選択", ["棒図", "線図"])
            select_column_names = st.multiselect('データを選択', column_names)

            # Plot By Streamlit
            if type_of_plot == '棒図':
                cust_data = df[select_column_names]
                st.bar_chart(cust_data)

            elif type_of_plot == '線図':
                cust_data = df[select_column_names]
                st.line_chart(cust_data)
