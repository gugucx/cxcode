# モジュールインポート
# import time
import streamlit as st
from streamlit import caching
import matplotlib.pyplot as plt
import scipy.stats as sts
import numpy as np
import pandas as pd
import json

# last_time = time.time()
# CACHE初期化
caching.clear_cache()

# ツールを選択
st.sidebar.title("Dscouter (Ver.1.0.0)")
st.sidebar.subheader("ツールを選択")
select_tool = st.sidebar.selectbox("", ["工程能力まとめ", "データ可視化"])
if select_tool == "工程能力まとめ":
    # データインプット
    st.title("工程能力まとめツール (Ver1.0.0)")
    upload_data = st.sidebar.file_uploader("", type=["csv"])
    if upload_data is not None:
        df = pd.read_csv(upload_data)
        column_names = df.columns.tolist()

        # パラメータ初期設定
        fig_width = 2.5
        fig_height = 1.2
        dpi_value = 96.0
        axis_font = 6.0
        sl_color = "red"
        sl_style = "-"
        sl_width = 1.0
        hist_color = "grey"
        hist_bins = 10
        curve_color = "tab:blue"
        curve_style = "-"
        curve_width = 2.0
        al_color = "green"
        al_style = "--"
        al_width = 1.0
        usl = 1.0
        lsl = -1.0
        k_lh = 1.2
        max_x = 100.0
        min_x = -100.0
        # パラメータ導入
        st.sidebar.subheader("共通設定を導入")
        upload_settings = st.sidebar.file_uploader("", type=["json"])
        if upload_settings is not None:
            para_load = json.load(upload_settings)
            fig_width = para_load["fig_width"]
            fig_height = para_load["fig_height"]
            dpi_value = para_load["dpi_value"]
            axis_font = para_load["axis_font"]
            sl_color = para_load["sl_color"]
            sl_style = para_load["sl_style"]
            sl_width = para_load["sl_width"]
            hist_color = para_load["hist_color"]
            hist_bins = para_load["hist_bins"]
            curve_color = para_load["curve_color"]
            curve_style = para_load["curve_style"]
            curve_width = para_load["curve_width"]
            al_color = para_load["al_color"]
            al_style = para_load["al_style"]
            al_width = para_load["al_width"]
            usl = para_load["usl"]
            lsl = para_load["lsl"]
            k_lh = para_load["k_lh"]
            max_x = para_load["max_x"]
            min_x = para_load["min_x"]

        # グラフ設定
        st.sidebar.subheader("グラフ設定")
        if st.sidebar.checkbox("グラフサイズ設定"):
            # グラフDPI
            dpi_value = st.sidebar.number_input(
                "グラフDPI", min_value=96.0, max_value=384.0, value=dpi_value, step=96.0
            )
            # グラフサイズ
            fig_width = st.sidebar.slider("グラフ幅(pix)", 0.1, 5.0, fig_width)
            fig_height = st.sidebar.slider("グラフ高さ(pix)", 0.1, 5.0, fig_height)
        # グラフ軸表示有無
        axis_setting = "axis_setting_no"
        if st.sidebar.checkbox("軸ラベルを表示"):
            axis_setting = "axis_setting_yes"
            axis_font = st.sidebar.slider("軸フォントサイズ", 1.0, 12.0, axis_font)
        # プロット設定
        st.sidebar.subheader("プロット設定")
        # 点図階級設定
        hist_bins = st.sidebar.slider("ヒストグラム階級設定", 1, 50, hist_bins)
        # プロット詳細設定
        if st.sidebar.checkbox("プロット線設定"):
            # 線高さ
            k_lh = st.sidebar.slider("線高さ", 0.0, 5.0, k_lh)
            st.sidebar.text("▼規格線関連設定")
            # 色
            sl_color = st.sidebar.selectbox("線色", [sl_color, "green", "tab:blue"])
            # 線タイプ
            sl_style = st.sidebar.selectbox("線タイプ", [sl_style, "--", "-.", ":"])
            # 線幅
            sl_width = st.sidebar.slider("線幅", 0.0, 3.0, sl_width)
            # 点図関連設定
            st.sidebar.text("▼ヒストグラム関連設定")
            # 色
            hist_color = st.sidebar.selectbox(
                "色", [hist_color, "tab:blue", "red", "green"]
            )
            # 正規分布曲線
            st.sidebar.text("▼正規分布曲線関連設定")
            # 色
            curve_color = st.sidebar.selectbox("線色　", [curve_color, "red", "green"])
            # 線タイプ
            curve_style = st.sidebar.selectbox("線タイプ　", [curve_style, "--", "-.", ":"])
            # 線幅
            curve_width = st.sidebar.slider("線幅　", 0.0, 3.0, curve_width)
            # アクションライン
            st.sidebar.text("▼アクションライン関連設定")
            # 色
            al_color = st.sidebar.selectbox("線色　　", [al_color, "tab:blue", "red"])
            # 線タイプ
            al_style = st.sidebar.selectbox("線タイプ　　", [al_style, "-", "-.", ":"])
            # 線幅
            al_width = st.sidebar.slider("線幅　　", 0.0, 3.0, al_width)
        # 工程能力線図プロット関数

        def pci_fig(column_name, column_data, usl, lsl, max_x, min_x):
            n = len(df[column_name])
            mu = df[column_name].mean()
            sig = column_data.std(ddof=1)
            lal = mu - 9 * sig
            ual = mu + 9 * sig
            # 上下限規格設定
            usl = st.number_input(str([column_name]) + "上限規格値　個別設定", value=usl)
            lsl = st.number_input(str([column_name]) + "下限規格値　個別設定", value=lsl)
            # プロット範囲設定
            if xlim_setting == "xlim_setting_yes":
                max_x = st.number_input(
                    str([column_name]) + "プロット上限範囲　個別設定", value=max_x
                )
                min_x = st.number_input(
                    str([column_name]) + "プロット下限範囲　個別設定", value=min_x
                )
            cpu = (usl - mu) / (3 * sig)
            cpl = (mu - lsl) / (3 * sig)
            cpk = min(cpu, cpl)
            nx = np.linspace(mu - 5.2 * sig, mu + 5.2 * sig, 1000)
            ny = sts.norm.pdf(nx, mu, sig)
            lh = max(ny) * k_lh
            # 表示
            # 工程能力指数表示
            st.success(
                str([column_name])
                + f"の工程能力まとめ： n={n:.0f}, Ave={mu:.2f}, σ={sig:.2f}, Cpk={cpk:.2f}, UAL={ual:.2f}, LAL={lal:.2f}"
            )
            # 図表示設定
            fig1 = plt.figure(dpi=dpi_value, figsize=[fig_width, fig_height])
            fig1.patch.set_alpha(0.0)
            sub_plot = fig1.add_subplot()
            sub_plot.patch.set_alpha(0.0)
            plt.rcParams["font.size"] = axis_font
            plt.tight_layout()
            if xlim_setting == "xlim_setting_yes":
                plt.xlim(min_x, max_x)
            if axis_setting == "axis_setting_no":
                plt.yticks([])
                plt.xticks([])
            plt.hist(
                df[column_name],
                bins=hist_bins,
                color=hist_color,
                rwidth=0.8,
                density=True,
                alpha=0.75,
            )
            plt.plot(
                nx, ny, color=curve_color, linewidth=curve_width, linestyle=curve_style
            )
            plt.vlines(
                usl, 0, lh, color=sl_color, linewidth=sl_width, linestyle=sl_style
            )
            plt.vlines(
                lsl, 0, lh, color=sl_color, linewidth=sl_width, linestyle=sl_style
            )
            plt.vlines(
                ual, 0, lh, color=al_color, linewidth=al_width, linestyle=al_style
            )
            plt.vlines(
                lal, 0, lh, color=al_color, linewidth=al_width, linestyle=al_style
            )
            return st.pyplot()

        # 出力設定
        if st.checkbox("CSVデータ表示"):
            st.dataframe(df)
        st.subheader("規格設定")
        st.warning("片側規格の場合、規格ない側を大きく設定してください（例：上限:1.00 ; 下限:-100.00）")

        usl = st.number_input("上限規格値　共通設定", value=usl)
        lsl = st.number_input("下限規格値　共通設定", value=lsl)
        # プロット範囲設定
        st.subheader("プロット範囲設定")
        xlim_setting = "xlim_setting_no"
        if st.checkbox("プロット範囲手動設定　(※片側規格の場合、設定必須)"):
            max_x = st.number_input("プロット上限範囲　共通設定", value=max_x)
            min_x = st.number_input("プロット下限範囲　共通設定", value=min_x)
            xlim_setting = "xlim_setting_yes"
        # 工程能力線図表示
        display_type = st.radio("", ("工程能力線図（項目別）", "工程能力線図（全項目）"))
        # # 単一項目選択表示
        if display_type == "工程能力線図（項目別）":
            select_column_names = st.selectbox("表示項目を選択", column_names)
            select_column_data = df[select_column_names]
            pci_fig(select_column_names, select_column_data, usl, lsl, max_x, min_x)
            plt.close()
            # st.write('it took {:.3f} seconds to process data'.format(
            #     time.time()-last_time))
        # 全項目表示
        if display_type == "工程能力線図（全項目）":
            for column_name in column_names:
                column_data = df[column_name]
                pci_fig(column_name, column_data, usl, lsl, max_x, min_x)
                plt.close()
            # st.write('it took {:.3f} seconds to process data'.format(
            #     time.time()-last_time))
        para = {
            "fig_width": fig_width,
            "fig_height": fig_height,
            "dpi_value": dpi_value,
            "axis_font": axis_font,
            "sl_color": sl_color,
            "sl_style": sl_style,
            "sl_width": sl_width,
            "hist_color": hist_color,
            "hist_bins": hist_bins,
            "curve_color": curve_color,
            "curve_style": curve_style,
            "curve_width": curve_width,
            "al_color": al_color,
            "al_style": al_style,
            "al_width": al_width,
            "usl": usl,
            "lsl": lsl,
            "k_lh": k_lh,
            "max_x": max_x,
            "min_x": min_x,
        }
        st.sidebar.subheader("共通設定を保存")
        if st.sidebar.button("settings.json に保存"):
            with open("settings.json", "w") as f:
                json.dump(para, f)
            path = "C:/Users/chenx/OneDrive/git/cxcode/"
            st.sidebar.success("設定をsettings.jsonに保存しました" + path)

# if select_tool=='データ可視化':
if select_tool == "データ可視化":
    # データインプット
    st.title("データ可視化 ツール")
    st.subheader("データアップロード (CSVファイル)")
    upload_data = st.sidebar.file_uploader("", type=["csv"])
    if upload_data is not None:
        df = pd.read_csv(upload_data)
        st.subheader("CSVデータ表示 ")
        if st.checkbox("全データ表示 "):
            st.dataframe(df)
        # データ可視化
        column_names = df.columns.tolist()
        st.subheader("グラフ表示")
        # 単一項目選択表示
        if st.checkbox("単一データ表示"):
            type_of_plot = st.selectbox("グラフタイプを選択", ["棒図", "線図"])
            select_column_names = st.selectbox("データを選択", column_names)

            # Plot By Streamlit
            if type_of_plot == "棒図":
                cust_data = df[select_column_names]
                st.bar_chart(cust_data)

            elif type_of_plot == "線図":
                cust_data = df[select_column_names]
                st.line_chart(cust_data)

        if st.checkbox("複数データ表示"):
            type_of_plot = st.selectbox("グラフタイプを選択", ["棒図", "線図"])
            select_column_names = st.multiselect("データを選択", column_names)

            # Plot By Streamlit
            if type_of_plot == "棒図":
                cust_data = df[select_column_names]
                st.bar_chart(cust_data)

            elif type_of_plot == "線図":
                cust_data = df[select_column_names]
                st.line_chart(cust_data)
