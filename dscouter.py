# モジュールインポート
# import time
import streamlit as st
from streamlit import caching
import matplotlib.pyplot as plt
import scipy.stats as sts
import numpy as np
import pandas as pd

# last_time = time.time()

# CACHE初期化
caching.clear_cache()

# ツールを選択
st.sidebar.title("Dscouter (Ver.1.0.0)")
select_tool = st.sidebar.selectbox("ツールを選択", ["工程能力まとめ", "データ可視化"])
if select_tool == "工程能力まとめ":
    # データインプット
    st.title("工程能力まとめ (Ver1.0.0)")

    upload_data = st.file_uploader("", type=["csv"])
    if upload_data is not None:
        df = pd.read_csv(upload_data)
        column_names = df.columns.tolist()

        # 設定(サイドバー)
        # 設定を保存
        # 片側規格の場合
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
        max_x = 100
        min_x = -100

        # 共通規格値設定
        st.sidebar.subheader("規格値共通設定")
        usl = st.sidebar.number_input("上限規格値　共通設定", value=usl)
        lsl = st.sidebar.number_input("下限規格値　共通設定", value=lsl)
        sl_setting = "sl_setting_no"
        if st.sidebar.checkbox("規格値　個別設定"):
            sl_setting = "sl_setting_yes"
        # グラフ設定
        st.sidebar.subheader("グラフ設定")
        if st.sidebar.checkbox("グラフ詳細設定"):
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
        hist_bins = st.sidebar.slider("点図階級", 1, 50, hist_bins)
        # プロット範囲設定
        pltarea_setting = "pltarea_setting_no"
        if st.sidebar.checkbox("プロット範囲 設定"):
            max_x = st.sidebar.number_input("プロット上限範囲　共通設定", value=max_x)
            min_x = st.sidebar.number_input("プロット下限範囲　共通設定", value=min_x)
            pltarea_setting = "pltarea_setting_yes"
        # プロット詳細設定
        if st.sidebar.checkbox("その他プロット詳細設定"):
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
            st.sidebar.text("▼点図関連設定")
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
            if sl_setting == "sl_setting_yes":
                usl = st.number_input(str([column_name]) + "上限規格値　個別設定", value=usl)
                lsl = st.number_input(str([column_name]) + "下限規格値　個別設定", value=lsl)
            # プロット範囲設定
            if pltarea_setting == "pltarea_setting_yes":
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
                + f"： n={n:.0f}, Ave={mu:.2f}, σ={sig:.2f}, Cpk={cpk:.2f}, UAL={ual:.2f}, LAL={lal:.2f}"
            )
            # 図表示設定
            fig1 = plt.figure(dpi=dpi_value, figsize=[fig_width, fig_height])
            fig1.patch.set_alpha(0.0)
            sub_plot = fig1.add_subplot()
            sub_plot.patch.set_alpha(0.0)
            plt.rcParams["font.size"] = axis_font
            plt.tight_layout()
            if pltarea_setting == "pltarea_setting_yes":
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
        if st.checkbox("アップロードデータ表示"):
            st.dataframe(df)
        # 工程能力線図表示
        st.subheader("工程能力線図表示")
        display_type = st.radio("", ("項目別表示", "全項目表示"))
        # # 単一項目選択表示
        if display_type == "項目別表示":
            select_column_names = st.selectbox("表示項目を選択", column_names)
            select_column_data = df[select_column_names]
            pci_fig(select_column_names, select_column_data, usl, lsl, max_x, min_x)
            plt.close()
            # st.write('it took {:.3f} seconds to process data'.format(
            #     time.time()-last_time))
        # 全項目表示
        if display_type == "全項目表示":
            for column_name in column_names:
                column_data = df[column_name]
                pci_fig(column_name, column_data, usl, lsl, max_x, min_x)
                plt.close()
            # st.write('it took {:.3f} seconds to process data'.format(
            #     time.time()-last_time))

# if select_tool=='データ可視化':
if select_tool == "データ可視化":
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
