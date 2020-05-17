# モジュールインポート
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sts
import numpy as np
import pandas as pd
import json

# CACHE初期化
st.caching.clear_cache()

# ツールを選択
st.sidebar.title("Dscouter (Ver.1.0.0)")
select_tool = st.sidebar.selectbox("ツール選択", ["工程能力まとめ", "データ可視化"])
if select_tool == "工程能力まとめ":
    st.title("工程能力まとめツール")
    # 共通設定値初期化
    # plot area settings
    plot_area_setting = "plot_area_setting_no"
    plot_area_individual_setting = "plot_area_individual_setting_no"
    plot_area_max_x = 100.0
    plot_area_min_x = -100.0
    # axis label settings
    axis_label_setting = "axis_label_setting_no"
    axis_label_font = 6.0
    # graph size settings
    graph_size_width = 2.5
    graph_size_height = 1.2
    graph_size_dpi = 96.0
    # hist graph settings
    hist_graph_color = "grey"
    hist_graph_bins = 10
    # spec limit settings
    spec_limit_setting = "spec_limit_setting_no"
    spec_limit_type_setting = "dual_spec_limit"
    spec_limit_color = "red"
    spec_limit_style = "-"
    spec_limit_width = 1.0
    spec_limit_upper = 1.0
    spec_limit_lower = -1.0
    spec_limit_single = 1.0
    # normal distribution curve settings
    curve_color = "tab:blue"
    curve_style = "-"
    curve_width = 2.0
    # action line settings
    action_line_color = "green"
    action_line_style = "--"
    action_line_width = 1.0
    # line height settings
    line_height_k = 1.2

    # データアップロード
    upload_data = st.sidebar.file_uploader("データアップロード", type=["csv"])
    if upload_data is not None:
        df = pd.read_csv(upload_data)
        column_names = df.columns.tolist()
        # 元データの表示
        if st.checkbox("元データの表示"):
            st.dataframe(df)
        # 共通設定の導入
        upload_settings = st.sidebar.file_uploader("共通設定導入", type=["json"])
        if upload_settings is not None:
            settings_load = json.load(upload_settings)
            # plot area settings
            plot_area_setting = settings_load["plot_area_setting"]
            plot_area_individual_setting = settings_load["plot_area_individual_setting"]
            plot_area_max_x = settings_load["plot_area_max_x"]
            plot_area_min_x = settings_load["plot_area_min_x"]
            # axis label settings
            axis_label_setting = settings_load["axis_label_setting"]
            axis_label_font = settings_load["axis_label_font"]
            # graph size settings
            graph_size_width = settings_load["graph_size_width"]
            graph_size_height = settings_load["graph_size_height"]
            graph_size_dpi = settings_load["graph_size_dpi"]
            # hist graph settings
            hist_graph_color = settings_load["hist_graph_color"]
            hist_graph_bins = settings_load["hist_graph_bins"]
            # spec limit settings
            spec_limit_setting = settings_load["spec_limit_setting"]
            spec_limit_type_setting = settings_load["spec_limit_type_setting"]
            spec_limit_color = settings_load["spec_limit_color"]
            spec_limit_style = settings_load["spec_limit_style"]
            spec_limit_width = settings_load["spec_limit_width"]
            spec_limit_upper = settings_load["spec_limit_upper"]
            spec_limit_lower = settings_load["spec_limit_lower"]
            # normal distribution curve settings
            curve_color = settings_load["curve_color"]
            curve_style = settings_load["curve_style"]
            curve_width = settings_load["curve_width"]
            # action line settings
            action_line_color = settings_load["action_line_color"]
            action_line_style = settings_load["action_line_style"]
            action_line_width = settings_load["action_line_width"]
            # line height settings
            line_height_k = settings_load["line_height_k"]
        # 両側、片側規格選定
        limit_type = st.sidebar.radio("規格種類選択", ("両側規格", "片側規格"))
        # # 単一項目選択表示
        if limit_type == "両側規格":
            spec_limit_type_setting = "dual_spec_limit"
            # 共通規格値入力
            spec_limit_upper = st.sidebar.number_input(
                "上限規格値(共通)を入力", value=spec_limit_upper
            )
            spec_limit_lower = st.sidebar.number_input(
                "下限規格値(共通)を入力", value=spec_limit_lower
            )
        else:
            spec_limit_type_setting = "single_spec_limit"
            # 片側規格値入力
            spec_limit_single = st.sidebar.number_input(
                "片側規格値(共通)を入力", value=spec_limit_single
            )
        if st.sidebar.checkbox("規格値の個別設定"):
            spec_limit_setting = "spec_limit_setting_yes"
        # プロット範囲設定
        if st.checkbox("プロット範囲の設定"):
            plot_area_setting = "plot_area_setting_yes"
            plot_area_max_x = st.number_input("プロット上限範囲(共通)を入力", value=plot_area_max_x)
            plot_area_min_x = st.number_input("プロット下限範囲(共通)を入力", value=plot_area_min_x)
            if st.checkbox("プロット範囲を個別設定"):
                plot_area_individual_setting = "plot_area_individual_setting_yes"
        # 工程能力線図の表示設定
        if st.checkbox("工程能力線図表示内容の詳細設定"):
            # グラフ軸表示有無
            if st.checkbox("軸ラベルの表示"):
                axis_label_setting = "axis_label_setting_yes"
                axis_label_font = st.slider(
                    "軸フォントサイズ設定", 1.0, 12.0, axis_label_font, step=0.5
                )
            if st.checkbox("グラフサイズ設定"):
                # グラフDPI
                graph_size_dpi = st.number_input(
                    "グラフDPI",
                    min_value=96.0,
                    max_value=384.0,
                    value=graph_size_dpi,
                    step=96.0,
                )
                # グラフサイズ
                graph_size_width = st.slider(
                    "グラフ幅(pix)", 0.1, 5.0, graph_size_width, step=0.1
                )
                graph_size_height = st.slider(
                    "グラフ高さ(pix)", 0.1, 5.0, graph_size_height, step=0.1
                )
            # ヒストグラム詳細設定
            if st.checkbox("ヒストグラム詳細設定"):
                hist_graph_bins = st.slider("ヒストグラム階級設定", 1, 50, hist_graph_bins)
                # 色
                hist_graph_color = st.selectbox(
                    "色", [hist_graph_color, "tab:blue", "red", "green"]
                )
            # プロット詳細設定
            if st.checkbox("規格線詳細設定"):
                # 色
                spec_limit_color = st.selectbox(
                    "規格線色", [spec_limit_color, "green", "tab:blue"]
                )
                # 線タイプ
                spec_limit_style = st.selectbox(
                    "規格線タイプ", [spec_limit_style, "--", "-.", ":"]
                )
                # 線幅
                spec_limit_width = st.slider(
                    "規格線幅", 0.0, 3.0, spec_limit_width, step=0.5
                )
                # 正規分布曲線
            if st.checkbox("正規分布曲線詳細設定"):
                # 色
                curve_color = st.selectbox("正規分布曲線色", [curve_color, "red", "green"])
                # 線タイプ
                curve_style = st.selectbox("正規分布曲線タイプ", [curve_style, "--", "-.", ":"])
                # 線幅
                curve_width = st.slider("正規分布曲線幅", 0.0, 3.0, curve_width, step=0.5)
                # アクションライン
            if st.checkbox("アクションライン詳細設定"):
                # 色
                action_line_color = st.selectbox(
                    "アクションライン色", [action_line_color, "tab:blue", "red"]
                )
                # 線タイプ
                action_line_style = st.selectbox(
                    "アクションラインタイプ", [action_line_style, "-", "-.", ":"]
                )
                # 線幅
                action_line_width = st.slider(
                    "アクションライン幅", 0.0, 3.0, action_line_width, step=0.5
                )
            if st.checkbox("プロット線高さ設定"):
                # 線高さ
                line_height_k = st.slider("プロット線高さ", 0.0, 5.0, line_height_k, step=0.1)
        # 共通設定の保存と導入
        if st.checkbox("共通設定の保存"):
            init_settings = {
                # plot area settings
                "plot_area_setting": plot_area_setting,
                "plot_area_individual_setting": plot_area_individual_setting,
                "plot_area_max_x": plot_area_max_x,
                "plot_area_min_x": plot_area_min_x,
                # axis label settings
                "axis_label_setting": axis_label_setting,
                "axis_label_font": axis_label_font,
                # graph size settings
                "graph_size_width": graph_size_width,
                "graph_size_height": graph_size_height,
                "graph_size_dpi": graph_size_dpi,
                # hist graph settings
                "hist_graph_color": hist_graph_color,
                "hist_graph_bins": hist_graph_bins,
                # spec limit settings
                "spec_limit_setting": spec_limit_setting,
                "spec_limit_type_setting": spec_limit_type_setting,
                "spec_limit_color": spec_limit_color,
                "spec_limit_style": spec_limit_style,
                "spec_limit_width": spec_limit_width,
                "spec_limit_upper": spec_limit_upper,
                "spec_limit_lower": spec_limit_lower,
                "spec_limit_single": spec_limit_single,
                # normal distribution curve settings
                "curve_color": curve_color,
                "curve_style": curve_style,
                "curve_width": curve_width,
                # action line settings
                "action_line_color": action_line_color,
                "action_line_style": action_line_style,
                "action_line_width": action_line_width,
                # line height settings
                "line_height_k": line_height_k,
            }
            with open("settings.json", "w") as f:
                json.dump(init_settings, f, indent=4)
            path = "C:/Users/chenx/OneDrive/git/cxcode/"
            st.success("設定を" + path + "settings.jsonに保存しました")

        # 工程能力線図表示
        display_type = st.sidebar.radio("工程能力線図表示種類を選択", ("工程能力線図（項目別）", "工程能力線図（全項目）"))

        # 工程能力線図プロット関数
        def pci_fig(
            column_name,
            column_data,
            spec_limit_upper,
            spec_limit_lower,
            spec_limit_single,
            plot_area_max_x,
            plot_area_min_x,
        ):
            # プロット範囲設定
            if plot_area_individual_setting == "plot_area_individual_setting_yes":
                plot_area_max_x = st.number_input(
                    str([column_name]) + "プロット上限範囲(個別)を入力", value=plot_area_max_x
                )
                plot_area_min_x = st.number_input(
                    str([column_name]) + "プロット下限範囲(個別)を入力", value=plot_area_min_x
                )
            n = len(df[column_name])
            mu = df[column_name].mean()
            sig = column_data.std(ddof=1)
            # 両側規格の場合
            if spec_limit_type_setting == "dual_spec_limit":
                # 上下限規格個別設定
                if spec_limit_setting == "spec_limit_setting_yes":
                    spec_limit_upper = st.number_input(
                        str([column_name]) + "上限規格値(個別)を入力", value=spec_limit_upper
                    )
                    spec_limit_lower = st.number_input(
                        str([column_name]) + "下限規格値(個別)を入力", value=spec_limit_lower
                    )
                lal = mu - 9 * sig
                ual = mu + 9 * sig
                cpu = (spec_limit_upper - mu) / (3 * sig)
                cpl = (mu - spec_limit_lower) / (3 * sig)
                cpk = min(cpu, cpl)
            # 片側規格の場合
            else:
                # 片側規格個別設定
                if spec_limit_setting == "spec_limit_setting_yes":
                    spec_limit_single = st.number_input(
                        str([column_name]) + "片側規格値(個別)を入力", value=spec_limit_single
                    )
                if spec_limit_single > mu:
                    sal = mu + 9 * sig
                    cpk = (spec_limit_single - mu) / (3 * sig)
                elif spec_limit_single <= mu:
                    sal = mu - 9 * sig
                    cpk = (mu - spec_limit_single) / (3 * sig)
            nx = np.linspace(mu - 5.2 * sig, mu + 5.2 * sig, 1000)
            ny = sts.norm.pdf(nx, mu, sig)
            lh = max(ny) * line_height_k
            # 表示
            # 工程能力指数表示
            if spec_limit_type_setting == "dual_spec_limit":
                st.success(
                    str([column_name])
                    + f"の工程能力まとめ： n={n:.0f}, Ave={mu:.2f}, σ={sig:.2f}, Cpk={cpk:.2f}, UAL={ual:.2f}, LAL={lal:.2f}"
                )
            else:
                st.success(
                    str([column_name])
                    + f"の工程能力まとめ： n={n:.0f}, Ave={mu:.2f}, σ={sig:.2f}, Cpk={cpk:.2f}, AL={sal:.2f}"
                )
            # 図表示設定
            fig1 = plt.figure(
                dpi=graph_size_dpi, figsize=[graph_size_width, graph_size_height]
            )
            fig1.patch.set_alpha(0.0)
            sub_plot = fig1.add_subplot()
            sub_plot.patch.set_alpha(0.0)
            plt.rcParams["font.size"] = axis_label_font
            plt.tight_layout()
            if plot_area_setting == "plot_area_setting_yes":
                plt.xlim(plot_area_min_x, plot_area_max_x)
            if axis_label_setting == "axis_label_setting_no":
                plt.yticks([])
                plt.xticks([])
            plt.hist(
                df[column_name],
                bins=hist_graph_bins,
                color=hist_graph_color,
                rwidth=0.8,
                density=True,
                alpha=0.75,
            )
            plt.plot(
                nx, ny, color=curve_color, linewidth=curve_width, linestyle=curve_style
            )
            if spec_limit_type_setting == "dual_spec_limit":
                plt.vlines(
                    spec_limit_upper,
                    0,
                    lh,
                    color=spec_limit_color,
                    linewidth=spec_limit_width,
                    linestyle=spec_limit_style,
                )
                plt.vlines(
                    spec_limit_lower,
                    0,
                    lh,
                    color=spec_limit_color,
                    linewidth=spec_limit_width,
                    linestyle=spec_limit_style,
                )
                plt.vlines(
                    ual,
                    0,
                    lh,
                    color=action_line_color,
                    linewidth=action_line_width,
                    linestyle=action_line_style,
                )
                plt.vlines(
                    lal,
                    0,
                    lh,
                    color=action_line_color,
                    linewidth=action_line_width,
                    linestyle=action_line_style,
                )
            else:
                plt.vlines(
                    spec_limit_single,
                    0,
                    lh,
                    color=spec_limit_color,
                    linewidth=spec_limit_width,
                    linestyle=spec_limit_style,
                )
                plt.vlines(
                    sal,
                    0,
                    lh,
                    color=action_line_color,
                    linewidth=action_line_width,
                    linestyle=action_line_style,
                )
            return st.pyplot()

        # 結果表示
        st.header("工程能力まとめ結果一覧")
        # 単一項目選択表示
        if display_type == "工程能力線図（項目別）":
            select_column_names = st.selectbox("工程能力線図の表示項目を選択", column_names)
            select_column_data = df[select_column_names]
            pci_fig(
                select_column_names,
                select_column_data,
                spec_limit_upper,
                spec_limit_lower,
                spec_limit_single,
                plot_area_max_x,
                plot_area_min_x,
            )
            plt.close()
        # 全項目表示
        if display_type == "工程能力線図（全項目）":
            for column_name in column_names:
                column_data = df[column_name]
                pci_fig(
                    column_name,
                    column_data,
                    spec_limit_upper,
                    spec_limit_lower,
                    spec_limit_single,
                    plot_area_max_x,
                    plot_area_min_x,
                )
                plt.close()

# データ可視化ツール
if select_tool == "データ可視化":
    # データインプット
    st.title("データ可視化 ツール")
    upload_data = st.sidebar.file_uploader("データアップロード", type=["csv"])
    if upload_data is not None:
        df = pd.read_csv(upload_data)

        if st.checkbox("元データの表示 "):
            st.dataframe(df)
        # データ可視化
        column_names = df.columns.tolist()
        st.subheader("表示結果一覧")
        graph_type = st.sidebar.radio("グラフタイプを選択", ("棒図", "線図", "散布図行列"))
        select_column_names = st.multiselect(
            "データを選択", column_names, default=column_names[0]
        )

        # Plot By Streamlit
        if graph_type == "棒図":
            barchart_data = df[select_column_names]
            st.bar_chart(barchart_data)

        elif graph_type == "線図":
            linechart_data = df[select_column_names]
            st.line_chart(linechart_data)

        elif graph_type == "散布図行列":
            pairplot_data = df[select_column_names]
            sns.set_style("whitegrid", {"font.sans-serif": ["Meiryo", "Arial"]})
            sns.pairplot(pairplot_data, kind="reg")
            st.pyplot()
            plt.close()
