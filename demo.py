import os
import hmac
from typing import Optional, Tuple, Dict, List
import numpy as np
import pandas as pd
import streamlit as st
import re
import unicodedata
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import date
from streamlit.components.v1 import html as st_html
from datetime import datetime, timedelta
import datetime as dt
from pathlib import Path


st.set_page_config(page_title="統合版", layout="wide")

# Streamlit Cloud対策: safe_segmented_control が環境によってSegmentation faultを起こすため、
# 同等の横並びradioに退避する薄いラッパー。
def safe_segmented_control(label, options, *, default=None, key=None, label_visibility="visible", **kwargs):
    opts = list(options)
    idx = opts.index(default) if default in opts else 0
    safe_label = label if str(label).strip() else "選択"
    safe_visibility = label_visibility
    if not str(label).strip():
        safe_visibility = "collapsed"
    return st.radio(
        safe_label,
        opts,
        index=idx,
        horizontal=True,
        key=key,
        label_visibility=safe_visibility,
    )
ANCHOR_YEAR = 2000

def require_password_gate():
    if st.session_state.get("authed", False):
        return

    st.title("ログイン")
    st.caption("共通パスワードを入力してください。")

    pw = st.text_input("パスワード", type="password", key="__pw")

    if st.button("ログイン", use_container_width=True):
        expected = st.secrets.get("APP_PASSWORD", "")
        if expected and hmac.compare_digest(str(pw), str(expected)):
            st.session_state["authed"] = True
            st.session_state.pop("__pw", None)
            st.success("ログインしました。")
            st.rerun()
        else:
            st.error("パスワードが違います。")

    st.stop()   # ← 重要：認証されるまで本体を表示しない

DEFAULT_BASE_DIR = "data"
base_dir = os.environ.get("APP_BASE_DIR", DEFAULT_BASE_DIR)
def pjoin(*parts: str) -> str:
    return os.path.normpath(os.path.join(*parts))

MATURITY_PATH = pjoin(base_dir, "maturity.csv")          
LARVAE_PATH   = pjoin(base_dir, "larvae.csv")            

COLLECTOR_NUMBER_PATH = pjoin(base_dir, "collector_number.csv")
TITLE_SIZE = 18
TEMP_MIN, TEMP_MAX = -2.0, 40.0

from typing import Optional

def resolve_dr_path(base_dir: str, filename: str) -> Optional[str]:
    parent = pjoin(base_dir, 'pred')
    try:
        parent_exists = os.path.exists(parent)
    except Exception:
        parent_exists = False
    if not parent_exists:
        return None

    fn = unicodedata.normalize('NFC', filename.strip())
    base, ext = os.path.splitext(fn)
    if ext == '':
        ext = '.csv'
    target_base = unicodedata.normalize('NFC', base).lower()

    candidate = pjoin(parent, base + ext)
    try:
        if os.path.exists(candidate):
            return candidate
    except Exception:
        pass

    try:
        files = os.listdir(parent)
    except Exception:
        files = []
    for f in files:
        nf = unicodedata.normalize('NFC', f)
        b, e = os.path.splitext(nf)
        if b.lower() == target_base and e.lower() == '.csv':
            return pjoin(parent, nf)
    return None

def list_dr_files_safe(base_dir: str) -> list:
    parent = pjoin(base_dir, 'pred')
    out = []
    try:
        if os.path.exists(parent):
            for f in os.listdir(parent):
                nf = unicodedata.normalize('NFC', f)
                if nf.lower().endswith('.csv'):
                    out.append(nf)
    except Exception:
        pass
    return sorted(out)

def anchored_md_series(s):
    s = pd.to_datetime(s, errors="coerce")
    return pd.to_datetime(s.dt.strftime(f"{ANCHOR_YEAR}-%m-%d"), errors="coerce").dt.date

def filter_by_areas(df, areas):
    if df is None or len(df) == 0:
        return df
    if areas and "Area" in df.columns:
        return df[df["Area"].astype(str).isin(areas)]
    return df

@st.cache_data(show_spinner=False)
def read_csv_path(path: str, try_encodings=("utf-8", "utf-8-sig", "cp932"), fp: str = ""):
    last_err = None
    for enc in try_encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            df.columns = [c.strip() for c in df.columns]
            return df
        except Exception as e:
            last_err = e
            continue
    st.error(f"CSV読み込みに失敗しました: {path}\n{last_err}")
    return None

def load_all_areas():
    areas = set()
    for path in [LARVAE_PATH]:
        df = read_csv_path(path)
        if df is not None and "Area" in df.columns:
            areas.update(df["Area"].dropna().astype(str).unique().tolist())
    return sorted(list(areas))

def safe_merge_asof_by_depth(
    left: pd.DataFrame,
    right: pd.DataFrame,
    tolerance: pd.Timedelta,
    right_value_cols: List[str],
    suffixes: Tuple[str, str] = ("_x", "_y"),
) -> pd.DataFrame:

    out_list = []
    common_depths = sorted(
        set(left["depth_m"].dropna().unique()).intersection(
            set(right["depth_m"].dropna().unique())
        )
    )
    for d in common_depths:
        l = left[left["depth_m"] == d].sort_values("datetime")
        r = right[right["depth_m"] == d].sort_values("datetime")[
            ["datetime", "depth_m"] + right_value_cols
        ]
        if l.empty or r.empty:
            continue
        merged = pd.merge_asof(
            l, r, on="datetime", by="depth_m",
            tolerance=tolerance, direction="nearest",
            suffixes=suffixes
        )
        out_list.append(merged)
    if not out_list:
        return pd.DataFrame(columns=list(left.columns) + right_value_cols)
    return pd.concat(out_list, ignore_index=True)

@st.cache_data(show_spinner=False)
def load_dr_single_file(base_dir: str, filename: str) -> pd.DataFrame:
    path = resolve_dr_path(base_dir, filename)
    if path is None or not os.path.exists(path):
        safe_name = filename if filename.endswith('.csv') else f'{filename}.csv'
        st.error(f'ファイルが見つかりません: {pjoin(base_dir, "pred", safe_name)}')
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception as e:
        st.error(f'読み込み失敗: {path} ({e})')
        return pd.DataFrame()
    df.columns = [c.strip() for c in df.columns]
    df['datetime'] = utc_to_jst_naive(df.get('Date'))
    df['depth_m'] = pd.to_numeric(df.get('Depth'), errors='coerce').round(0).astype('Int64')
    df = df.rename(columns={'Temp': 'pred_temp', 'Salinity': 'pred_sal'})
    df = df.dropna(subset=['datetime', 'depth_m']).copy()
    if ('U' in df.columns) and ('V' in df.columns):
        df['U'] = pd.to_numeric(df['U'], errors='coerce')
        df['V'] = pd.to_numeric(df['V'], errors='coerce')
        df['Speed'] = np.sqrt(np.square(df['U']) + np.square(df['V']))
        df['Direction_deg'] = (np.degrees(np.arctan2(df['U'], df['V'])) + 360.0) % 360.0
    df['date_day'] = df['datetime'].dt.date
    df['hour'] = df['datetime'].dt.hour
    return df
    
@st.cache_data(show_spinner=False)
def compute_depthwise_regression(
    base_dir: str,
    train_filename: str,
    tolerance_min: int = 30,
    start_dt: Optional[pd.Timestamp] = None,
    end_dt: Optional[pd.Timestamp] = None,
    min_pairs: int = 10,
) -> Tuple[Optional[Dict[int, Tuple[float, float]]], Optional[Dict[int, int]]]:

    dr_path  = pjoin(base_dir, "pred", train_filename)
    obs_path = pjoin(base_dir, "obs",  train_filename)
    if not (os.path.exists(dr_path) and os.path.exists(obs_path)):
        return None, None
    try:
        pred = pd.read_csv(dr_path)
        obs  = pd.read_csv(obs_path)
    except Exception as e:
        st.warning(f"補正用ファイルの読み込みに失敗しました: {e}")
        return None, None

    pred["datetime"] = utc_to_jst_naive(pred.get("Date"))
    obs["datetime"]  = jst_to_naive(obs.get("Date"))
    pred["depth_m"]  = pd.to_numeric(pred.get("Depth"), errors="coerce").round(0).astype("Int64")
    obs["depth_m"]   = pd.to_numeric(obs.get("Depth"),  errors="coerce").round(0).astype("Int64")
    pred = pred.dropna(subset=["datetime", "depth_m"]).copy()
    obs  = obs .dropna(subset=["datetime", "depth_m"]).copy()
    pred = pred.rename(columns={"Temp": "pred_temp"})
    obs  = obs .rename(columns={"Temp": "obs_temp"})
    if "pred_temp" not in pred.columns or "obs_temp" not in obs.columns:
        return None, None

    if start_dt is not None:
        pred = pred[pred["datetime"] >= start_dt]
        obs  = obs [obs ["datetime"] >= start_dt]
    if end_dt is not None:
        pred = pred[pred["datetime"] <= end_dt]
        obs  = obs [obs ["datetime"] <= end_dt]
    if pred.empty or obs.empty:
        return None, None

    tol = pd.Timedelta(minutes=int(tolerance_min))
    merged = safe_merge_asof_by_depth(
        pred.sort_values(["depth_m","datetime"]),
        obs .sort_values(["depth_m","datetime"]),
        tol, right_value_cols=["obs_temp"], suffixes=("", "")
    )
    pair = merged.dropna(subset=["pred_temp", "obs_temp", "depth_m"]).copy()
    if pair.empty:
        return None, None

    reg_depth, n_depth = {}, {}
    for d, g in pair.groupby("depth_m"):
        X = g["pred_temp"].astype(float).values
        y = g["obs_temp" ].astype(float).values
        mask = np.isfinite(X) & np.isfinite(y)
        X, y = X[mask], y[mask]
        n_depth[int(d)] = int(len(X))
        if len(X) >= min_pairs:
            A = np.vstack([X, np.ones_like(X)]).T
            beta, alpha = np.linalg.lstsq(A, y, rcond=None)[0]
            reg_depth[int(d)] = (float(alpha), float(beta))
    return (reg_depth if reg_depth else None, n_depth if n_depth else None)

def preprocess_gsi(df_gsi: pd.DataFrame) -> pd.DataFrame:
    df = df_gsi.copy()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    df["GSI"] = (
        df["GSI"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .replace("", np.nan)
        .astype(float)
    )

    df["Area"] = df["Area"].astype(str).str.strip()

    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].astype(str).str.lower().str.strip()
    else:
        df["Sex"] = "unknown"

    return df.dropna(subset=["Date", "Area"])

def compute_map_gsi(
    df_gsi: pd.DataFrame,
    area: str,
    week_start: pd.Timestamp,
    week_end: pd.Timestamp
) -> float:
    g = df_gsi[
        (df_gsi["Area"] == area) &
        (df_gsi["Date"] >= week_start) &
        (df_gsi["Date"] <= week_end)
    ]

    valid = g["GSI"].dropna()

    if len(valid) >= 1:
        return float(valid.mean())
    else:
        return np.nan

def compute_comment_gsi(
    df_gsi: pd.DataFrame,
    area: str,
    sex: str,
    week_start: pd.Timestamp,
    week_end: pd.Timestamp
) -> float:
    g = df_gsi[
        (df_gsi["Area"] == area) &
        (df_gsi["Sex"] == sex.lower()) &
        (df_gsi["Date"] >= week_start) &
        (df_gsi["Date"] <= week_end)
    ]

    valid = g["GSI"].dropna()

    if len(valid) >= 1:
        return float(valid.mean())
    else:
        return np.nan

# 水温
def render_water_with_optional_gsi_overlay(selected_areas_for_gsi: List[str]):
    # --- ファイル選択・存在チェック ---
    parent_folder_dr = pjoin(base_dir, "pred")
    if not os.path.exists(parent_folder_dr):
        st.error(f"フォルダが見つかりません: {parent_folder_dr}")
        st.stop()
    dr_files = list_dr_files_safe(base_dir)
    if not dr_files:
        st.warning("pred に CSV がありません")
        st.stop()

    # ユーティリティ
    def parse_mmdd(s: str) -> dt.date:
        try:
            m, d = dt.datetime.strptime(s.strip(), "%m/%d").month, dt.datetime.strptime(s.strip(), "%m/%d").day
            return dt.date(ANCHOR_YEAR, m, d)
        except Exception:
            return None

    def to_anchor_ts(ts: pd.Series) -> pd.Series:
        d = pd.to_datetime(ts, errors="coerce")
        return pd.to_datetime(d.dt.strftime(f"{ANCHOR_YEAR}-%m-%d %H:%M:%S"))

    def mmdd_mask(series_dt: pd.Series, start_anchor: pd.Timestamp, end_anchor: pd.Timestamp) -> pd.Series:
        anchored = pd.to_datetime(series_dt.dt.strftime(f"{ANCHOR_YEAR}-%m-%d %H:%M:%S"))
        if start_anchor <= end_anchor:
            return (anchored >= start_anchor) & (anchored <= end_anchor)
        else:
            # wrap（例：12/15〜01/15）は「>= start」or「<= end」
            return (anchored >= start_anchor) | (anchored <= end_anchor)

    def anchored_day_span(start_anchor: pd.Timestamp, end_anchor: pd.Timestamp) -> int:
        y_start = pd.Timestamp(f"{ANCHOR_YEAR}-01-01")
        y_end = pd.Timestamp(f"{ANCHOR_YEAR}-12-31")
        if start_anchor <= end_anchor:
            return (end_anchor - start_anchor).days + 1
        else:
            return (y_end - start_anchor).days + 1 + (end_anchor - y_start).days + 1

# ラーバ
from typing import List
from datetime import date
import streamlit as st
import re
import pandas as pd
import numpy as np

def _larvae_render_horizontal_with_year_column(
    q: pd.DataFrame,
    size_ints: list,
    band_labels: list,
    band_to_category,
    category_colors: dict,
    max_days: int,
    x_max: float
):

    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    def mmdd_to_md(mdstr: str) -> str:
        m, d = mdstr.split('-')
        return f"{int(m)}/{int(d)}"

    if q.empty:
        st.info("選択条件に該当するデータがありません。")
        return

    days_all = sorted(
        q["MMDD"].astype(str).unique(),
        key=lambda s: pd.to_datetime(f"2000-{s}").dayofyear
    )
    if len(days_all) > max_days:
        days_all = days_all[:max_days]

    years_to_show = sorted(q["Year"].unique().tolist(), reverse=True)
    if not years_to_show:
        st.info("選択年の期間内データがありません。")
        return

    first_bin = size_ints[0] if size_ints else 0
    def bin_low(s: int) -> int:
        return first_bin + ((s - first_bin) // 20) * 20

    def calc_vals(g: pd.DataFrame):
        bins_sum = {bl: 0.0 for bl in {bin_low(si) for si in size_ints}}
        if not g.empty:
            for si in size_ints:
                col = str(si)
                if col in g.columns:
                    bins_sum[bin_low(si)] += g[col].sum()
        labels = [f"{bl}-{bl+20}" for bl in sorted(bins_sum.keys())]
        vals   = [bins_sum[bl]     for bl in sorted(bins_sum.keys())]
        return labels, vals

    n_days = len(days_all)
    titles = []
    for _ in years_to_show:
        titles += [""] + [mmdd_to_md(md) for md in days_all]

    fig = make_subplots(
        rows=len(years_to_show), cols=n_days + 1,
        shared_yaxes=False, shared_xaxes=True,
        horizontal_spacing=0.01, vertical_spacing=0.08,
        subplot_titles=titles,
        column_widths=[0.02] + [(1.0 - 0.02) / max(1, n_days)] * n_days
    )

    for r, _ in enumerate(years_to_show, start=1):
        fig.update_xaxes(visible=False, row=r, col=1)
        fig.update_yaxes(visible=False, row=r, col=1)

    import plotly.graph_objects as go
    for r, yr in enumerate(years_to_show, start=1):
        for idx, md in enumerate(days_all, start=2):
            dyear = q[q["Year"] == yr]
            gmd   = dyear[dyear["MMDD"].astype(str) == md]
            labels, vals = calc_vals(gmd)
            colors_per_bar = [category_colors.get(band_to_category(lbl), "#cccccc") for lbl in labels]

            fig.add_trace(go.Bar(
                x=vals, y=labels, orientation="h",
                marker=dict(color=colors_per_bar, line=dict(color="#000", width=1)),
                showlegend=False, opacity=0.6,
                hovertemplate=(f"年: {yr}<br>日: {mmdd_to_md(md)}<br>帯: %{{y}}<br>合計: %{{x:.2f}}")
            ), row=r, col=idx)

            fig.update_yaxes(
                categoryorder="array", categoryarray=band_labels, automargin=True,
                showticklabels=(idx == 2),
                ticks=("outside" if idx == 2 else ""),
                row=r, col=idx
            )
            fig.update_xaxes(range=[0, x_max], row=r, col=idx)

        R = max(1, len(years_to_show))
        y_paper_mid = 1 - (r - 0.5) / R
        fig.add_annotation(
            text=f"{yr}年",
            xref="paper", yref="paper",
            x=0, y=y_paper_mid,
            xanchor="right", yanchor="middle",
            xshift=-80,
            showarrow=False,
            align="center",
            font=dict(size=12, color="#222"),
            textangle=-90
        )

    fig.update_layout(
        xaxis_title="", yaxis_title="サイズ帯（μm）",
        plot_bgcolor="white", paper_bgcolor="white",
        height=max(260, 240 * len(years_to_show)),
        margin=dict(l=120, r=10, t=60, b=10),
        font=dict(size=13, color="#222"),
        legend=dict(orientation="h", y=-0.12)
    )
    st.plotly_chart(fig, use_container_width=True)


def render_larvae_mode(selected_areas: Optional[List[str]]):
    import plotly.graph_objects as go
    from datetime import date

    # 読み込み・前処理
    df = read_csv_path(LARVAE_PATH)
    if df is None:
        st.stop()

    df["Date"]   = pd.to_datetime(df["Date"], errors="coerce")
    df["Year"]   = df["Date"].dt.year
    df["MMDD"]   = df["Date"].dt.strftime("%m-%d")
    df["md_doy"] = pd.to_datetime("2000-" + df["MMDD"], format="%Y-%m-%d").dt.dayofyear
    if "Area" in df.columns:
        df["Area"] = df["Area"].astype(str)

    size_cols  = [c for c in df.columns if c.isdigit()]  # 例: "160", "180", ...
    size_ints  = sorted(int(c) for c in size_cols)
    others_col = next((c for c in df.columns if c.lower().startswith("others")), None)
    for c in size_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    if others_col:
        df[others_col] = pd.to_numeric(df[others_col], errors="coerce").fillna(0.0)

    areas_all = sorted(df["Area"].dropna().astype(str).unique().tolist()) if "Area" in df.columns else []
    default_areas = (selected_areas or [])

    sel_areas_main = st.multiselect(
        "エリア選択（複数可）",
        options=areas_all,
        default=default_areas,
        key='larv_areas_main'
    )

    sel_areas_main = sel_areas_main or []

    if not sel_areas_main:
        st.info("エリアを選択してください。")
        return

    years_all = sorted(df["Year"].dropna().unique().tolist())
    latest = years_all[-1] if years_all else None

    c1, c2, c3 = st.columns([1.1, 2.0, 1.0])
    with c1:
        years_sel = st.multiselect("表示年", years_all, default=[latest] if latest else [], key='larv_years')

    min_md = date(ANCHOR_YEAR, 3, 1)
    max_md = date(ANCHOR_YEAR, 7, 31)
    def_start = date(ANCHOR_YEAR, 3, 1)
    def_end = date(ANCHOR_YEAR, 4, 30)

    with c2:
        sel_md_start, sel_md_end = st.slider(
            "対象期間（MM-DD）",
            min_value=min_md,
            max_value=max_md,
            value=(def_start, def_end),
            format="MM-DD",
            key='larv_period'
        )

    with c3:
        try:
            mode_b = safe_segmented_control('', options=['日別推移','期間内比率'], default='日別推移', key='larv_mode', label_visibility='collapsed')
        except Exception:
            mode_b = st.radio('', ['日別推移','期間内比率'], index=0, horizontal=True, key='larv_mode_radio', label_visibility='collapsed')

    max_days = 5

    def to_doy(d: date) -> int:
        return pd.Timestamp(d).day_of_year

    s_doy = to_doy(sel_md_start)
    e_doy = to_doy(sel_md_end)

    def in_window(md_doy: int, s: int, e: int) -> bool:
        return (s <= e and s <= md_doy <= e) or (s > e and (md_doy >= s or md_doy <= e))

    if not size_ints:
        st.info('サイズ列が見つかりません。')
        return

    first_bin = size_ints[0]

    def bin_low(s: int) -> int:
        return first_bin + ((s - first_bin) // 20) * 20

    band_labels = sorted({f"{bin_low(si)}-{bin_low(si)+20}" for si in size_ints}, key=lambda t: int(t.split('-')[0]))

    category_colors = {
        '<200': '#1f77b4',
        '200-259': '#ff7f0e',
        '>=260': '#d62728'
    }

    def band_to_category(band_label: str) -> str:
        try:
            low = int(band_label.split('-')[0])
        except Exception:
            return '<200'
        if low < 200:
            return '<200'
        elif 200 <= low <= 259:
            return '200-259'
        else:
            return '>=260'

    q_days = df[df['Area'].isin(sel_areas_main)].copy()
    if years_sel:
        q_days = q_days[q_days['Year'].isin(years_sel)]
    q_days = q_days[q_days['md_doy'].apply(lambda d: in_window(int(d), s_doy, e_doy))]

    days_all = sorted(
        q_days['MMDD'].astype(str).unique().tolist(),
        key=lambda s: pd.to_datetime(f"2000-{s}").dayofyear
    )
    days_show = days_all[:max_days] if len(days_all) > max_days else days_all

    auto_max_global = 0.0
    for area in sel_areas_main:
        df_area = filter_by_areas(df, [area])
        q_test = df_area.copy()
        if years_sel:
            q_test = q_test[q_test['Year'].isin(years_sel)]
        q_test = q_test[q_test['md_doy'].apply(lambda d: in_window(int(d), s_doy, e_doy))]
        if q_test.empty:
            continue
        for md in days_show:
            gmd = q_test[q_test['MMDD'].astype(str) == md]
            if gmd.empty:
                continue
            bins_sum = {}
            for si in size_ints:
                b = bin_low(si)
                bins_sum[b] = bins_sum.get(b, 0.0) + gmd[str(si)].sum()
            local_max = max(bins_sum.values()) if bins_sum else 0.0
            auto_max_global = max(auto_max_global, float(local_max))

    x_max_global = float(auto_max_global) if auto_max_global > 0 else 1.0

    tables_to_show: list[tuple[str, pd.DataFrame]] = []

    for i, area in enumerate(sel_areas_main):

        df_area = filter_by_areas(df, [area])

        q_area = df_area.copy()
        if years_sel:
            q_area = q_area[q_area["Year"].isin(years_sel)]
        q_area = q_area[q_area["md_doy"].apply(lambda d: in_window(d, s_doy, e_doy))]

        if q_area.empty or not size_cols:
            st.info("選択条件に該当するデータがありません。")
            if i < len(sel_areas_main or []) - 1:
                st.markdown("---")  
            continue

        if mode_b == "期間内比率":
            rows = []
            for yr, g in q_area.groupby("Year"):
                total = g[size_cols].sum().sum()  # Othersは除外
                bins_sum = {}
                for si in size_ints:
                    b = bin_low(si)
                    bins_sum[b] = bins_sum.get(b, 0.0) + g[str(si)].sum()
                for b_low in sorted(bins_sum.keys()):
                    ratio = (bins_sum[b_low] / total * 100) if total else 0.0
                    rows.append({"Year": yr, "帯": f"{b_low}-{b_low+20}", "比率%": ratio})
            bars_df = pd.DataFrame(rows)
            if bars_df.empty:
                st.info("棒グラフ用データがありません。")
            else:
                bands = sorted(bars_df["帯"].unique(), key=lambda t: int(t.split("-")[0]))
                years_sorted = sorted(bars_df["Year"].unique())  

                def opacity_for_year(yr: int) -> float:
                    if len(years_sorted) == 1:
                        return 0.95
                    i = years_sorted.index(yr)            # 0..n-1
                    frac = (i + 1) / len(years_sorted)    # 0..1
                    return min(1.0, 0.30 + 0.75 * frac)   # 0.30〜1.00

                def hex_to_rgba(hex_color: str, alpha: float) -> str:
                    h = hex_color.lstrip("#")
                    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
                    return f"rgba({r},{g},{b},{alpha:.3f})"

                fig = go.Figure()
                for yr in years_sorted:
                    d = bars_df[bars_df["Year"] == yr].set_index("帯").reindex(bands).reset_index()
                    d["比率%"] = d["比率%"].fillna(0.0)
                    alpha = opacity_for_year(yr)
                    colors_per_bar = [
                        hex_to_rgba(category_colors[band_to_category(band)], alpha)
                        for band in d["帯"]
                    ]
                    fig.add_trace(go.Bar(
                        x=d["帯"], y=d["比率%"], name=str(yr),
                        marker=dict(color=colors_per_bar, line=dict(color="rgba(0,0,0,0.65)", width=1)),
                        opacity=0.95,
                        hovertemplate=f"年: {yr}<br>帯: %{{x}}<br>比率: %{{y:.1f}}%",
                        legendgroup=str(yr)
                    ))

                fig.update_layout(
                    barmode="group",
                    xaxis_title="サイズ帯（μm）",
                    yaxis_title="比率（%）",
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    height=330,
                    margin=dict(l=10, r=10, t=30, b=10),
                    font=dict(size=14, color="#222"),
                    legend=dict(orientation="h", y=-0.18)
                )
                fig.update_yaxes(gridcolor="rgba(0,0,0,0.06)")
                st.plotly_chart(fig, use_container_width=True)

        else:
            _larvae_render_horizontal_with_year_column(
                q=q_area,
                size_ints=size_ints,
                band_labels=band_labels,
                band_to_category=band_to_category,
                category_colors=category_colors,
                max_days=max_days,
                x_max=x_max_global
            )

        st.caption(
            f"期間: {sel_md_start.strftime('%m-%d')} 〜 {sel_md_end.strftime('%m-%d')} / "
            f"Area: {area} / 年: {', '.join(map(str, years_sel)) if years_sel else '全て'}"
        )

        summary_rows = []
        cols_lt200    = [str(s) for s in size_ints if s < 200]
        cols_200_259  = [str(s) for s in size_ints if 200 <= s <= 259]
        cols_ge260    = [str(s) for s in size_ints if s >= 260]

        for yr, g in q_area.groupby("Year"):
            total_ex_others = g[size_cols].sum().sum()
            sum_lt200   = g[cols_lt200   ].sum().sum() if cols_lt200   else 0.0
            sum_200_259 = g[cols_200_259 ].sum().sum() if cols_200_259 else 0.0
            sum_ge260   = g[cols_ge260   ].sum().sum() if cols_ge260   else 0.0
            summary_rows += [
                {"サイズ": "200μm未満",  "年": yr, "合計": sum_lt200,   "割合": (sum_lt200   / total_ex_others * 100) if total_ex_others else 0.0},
                {"サイズ": "200-259μm", "年": yr, "合計": sum_200_259, "割合": (sum_200_259 / total_ex_others * 100) if total_ex_others else 0.0},
                {"サイズ": "260μm以上",  "年": yr, "合計": sum_ge260,   "割合": (sum_ge260   / total_ex_others * 100) if total_ex_others else 0.0},
            ]

        priority = {'260μm以上': 0, '200-259μm': 1, '200μm未満': 2}
        summary_df = pd.DataFrame(summary_rows, columns=["サイズ", "年", "合計", "割合"])
        if not summary_df.empty:
            summary_df["__order"] = summary_df["サイズ"].map(priority)
            summary_df = summary_df.sort_values(['年','__order'], ascending=[False, True]).drop(columns='__order')
            tables_to_show.append((area, summary_df))

        if i < len(sel_areas_main or []) - 1:
            st.markdown("---")

    if tables_to_show:
        place_bg = {"200μm未満": "#e6f3ff", "200-259μm": "#fff3e0", "260μm以上": "#ffe6e6"}

        def color_by_size(row: pd.Series):
            bg = place_bg.get(row.get("サイズ"), "")
            return [f"background-color: {bg}" if bg else "" for _ in row]

        for area, summary_df in tables_to_show:
            with st.expander(f"Area: {area}", expanded=False):  
                styled = (
                    summary_df.style
                    .apply(color_by_size, axis=1)
                    .set_properties(**{"border-color": "#ddd"})
                    .format({"合計": "{:.1f}", "割合": "{:.1f}"})
                )
                st.dataframe(styled, use_container_width=True)
        if i < len(sel_areas_main or []) - 1:
            st.markdown("---")

# 経年比較
def render_yearly_compare_mode():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from os.path import join as pjoin
    import plotly.express as px

    LARVAE_PATH = pjoin(base_dir, "larvae.csv")
    COLLECTOR_NUMBER_PATH = pjoin(base_dir, "collector_number.csv")
    COLLECTOR_SIZE_PATH = pjoin(base_dir, "collector_size.csv")

    df_l = read_csv_path(LARVAE_PATH, fp=file_fingerprint(LARVAE_PATH))
    df_c = read_csv_path(COLLECTOR_NUMBER_PATH, fp=file_fingerprint(COLLECTOR_NUMBER_PATH))
    df_s = read_csv_path(COLLECTOR_SIZE_PATH, fp=file_fingerprint(COLLECTOR_SIZE_PATH))

    if df_c is None:
        st.error("データが見つかりません。")
        st.stop()

    def get_period_label(dt):
        if pd.isna(dt): return "不明"
        day = dt.day
        if day <= 10: p = "上旬"
        elif day <= 20: p = "中旬"
        else: p = "下旬"
        return f"{dt.month}月{p}"

    df_c = df_c.copy()
    for c in ["Drop_Date", "Monitoring_Date"]: 
        df_c[c] = pd.to_datetime(df_c[c], errors="coerce")
    
    df_c["Scallop"] = pd.to_numeric(df_c["Scallop"], errors="coerce")
    df_c = df_c.dropna(subset=["Drop_Date", "Monitoring_Date", "Scallop"])
    
    df_c["Year"] = df_c["Drop_Date"].dt.year.astype(str)
    df_c["Period_Label"] = df_c["Drop_Date"].apply(get_period_label)
    df_c["Elapsed_Days"] = (df_c["Monitoring_Date"] - df_c["Drop_Date"]).dt.days
    df_c["Area"] = df_c.get("Area", "Unknown").astype(str).str.strip()

    df_c_agg = df_c.groupby(["Area", "Year", "Period_Label", "Elapsed_Days"], as_index=False)[["Scallop"]].mean().round(1)

    df_s_agg = pd.DataFrame()
    if df_s is not None and not df_s.empty:
        df_s = df_s.copy()
        shell_col = next((c for c in df_s.columns if any(k in c.lower() for k in ["shell", "殻長"])), None)
        if shell_col:
            for c in ["Drop_Date", "Monitoring_Date"]: 
                df_s[c] = pd.to_datetime(df_s[c], errors="coerce")
            df_s["val"] = pd.to_numeric(df_s[shell_col], errors="coerce")
            df_s = df_s.dropna(subset=["Drop_Date", "Monitoring_Date", "val"])
            df_s["Year"] = df_s["Drop_Date"].dt.year.astype(str)
            df_s["Period_Label"] = df_s["Drop_Date"].apply(get_period_label)
            df_s["Elapsed_Days"] = (df_s["Monitoring_Date"] - df_s["Drop_Date"]).dt.days
            df_s["Area"] = df_s.get("Area", "Unknown").astype(str).str.strip()
            
            df_s_agg = df_s.groupby(["Area", "Year", "Period_Label", "Elapsed_Days"], as_index=False)[["val"]].agg(["mean", "min", "max"])
            df_s_agg.columns = ["_".join(col).strip("_") for col in df_s_agg.columns]
            df_s_agg = df_s_agg.rename(columns={"val_mean": "mean_s", "val_min": "min_s", "val_max": "max_s"}).round(2)

    all_years = sorted(df_c_agg["Year"].unique(), reverse=True)
    c1, c2 = st.columns([1, 2])
    area_sel = c1.selectbox("エリア選択", sorted(df_c_agg["Area"].unique()))
    display_years = c2.multiselect("表示年", all_years, default=all_years[:2])

    fig = make_subplots(rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.15)

    y_palette = px.colors.qualitative.D3
    symbols = ["circle", "diamond", "square", "triangle-up", "star"]
    period_order = [f"{m}月{p}" for m in range(3, 8) for p in ["上旬", "中旬", "下旬"]]

    for i, yr in enumerate(sorted(display_years, reverse=True)):
        yr_col = y_palette[i % len(y_palette)]
        yr_sym = symbols[i % len(symbols)]
        rgba_faint = yr_col.replace('rgb', 'rgba').replace(')', ', 0.25)') if 'rgb' in yr_col else yr_col
        
        df_yr_c = df_c_agg[(df_c_agg["Area"] == area_sel) & (df_c_agg["Year"] == yr)]
        active_periods = [p for p in period_order if p in df_yr_c["Period_Label"].unique()]
        
        for p_label in active_periods:
            group = df_yr_c[df_yr_c["Period_Label"] == p_label].sort_values("Elapsed_Days")
            legend_label = f"{yr[-2:]}年 {p_label}"

            fig.add_trace(go.Scatter(
                x=[0] + group["Elapsed_Days"].tolist(), y=[0] + group["Scallop"].tolist(),
                mode="lines+markers", name=legend_label, legendgroup=legend_label,
                line=dict(color=yr_col, width=1.5, dash=None if "上旬" in p_label else "dash" if "中旬" in p_label else "dot"),
                marker=dict(symbol=yr_sym, size=10),
                hovertemplate=f"<b>{legend_label}</b><br>経過: %{{x}}日<br>付着: %{{y:.1f}}個<extra></extra>"
            ), row=1, col=1)

            if not df_s_agg.empty:
                s_group = df_s_agg[(df_s_agg["Area"] == area_sel) & (df_s_agg["Year"] == yr) & (df_s_agg["Period_Label"] == p_label)].sort_values("Elapsed_Days")
                if not s_group.empty:
                    fig.add_trace(go.Scatter(
                        x=s_group["Elapsed_Days"], y=s_group["mean_s"],
                        mode="lines+markers", name=legend_label, legendgroup=legend_label, showlegend=False,
                        line=dict(color=rgba_faint, width=1, dash="dash"),
                        marker=dict(symbol=yr_sym, size=8, color=yr_col),
                        error_y=dict(type='data', symmetric=False, array=(s_group["max_s"] - s_group["mean_s"]),
                                     arrayminus=(s_group["mean_s"] - s_group["min_s"]), visible=True, thickness=1, width=2, color=rgba_faint),
                        hovertemplate=f"<b>{legend_label}</b><br>殻長: %{{y:.2f}}mm<extra></extra>"
                    ), row=2, col=1)

    fig.update_layout(
        height=850,
        margin=dict(t=50, b=60, r=20, l=50), 
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02, 
            xanchor="left",
            x=0,
            font=dict(size=11)
        )
    )
    
    fig.update_yaxes(title=dict(text="付着数 (平均個/袋)", standoff=2), row=1, col=1)
    fig.update_yaxes(title=dict(text="殻長 (mm)", standoff=2), range=[0, 5], row=2, col=1)

    fig.update_xaxes(showticklabels=True, title_text="経過日数 (日)", row=1, col=1)
    fig.update_xaxes(showticklabels=True, title_text="経過日数 (日)", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

# カレンダー部品
HEAD_LENGTH_RATIO = 0.55
HEAD_HALF_HEIGHT_RATIO = 0.35
SHAFT_WIDTH_PX = 4.0
OUTLIER_TH = 4.0          
OUTLIER_TH_OBS = 2.0      
PHYS_MIN, PHYS_MAX = -1.5, 35.0

# ガイダンス
BASE_DIR = base_dir
PRED_DIR = "pred"
OBS_DIR = "obs"
CORR_DIR = "corr"
CMEM_DIR = "cmem"
CMEM_THETAO_DIR = "thetao"
CMEM_CHL_DIR = "chl"

# 固定パラメータ
RECENT_DAYS = 7
OUTLIER_TH = 4.0          
OUTLIER_TH_OBS = 2.0      
OBS_MATCH_TOL_MIN = 60    
CORR_MATCH_TOL_MIN = 60   
TEMP_MIN, TEMP_MAX = -2.0, 40.0
PHYS_MIN, PHYS_MAX = -1.5, 35.0
HIGH_TEMP_TH = 22.0
RANGE_STABLE = 0.5
DELTA_THRESH = 0.3
DISPLAY_MODE = "arrow"

WEEK_WINDOW_FORWARD = True

def _pick_series_corr_then_pred(g: pd.DataFrame) -> Optional[pd.Series]:
    cand = None
    if "corr_temp" in g.columns:
        c = pd.to_numeric(g["corr_temp"], errors="coerce")
        if c.notna().sum() >= 1:
            cand = c
    if cand is None and "pred_temp" in g.columns:
        p = pd.to_numeric(g["pred_temp"], errors="coerce")
        if p.notna().sum() >= 1:
            cand = p
    return cand
def utc_to_jst_naive(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    dt = dt.dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)
    return dt
def jst_to_naive(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", utc=False)
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)
    return dt
def safe_merge_asof_by_depth_keep_left(
    left: pd.DataFrame,
    right: pd.DataFrame,
    tolerance: pd.Timedelta,
    right_value_cols: List[str],
    suffixes: Tuple[str, str] = ("_x", "_y"),
) -> pd.DataFrame:
    out_list: List[pd.DataFrame] = []
    left_depths = sorted(set(left["depth_m"].dropna().unique()))
    for d in left_depths:
        l = left[left["depth_m"] == d].sort_values("datetime")
        r = right[right["depth_m"] == d].sort_values("datetime")[["datetime", "depth_m"] + right_value_cols]
        if l.empty:
            continue
        if r.empty:
            pad = l.copy()
            for c in right_value_cols:
                pad[c] = np.nan
            out_list.append(pad)
        else:
            merged = pd.merge_asof(
                l, r, on="datetime", by="depth_m",
                tolerance=tolerance, direction="nearest", suffixes=suffixes
            )
            out_list.append(merged)
    if not out_list:
        out = left.copy()
        for c in right_value_cols:
            if c not in out.columns:
                out[c] = np.nan
        return out
    return pd.concat(out_list, ignore_index=True)
def _detect_column(df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
    cols = list(df.columns)
    for c in cols:
        if c.lower() in [k.lower() for k in keywords]:
            return c
    norm = {c: c.lower().replace("_", "") for c in cols}
    for c, n in norm.items():
        ok = all(k.lower().replace("_", "") in n for k in keywords)
        if ok:
            return c
    return None
def to_rgba(color: str, alpha: float = 0.18) -> str:
    if not isinstance(color, str) or not color:
        return f"rgba(0,150,0,{alpha})"
    c = color.strip().lower()
    if c.startswith("rgba(") and c.endswith(")"):
        try:
            nums = c[5:-1].split(",")
            r, g, b = [int(float(x)) for x in nums[:3]]
            return f"rgba({r},{g},{b},{alpha})"
        except Exception:
            return f"rgba(0,150,0,{alpha})"
    if c.startswith("rgb(") and c.endswith(")"):
        try:
            r, g, b = [int(float(x)) for x in c[4:-1].split(",")[:3]]
            return f"rgba({r},{g},{b},{alpha})"
        except Exception:
            return f"rgba(0,150,0,{alpha})"
    if c.startswith("#"):
        h = c.lstrip("#")
        try:
            if len(h) == 3:
                r = int(h[0]*2, 16); g = int(h[1]*2, 16); b = int(h[2]*2, 16)
            elif len(h) == 6:
                r = int(h[0:2], 16); g = int(h[2:4], 16); b = int(h[4:6], 16)
            else:
                return f"rgba(0,150,0,{alpha})"
            return f"rgba({r},{g},{b},{alpha})"
        except Exception:
            return f"rgba(0,150,0,{alpha})"
    return c

def file_fingerprint(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return "missing"
    try:
        st_ = p.stat()
        return f"mtime:{int(st_.st_mtime)}:size:{st_.st_size}"
    except Exception:
        return "exists"
def obs_fingerprint(base_dir: str, obs_dir: str, filename: str) -> str:
    path = os.path.normpath(os.path.join(base_dir, obs_dir, filename))
    return file_fingerprint(path)

@st.cache_data(show_spinner=False)
def load_pred(filename: str, fp: str = "") -> pd.DataFrame:
    path = pjoin(BASE_DIR, PRED_DIR, filename)
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception:
        return pd.DataFrame()
    df.columns = [c.strip() for c in df.columns]
    df["datetime"] = utc_to_jst_naive(df.get("Date"))
    df["depth_m"] = pd.to_numeric(df.get("Depth"), errors="coerce").round(0).astype("Int64")
    df = df.rename(columns={"Temp": "pred_temp"})
    if ("U" in df.columns) and ("V" in df.columns):
        df["U"] = pd.to_numeric(df["U"], errors="coerce")
        df["V"] = pd.to_numeric(df["V"], errors="coerce")
        df["Speed"] = np.sqrt(np.square(df["U"]) + np.square(df["V"]))
        df["Direction_deg"] = (np.degrees(np.arctan2(df["U"], df["V"])) + 360.0) % 360.0
    df = df.dropna(subset=["datetime", "depth_m"]).copy()
    df["date_day"] = df["datetime"].dt.date
    return df

@st.cache_data(show_spinner=False)
def load_corr_for(filename: str, fp: str = "") -> pd.DataFrame:
    name, ext = os.path.splitext(filename)
    corr_filename = f"{name}_corr{ext}"
    path = pjoin(BASE_DIR, CORR_DIR, corr_filename)
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception:
        return pd.DataFrame()
    df.columns = [c.strip() for c in df.columns]
    df["datetime"] = jst_to_naive(df.get("Date"))
    df["depth_m"] = pd.to_numeric(df.get("Depth"), errors="coerce").round(0).astype("Int64")
    corr_col = _detect_column(df, ["corr", "temp"]) or ("CorrTemp" if "CorrTemp" in df.columns else None)
    if corr_col is None:
        corr_col = "Temp" if "Temp" in df.columns else None
    if corr_col is None:
        return pd.DataFrame()
    low_col  = _detect_column(df, ["corr", "low"])  or ("CorrLow"  if "CorrLow"  in df.columns else None)
    high_col = _detect_column(df, ["corr", "high"]) or ("CorrHigh" if "CorrHigh" in df.columns else None)
    rename_map = {corr_col: "corr_temp"}
    if low_col:  rename_map[low_col]  = "corr_low"
    if high_col: rename_map[high_col] = "corr_high"
    df = df.rename(columns=rename_map)
    keep = ["datetime", "depth_m", "corr_temp"]
    if "corr_low" in df.columns:  keep.append("corr_low")
    if "corr_high" in df.columns: keep.append("corr_high")
    df = df[keep].dropna(subset=["datetime", "depth_m", "corr_temp"]).copy()
    df["date_day"] = df["datetime"].dt.date
    return df

@st.cache_data(show_spinner=False)
def load_obs_for(filename: str, fp: str = "") -> pd.DataFrame:
    path = pjoin(BASE_DIR, OBS_DIR, filename)
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception:
        return pd.DataFrame()
    df.columns = [c.strip() for c in df.columns]
    df["datetime"] = jst_to_naive(df.get("Date"))
    df["depth_m"] = pd.to_numeric(df.get("Depth"), errors="coerce").round(0).astype("Int64")
    df = df.rename(columns={"Temp": "obs_temp"})
    df = df.dropna(subset=["datetime", "depth_m"]).copy()
    df["date_day"] = df["datetime"].dt.date
    return df
def _extract_site_from_filename(fname: str, prefix: str) -> str:
    base = os.path.basename(fname)
    if base.lower().startswith(prefix.lower()) and base.lower().endswith('.csv'):
        return base[len(prefix):-4]
    return ""

def list_cmem_sites() -> List[str]:
    thetao_folder = pjoin(BASE_DIR, CMEM_DIR, CMEM_THETAO_DIR)
    chl_folder = pjoin(BASE_DIR, CMEM_DIR, CMEM_CHL_DIR)
    thetao_sites, chl_sites = set(), set()
    if os.path.exists(thetao_folder):
        for f in os.listdir(thetao_folder):
            s = _extract_site_from_filename(f, 'thetao_')
            if s:
                thetao_sites.add(s)
    if os.path.exists(chl_folder):
        for f in os.listdir(chl_folder):
            s = _extract_site_from_filename(f, 'chl_')
            if s:
                chl_sites.add(s)
    return sorted(thetao_sites.intersection(chl_sites))

def load_cmem_thetao(site: str, fp: str = "") -> pd.DataFrame:
    path = pjoin(BASE_DIR, CMEM_DIR, CMEM_THETAO_DIR, f"thetao_{site}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception:
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            return pd.DataFrame()
    df.columns = [c.strip() for c in df.columns]
    if 'flag' in df.columns:
        df = df[pd.to_numeric(df['flag'], errors='coerce') == 1]
    df['datetime'] = pd.to_datetime(df.get('Date'), errors='coerce')
    df['depth_m'] = pd.to_numeric(df.get('Depth'), errors='coerce').round(0).astype('Int64')
    val_col = 'Temp' if 'Temp' in df.columns else ('thetao' if 'thetao' in df.columns else None)
    if val_col is None:
        return pd.DataFrame()
    df = df.rename(columns={val_col: 'thetao'})
    df = df.dropna(subset=['datetime','depth_m','thetao']).copy()
    df['date_day'] = df['datetime'].dt.date
    return df

def load_cmem_chl(site: str, fp: str = "") -> pd.DataFrame:
    path = pjoin(BASE_DIR, CMEM_DIR, CMEM_CHL_DIR, f"chl_{site}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception:
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            return pd.DataFrame()
    df.columns = [c.strip() for c in df.columns]
    if 'flag' in df.columns:
        df = df[pd.to_numeric(df['flag'], errors='coerce') == 1]
    df['datetime'] = pd.to_datetime(df.get('Date'), errors='coerce')
    df['depth_m'] = pd.to_numeric(df.get('Depth'), errors='coerce').round(0).astype('Int64')
    val_col = 'chl' if 'chl' in df.columns else ('Temp' if 'Temp' in df.columns else None)
    if val_col is None:
        return pd.DataFrame()
    df = df.rename(columns={val_col: 'chl'})
    df = df.dropna(subset=['datetime','depth_m','chl']).copy()
    df['date_day'] = df['datetime'].dt.date
    return df
def add_corr(df_pred: pd.DataFrame, df_corr: pd.DataFrame) -> pd.DataFrame:
    if df_pred.empty or df_corr.empty:
        out = df_pred.copy()
        if "corr_temp" not in out.columns:
            out["corr_temp"] = np.nan
        if "corr_low" not in out.columns:
            out["corr_low"] = np.nan
        if "corr_high" not in out.columns:
            out["corr_high"] = np.nan
        return out

    tol = pd.Timedelta(minutes=CORR_MATCH_TOL_MIN)
    right_cols = ["corr_temp"]
    if "corr_low" in df_corr.columns: right_cols.append("corr_low")
    if "corr_high" in df_corr.columns: right_cols.append("corr_high")

    right = df_corr.sort_values(["depth_m", "datetime"])[["datetime", "depth_m"] + right_cols]
    left = df_pred.sort_values(["depth_m", "datetime"]).copy()

    merged = safe_merge_asof_by_depth_keep_left(
        left, right, tol, right_value_cols=right_cols, suffixes=("", "")
    )
    return merged

# ---- 余白圧縮CSS ----
def inject_compact_css():
    compact_css = """
    <style>
      [data-testid="stHeader"], header, .stAppHeader { display:none !important; height:0 !important; }
      footer, #MainMenu { display:none !important; height:0 !important; }
      .block-container { padding-top: 4px !important; padding-bottom: 4px !important; }
      .stMarkdown p { margin: 1px 0 !important; line-height: 1.12 !important; }

      /* 横並び（st.columns）の折り返しを抑制 */
      div[data-testid="stHorizontalBlock"] { gap: 6px !important; flex-wrap: nowrap !important; }
      div[data-testid="column"], div[data-testid="stColumn"] { min-width: 0 !important; }

      div[data-testid="stSelectbox"], div[data-testid="stRadio"], div[data-testid="stSegmentedControl"],
      div[data-testid="stMultiSelect"], div[data-testid="stDateInput"], div[data-testid="stSlider"],
      div[data-testid="stNumberInput"], div[data-testid="stCheckbox"] {
        margin-top: 0 !important;
        margin-bottom: 3px !important;
      }
      @media (max-width: 480px) {
        .block-container { padding-top: 1px !important; padding-bottom: 1px !important; }
        div[data-testid="stHorizontalBlock"] { gap: 4px !important; }
      }
    </style>
    """
    st_html(compact_css, height=0)
def get_arrow_svg(direction_deg, speed_mps):
    HLR = globals().get('HEAD_LENGTH_RATIO', 0.55)
    HHR = globals().get('HEAD_HALF_HEIGHT_RATIO', 0.35)
    SW  = globals().get('SHAFT_WIDTH_PX', 4.0)
    if pd.isna(speed_mps) or pd.isna(direction_deg):
        return ""
    css_angle = (direction_deg - 90) % 360
    def _style(s):
        if np.isnan(s): return 18, "#CCCCCC"
        speed_kt = s * 1.94384
        if speed_kt < 1.0: return 18, "#0000FF"
        elif speed_kt < 2.0: return 22, "#FFC107"
        else: return 26, "#FF0000"
    size, color = _style(speed_mps)
    head_length = size * HLR
    head_half_h = size * HHR
    line_end = size - head_length
    return f"""
<svg width="{size}" height="{size}" style="display:block;margin:0 auto;transform:rotate({css_angle}deg);">
  <line x1="4" y1="{size/2}" x2="{line_end}" y2="{size/2}"
        stroke="{color}" stroke-width="{globals().get('SHAFT_WIDTH_PX', 4.0)}" stroke-linecap="round"/>
  <polygon points="{line_end},{size/2 - head_half_h} {size},{size/2} {line_end},{size/2 + head_half_h}"
           fill="{color}"/>
</svg>
""".strip()
def get_color(temp: float, t_min: float = 0.0, t_max: float = 25.0) -> str:
    if pd.isna(temp): return "rgba(220,220,220,0.4)"
    ratio = (float(temp) - t_min) / (t_max - t_min)
    ratio = max(0, min(1, ratio))
    if ratio < 0.5:
        r = int(240 * ratio * 2); g = int(240 * ratio * 2); b = 240
    else:
        r = 240; g = int(240 * (1 - (ratio - 0.5) * 2)); b = int(240 * (1 - (ratio - 0.5) * 2))
    return f"rgba({r},{g},{b},0.4)"
def get_calendar_css(max_h_vh: int = 65) -> str:
    return f"""
    <style>
    .calendar-scroll-container {{
      overflow-x: auto; overflow-y: auto;
      max-height: {max_h_vh}vh; max-width: 100%;
      -webkit-overflow-scrolling: touch;
      border: 1px solid #e5e5e5; border-radius: 8px;
      isolation: isolate;
    }}
    .calendar-table {{
      border-collapse: separate; border-spacing: 0;
      width: max-content; min-width: 640px; font-size: 14px;
    }}
    .calendar-table th, .calendar-table td {{
      padding: 6px 10px;
      border-bottom: 1px solid #eee;
      text-align: center;
      white-space: nowrap;
    }}
    thead th {{
      position: sticky; top: 0;
      background: #fafafa; z-index: 2;
    }}
    .calendar-table tbody th.depth-cell,
    .calendar-table tbody td.depth-cell {{
      position: sticky; left: 0;
      background: #f7f7f7; z-index: 3;
      min-width: 56px; text-align: center; font-weight: 700 !important;
    }}
    thead th:first-child {{
      position: sticky; left: 0; top: 0;
      background: #f0f0f0; z-index: 4; min-width: 56px; text-align: center; font-weight: 700;
    }}
    .calendar-table .pred-small {{ font-size: 12px; color: #555; }}
    </style>
    """.strip()
def correction_effective(
    temp_pred: Optional[float],
    temp_corr: Optional[float],
    temp_obs: Optional[float] = None
) -> bool:
    if temp_pred is None or pd.isna(temp_pred): return False
    if temp_corr is None or pd.isna(temp_corr): return False
    if not (PHYS_MIN < float(temp_corr) < PHYS_MAX): return False
    if not (TEMP_MIN < float(temp_corr) < TEMP_MAX): return False
    if (temp_obs is not None) and (not pd.isna(temp_obs)):
        return abs(float(temp_corr) - float(temp_obs)) < OUTLIER_TH_OBS
    else:
        return abs(float(temp_corr) - float(temp_pred)) < OUTLIER_TH
def render_cell_html(
    temp_pred: Optional[float],
    speed_mps: Optional[float],
    dir_deg: Optional[float],
    temp_corr_raw: Optional[float],
    corr_on: bool,
    temp_obs: Optional[float] = None,
) -> str:
    corr_ok = corr_on and correction_effective(temp_pred, temp_corr_raw, temp_obs=temp_obs)
    bg_value = float(temp_corr_raw) if corr_ok else (float(temp_pred) if temp_pred is not None else np.nan)
    bg_color = get_color(bg_value) if not pd.isna(bg_value) else "rgba(220,220,220,0.6)"
    pred_label = f"{float(temp_pred):.1f}°C" if (temp_pred is not None and not pd.isna(temp_pred)) else "NaN"
    pred_html = f"<span class='pred-small'>{pred_label}</span>"

    speed_html, arrow_html = "", ""
    if (speed_mps is not None and not pd.isna(speed_mps)) and (dir_deg is not None and not pd.isna(dir_deg)):
        speed_kt = float(speed_mps) * 1.94384
        speed_html = f"<span style='font-size:12px;color:#444;'>{speed_kt:.1f} kt</span>"
        arrow_html = f"<span style='display:block;line-height:1;margin:0;padding:0;'>{get_arrow_svg(float(dir_deg), float(speed_mps))}</span>"

    corr_html = ""
    if corr_on and (temp_corr_raw is not None) and not pd.isna(temp_corr_raw) and corr_ok:
        corr_html = f"<span style='color:#D32F2F;font-weight:700;font-size:14px;'>{float(temp_corr_raw):.1f}°C</span>"

    content = (
        "<div style='display:flex;flex-direction:column;align-items:center;gap:2px;'>"
        + pred_html + speed_html + arrow_html + corr_html + "</div>"
    )
    return f"<td style='background:{bg_color}'>{content}</td>"
def build_weekly_table_html(df_period: pd.DataFrame, day_list: List[pd.Timestamp], depths: List[int], corr_on: bool) -> str:
    times = [d.strftime('%m/%d') for d in day_list]
    html = (
        '<div class="calendar-scroll-container"><table class="calendar-table">'
        "<thead><tr><th>水深</th>" + "".join([f"<th>{t}</th>" for t in times]) + "</tr></thead><tbody>"
    )
    for depth in depths:
        html += f"<tr><td class='depth-cell'>{depth}m</td>"
        for day in day_list:
            g = df_period[(df_period["date_day"] == day.date()) & (df_period["depth_m"] == depth)]
            if not g.empty:
                target_dt = pd.Timestamp(day.date()) + pd.Timedelta(hours=12)
                row = g.assign(_diff=(g["datetime"] - target_dt).abs()).sort_values("_diff").iloc[[0]]
                temp_ark = row
                temp_pred = float(temp_ark["pred_temp"].values[0]) if "pred_temp" in temp_ark.columns else np.nan
                speed_val = float(temp_ark["Speed"].values[0]) if "Speed" in temp_ark.columns else np.nan
                dir_val = float(temp_ark["Direction_deg"].values[0]) if "Direction_deg" in temp_ark.columns else np.nan
                temp_corr = float(temp_ark["corr_temp"].values[0]) if "corr_temp" in temp_ark.columns else None
                temp_obs = float(temp_ark["obs_temp"].values[0]) if ("obs_temp" in temp_ark.columns and not pd.isna(temp_ark["obs_temp"].values[0])) else None
                html += render_cell_html(temp_pred, speed_val, dir_val, temp_corr, corr_on, temp_obs=temp_obs)
            else:
                html += "<td>-</td>"
        html += "</tr>\n"
    html += "</tbody></table></div>"
    return html
def build_daily_table_html(df_day: pd.DataFrame, depths: List[int], corr_on: bool) -> str:
    hours_list = sorted(df_day["datetime"].dt.floor("h").unique())
    times_hr = [t.strftime('%H:%M') for t in hours_list]
    html = (
        '<div class="calendar-scroll-container"><table class="calendar-table">'
        "<thead><tr><th>水深</th>" + "".join([f"<th>{t}</th>" for t in times_hr]) + "</tr></thead><tbody>"
    )
    for depth in depths:
        html += f"<tr><td class='depth-cell'>{depth}m</td>"
        for t_obj in hours_list:
            row = df_day[(df_day["datetime"].dt.floor("h") == t_obj) & (df_day["depth_m"] == depth)]
            if not row.empty:
                temp_pred = float(row["pred_temp"].values[0]) if "pred_temp" in row.columns else np.nan
                speed_val = float(row["Speed"].values[0]) if "Speed" in row.columns else np.nan
                dir_val = float(row["Direction_deg"].values[0]) if "Direction_deg" in row.columns else np.nan
                temp_corr = float(row["corr_temp"].values[0]) if "corr_temp" in row.columns else None
                temp_obs = float(row["obs_temp"].values[0]) if ("obs_temp" in row.columns and not pd.isna(row["obs_temp"].values[0])) else None
                html += render_cell_html(temp_pred, speed_val, dir_val, temp_corr, corr_on, temp_obs=temp_obs)
            else:
                html += "<td>-</td>"
        html += "</tr>\n"
    html += "</tbody></table></div>"
    return html
def make_layer_groups(depths: List[int]) -> Dict[str, List[int]]:
    if not depths:
        return {"表層": [], "中層": [], "底層": []}
    d_sorted = sorted(depths); n = len(d_sorted)
    if n <= 3:
        top = d_sorted[:1]
        mid = d_sorted[1:2] if n >= 2 else []
        bot = d_sorted[2:] if n >= 3 else (d_sorted[-1:] if n >= 1 else [])
    elif n in (4, 5):
        top = d_sorted[:2]; mid = d_sorted[2:3]; bot = d_sorted[3:]
    else:
        top = d_sorted[:2]; bot = d_sorted[-2:]
        mid = [d for d in d_sorted if d not in top + bot]
        if len(mid) >= 3:
            c = len(mid) // 2
            mid = mid[c-1:c+1]
    return {"表層": top, "中層": mid, "底層": bot}
def summarize_weekly_for_depth(layer_name: str, target_depth: int, df_period: pd.DataFrame) -> Optional[str]:
    if df_period.empty or "depth_m" not in df_period.columns:
        return None
    g = df_period[df_period["depth_m"] == int(target_depth)].sort_values("datetime")
    if g.empty:
        return None

    series = _pick_series_corr_then_pred(g)
    if series is None:
        return None

    dfz = g.assign(val=pd.to_numeric(series, errors="coerce"))
    dfz = dfz[(dfz["val"] > PHYS_MIN) & (dfz["val"] < PHYS_MAX)].dropna(subset=["val"])
    if dfz.empty:
        return None
    if "date_day" not in dfz.columns:
        dfz["date_day"] = dfz["datetime"].dt.date

    daily = (
        dfz.groupby("date_day", as_index=False)["val"]
        .median()
        .sort_values("date_day")
    )
    temps = daily["val"]
    if temps.empty:
        return None

    rng_th = float(RANGE_STABLE)
    dlt_th = float(DELTA_THRESH)

    t_min, t_max = float(temps.min()), float(temps.max())
    if t_max >= HIGH_TEMP_TH:
        tag = f":red[高水温]（{t_min:.1f}℃～{t_max:.1f}℃）"
        return f"**{layer_name}**： {int(target_depth)}m{tag}"

    weekly_range = t_max - t_min
    if weekly_range < rng_th:
        t_start = float(temps.iloc[0])
        tag = f"安定（{t_start:.1f}℃）"
        return f"**{layer_name}**： {int(target_depth)}m{tag}"

    n = len(temps)
    idx_first = [i for i in [0, 1] if i < n]
    idx_last = [i for i in [6, 7] if i < n]
    first = temps.iloc[idx_first] if idx_first else temps.iloc[:max(1, n // 2)]
    last  = temps.iloc[idx_last]  if idx_last  else temps.iloc[max(1, n // 2):]
    delta = float(last.mean() - first.mean())

    first_mean = float(first.mean()); last_mean = float(last.mean())
    def payload_arrow() -> str: return f"{first_mean:.1f}℃→{last_mean:.1f}℃"
    def payload_range() -> str: return f"{t_min:.1f}–{t_max:.1f}℃"
    def payload() -> str: return payload_arrow() if DISPLAY_MODE == "arrow" else payload_range()

    if delta > +dlt_th:
        tag = f"上昇（{payload()}）"
    elif delta < -dlt_th:
        tag = f"下降（{payload()}）"
    else:
        t_start = float(temps.iloc[0]); t_end = float(temps.iloc[-1])
        end_diff = t_end - t_start
        if abs(end_diff) >= dlt_th:
            tag = f"{'上昇' if end_diff > 0 else '下降'}（{payload()}）"
        else:
            tag = f"安定（{payload()}）"
    return f"**{layer_name}**： {int(target_depth)}m{tag}"
def pick_shallow_mid_deep_min10_from_depths(depths: List[int]) -> List[int]:
    if not depths:
        return []
    xs = sorted(set(int(d) for d in depths))
    n = len(xs)
    if n <= 2:
        return xs
    low_idx = 0
    for i, d in enumerate(xs):
        if d >= 10:
            low_idx = i
            break
    high_idx = n - 1
    mid_idx = (low_idx + high_idx) // 2
    chosen = [xs[low_idx], xs[mid_idx], xs[high_idx]]
    return sorted(set(chosen))
def summarize_weekly_layer_temp(layer_name: str, layer_depths: List[int], df_period: pd.DataFrame) -> Optional[str]:
    if not layer_depths or df_period.empty or "depth_m" not in df_period.columns:
        return None
    valid_depths = set(pd.to_numeric(df_period["depth_m"], errors="coerce").dropna().astype(int))
    depths_in_data = sorted(int(d) for d in layer_depths if int(d) in valid_depths)
    if not depths_in_data:
        return None
    smd = pick_shallow_mid_deep_min10_from_depths(depths_in_data)
    if not smd:
        return None
    if layer_name == "表層":
        target_depth = smd[0]
    elif layer_name == "中層":
        target_depth = smd[min(1, len(smd)-1)]
    else:
        target_depth = smd[-1]
    return summarize_weekly_for_depth(layer_name, target_depth, df_period)
def dir_to_8pt_jp(deg: float) -> str:
    if pd.isna(deg): return ""
    dirs = ["北", "北東", "東", "南東", "南", "南西", "西", "北西"]
    idx = int(((float(deg) + 22.5) % 360) // 45)
    return dirs[idx]
def speed_class_from_mps(v_mps: Optional[float]) -> str:
    if v_mps is None or pd.isna(v_mps): return ""
    kt = float(v_mps) * 1.94384
    if kt >= 1.5: return "速"
    if kt >= 0.5: return "中"
    return "穏"
def summarize_daily_layer_flow(
    layer_name: str,
    layer_depths: List[int],
    df_day: pd.DataFrame,
    use_short_labels: bool = True,
    merge_same_segments: bool = False
) -> Optional[str]:
    if not layer_depths: return None
    DAY_BINS = [("朝", 4, 6), ("昼", 11, 13), ("夕", 16, 18)]
    order = {"朝": 0, "昼": 1, "夕": 2}
    rows: List[Tuple[str, str, str]] = []
    for label, h0, h1 in DAY_BINS:
        g = df_day[(df_day["depth_m"].isin(layer_depths)) & (df_day["datetime"].dt.hour.between(h0, h1))]
        if g.empty: continue
        U_mean = g["U"].mean() if "U" in g.columns else np.nan
        V_mean = g["V"].mean() if "V" in g.columns else np.nan
        if pd.notna(U_mean) and pd.notna(V_mean):
            speed_mean = float(np.sqrt(U_mean**2 + V_mean**2))
            dir_deg_mean = (np.degrees(np.arctan2(U_mean, V_mean)) + 360.0) % 360.0
        else:
            D = g["Direction_deg"].dropna() if "Direction_deg" in g.columns else pd.Series(dtype=float)
            if D.empty: continue
            rad = np.deg2rad(D.values)
            C = np.cos(rad).mean(); S = np.sin(rad).mean()
            dir_deg_mean = (np.degrees(np.arctan2(S, C)) + 360.0) % 360.0
            speed_mean = g["Speed"].mean() if "Speed" in g.columns else np.nan
        d_txt = dir_to_8pt_jp(dir_deg_mean) if pd.notna(dir_deg_mean) else ""
        v_cls = speed_class_from_mps(speed_mean) if pd.notna(speed_mean) else ""
        if use_short_labels and v_cls:
            v_map = {"穏やか": "穏", "中程度": "中", "速い": "速"}
            v_cls = v_map.get(v_cls, v_cls)
        if d_txt or v_cls:
            rows.append((label, d_txt, v_cls))
    if not rows: return None

    segments: List[str] = []
    if merge_same_segments:
        bucket: Dict[Tuple[str, str], List[str]] = {}
        for lbl, d, v in rows: bucket.setdefault((d, v), []).append(lbl)
        for (d, v), lbls in bucket.items():
            lbls_sorted = sorted(lbls, key=lambda x: order.get(x, 99))
            inner = "・".join([x for x in [d, v] if x])
            segments.append(f"{'・'.join(lbls_sorted)}（{inner}）")
    else:
        rows_sorted = sorted(rows, key=lambda r: order.get(r[0], 99))
        for lbl, d, v in rows_sorted:
            inner = "・".join([x for x in [d, v] if x])
            segments.append(f"{lbl}（{inner}）")
    return f"**{layer_name}**： " + "／".join(segments)

def render_water_mode():
    # --- ファイル選択（pred） ---
    pred_folder = pjoin(BASE_DIR, PRED_DIR)
    if not os.path.exists(pred_folder):
        st.error(f"フォルダが見つかりません: {pred_folder}")
        st.stop()
    pred_files = [f for f in os.listdir(pred_folder) if f.lower().endswith(".csv")]
    if not pred_files:
        st.warning("pred に CSV がありません")
        st.stop()
    selected_file = st.selectbox("", sorted(pred_files), key="water_selected_file", label_visibility="collapsed")

    # --- キャッシュキー用fingerprint ---
    pred_path = pjoin(BASE_DIR, PRED_DIR, selected_file)
    name, ext = os.path.splitext(selected_file)
    corr_path = pjoin(BASE_DIR, CORR_DIR, f"{name}_corr{ext}")
    obs_path  = pjoin(BASE_DIR, OBS_DIR, selected_file)
    fp_pred = file_fingerprint(pred_path)
    fp_corr = file_fingerprint(corr_path)
    fp_obs  = file_fingerprint(obs_path)

    df_pred = load_pred(selected_file, fp_pred)
    df_corr = load_corr_for(selected_file, fp_corr)
    df_obs  = load_obs_for(selected_file,  fp_obs)
    corr_available = not df_corr.empty

    if df_pred.empty:
        st.warning("予測データが読み込めませんでした")
        st.stop()

    latest_dt = df_pred["datetime"].max()
    available_days = sorted(df_pred["date_day"].unique()) if "date_day" in df_pred.columns else []
    if available_days:
        min_day = min(available_days); max_day = max(available_days)
    else:
        min_day = latest_dt.date(); max_day = latest_dt.date()

    try:
        graph_style = safe_segmented_control("", options=["コンター", "折れ線"], default="コンター", key="graph_style")
    except Exception:
        graph_style = st.radio("", ["コンター", "折れ線"], index=0, horizontal=True, key="graph_style_radio", label_visibility="collapsed")
    start_default = max(min_day, max_day - pd.Timedelta(days=10))
    start_day, end_day = st.slider(
        "", min_value=min_day, max_value=max_day, value=(start_default, max_day),
        key="graph_period_slider", label_visibility="collapsed"
    )
    title_suffix = f"（{start_day:%Y-%m-%d}〜{end_day:%Y-%m-%d}・時間別）"

    contour_agg = st.session_state.get("graph_contour_agg", "日平均")
    if graph_style == "コンター":
        try:
            contour_agg = safe_segmented_control("", options=["1時間", "日平均"], default="1時間", key="graph_contour_agg")
        except Exception:
            contour_agg = st.radio("", ["1時間", "日平均"], index=1, horizontal=True, key="graph_contour_agg_radio", label_visibility="collapsed")

    snap_freq = "1h" if contour_agg == "1時間" else "1D"

    df_period = df_pred[(df_pred["date_day"] >= start_day) & (df_pred["date_day"] <= end_day)].copy()
    df_period = df_period.sort_values("datetime")
    if "pred_temp" in df_period.columns and not df_period.empty:
        df_period = (
            df_period.groupby(["depth_m", "datetime"], as_index=False).agg({"pred_temp": "median"})
        )
    if not df_period.empty:
        df_period = (
            df_period.sort_values("datetime")
            .groupby("depth_m", group_keys=False)
            .apply(lambda g: (
                g.drop(columns=["depth_m"], errors="ignore").set_index("datetime")
                .resample("1h").median(numeric_only=True).interpolate(method="time", limit=2).reset_index()
                .assign(depth_m=int(g.name) if g.name is not None else pd.NA)
            ))
        )
    if "depth_m" in df_period.columns:
        df_period["depth_m"] = pd.to_numeric(df_period["depth_m"], errors="coerce").round(0).astype("Int64")

    merged_for_points = pd.DataFrame(columns=["datetime", "depth_m", "obs_temp"])
    if not df_obs.empty and not df_period.empty:
        df_obs_period = df_obs[(df_obs["date_day"] >= start_day) & (df_obs["date_day"] <= end_day)].copy()
        if not df_obs_period.empty:
            tol = pd.Timedelta(minutes=CORR_MATCH_TOL_MIN)
            left = df_period.sort_values(["depth_m","datetime"]).copy()
            right = df_obs_period.sort_values(["depth_m","datetime"])[["datetime","depth_m","obs_temp"]].copy()
            merged_for_points = safe_merge_asof_by_depth_keep_left(
                left=left, right=right, tolerance=tol, right_value_cols=["obs_temp"], suffixes=("","")
            )

    df_corr_period = pd.DataFrame()
    if corr_available:
        df_corr_period = df_corr[(df_corr["date_day"] >= start_day) & (df_corr["date_day"] <= end_day)].copy()
        if not df_corr_period.empty:
            use_cols = ["corr_temp"]
            if "corr_low" in df_corr_period.columns: use_cols.append("corr_low")
            if "corr_high" in df_corr_period.columns: use_cols.append("corr_high")
            df_corr_period = (
                df_corr_period.sort_values("datetime")
                .groupby("depth_m", group_keys=False)
                .apply(lambda g: (
                    g.drop(columns=["depth_m"], errors="ignore")
                    .set_index("datetime")[use_cols]
                    .resample("1h").median().dropna(how="all").reset_index()
                    .assign(depth_m=int(g.name) if g.name is not None else pd.NA)
                ))
            )

    if graph_style == "折れ線":
            fig = go.Figure()
            base_colors = px.colors.qualitative.Dark24
      
            depths_pred_all = sorted(set(df_period["depth_m"].dropna().astype(int).tolist())) if not df_period.empty else []
            depths_with_corr = set()
            if not df_corr_period.empty and "depth_m" in df_corr_period.columns:
                depths_with_corr = set(pd.to_numeric(df_corr_period["depth_m"], errors="coerce").dropna().astype(int).unique())
       
            depths_with_obs = set()
            if ("depth_m" in merged_for_points.columns) and ("obs_temp" in merged_for_points.columns):
                tmp_obs = merged_for_points.dropna(subset=["obs_temp"])
                if not tmp_obs.empty:
                    depths_with_obs = set(pd.to_numeric(tmp_obs["depth_m"], errors="coerce").dropna().astype(int).unique())
      

            both_corr_obs = sorted(depths_with_corr.intersection(depths_with_obs))
        
            def pick_shallow_mid_deep_min10(cands: List[int], k: int = 3) -> List[int]:
                if not cands:
                    return []
                xs = sorted(set(int(d) for d in cands))
                n = len(xs)
                if n <= 2:
                    return xs[:k]
                low_idx = 0
                for i, d in enumerate(xs):
                    if d >= 10:
                        low_idx = i
                        break
                high_idx = n - 1
                mid_idx = (low_idx + high_idx) // 2
                idxs = [low_idx, mid_idx, high_idx]
                chosen = [xs[i] for i in sorted(set(idxs))]
                if len(chosen) < k:
                    center = xs[mid_idx]
                    rest = [d for d in xs if d not in chosen]
                    rest_sorted = sorted(rest, key=lambda d: (abs(d - center), d))
                    chosen.extend(rest_sorted[:k - len(chosen)])

                return chosen[:k]

        
            if len(both_corr_obs) >= 3:
                default_depths = pick_shallow_mid_deep_min10(both_corr_obs, k=3)

            elif len(depths_with_corr) >= 3:
                default_depths = pick_shallow_mid_deep_min10(sorted(depths_with_corr), k=3)
            else:
                default_depths = pick_shallow_mid_deep_min10(depths_pred_all, k=3)
            if not default_depths:
                default_depths = depths_pred_all[: min(3, len(depths_pred_all))]
       
            selected_depths = st.multiselect(
                "", depths_pred_all, default=default_depths, key="graph_depths", label_visibility="collapsed"
            )
        
            def emphasize_color(hex_color: str) -> str:
                try:
                    rr = int(hex_color[1:3], 16); gg = int(hex_color[3:5], 16); bb = int(hex_color[5:7], 16)
                    rr = min(255, rr + 25); gg = min(255, gg + 25); bb = min(255, bb + 25)
                    return f"#{rr:02x}{gg:02x}{bb:02x}"
                except Exception:
                    return hex_color
        
            for i, d in enumerate(selected_depths):
                base_col = base_colors[i % len(base_colors)]
                corr_col = emphasize_color(base_col)
                lg = f"depth{int(d)}"
        
                g_pred = df_period[df_period["depth_m"] == d]
                g_corr = df_corr_period[df_corr_period["depth_m"] == d] if not df_corr_period.empty else pd.DataFrame()
                g_obs = merged_for_points[merged_for_points["depth_m"] == d] if ("depth_m" in merged_for_points.columns) else pd.DataFrame()
       
                if not g_corr.empty:
                    if ("corr_low" in g_corr.columns) and ("corr_high" in g_corr.columns):
                        fig.add_trace(go.Scatter(
                            x=g_corr["datetime"], y=g_corr["corr_low"].clip(lower=TEMP_MIN, upper=TEMP_MAX),
                            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip", name=f"{d}m 帯(下)"
                        ))
                        fig.add_trace(go.Scatter(
                            x=g_corr["datetime"], y=g_corr["corr_high"].clip(lower=TEMP_MIN, upper=TEMP_MAX),
                            mode="lines", line=dict(width=0),
                            fill='tonexty', fillcolor=to_rgba(corr_col, 0.18),
                            name=f"{d}m 信頼帯", legendgroup=lg, showlegend=False, hoverinfo="skip"
                        ))

                    y_corr = g_corr["corr_temp"].clip(lower=TEMP_MIN, upper=TEMP_MAX)
                    fig.add_trace(go.Scatter(
                        x=g_corr["datetime"], y=y_corr, mode="lines",
                        name=f"{d}m 補正", legendgroup=lg, showlegend=True,
                        line=dict(color=corr_col, width=3.0), opacity=1.0,
                        hovertemplate="%{x}<br>水深: " + f"{d}m" + "<br>補正水温: %{y:.2f} °C<extra></extra>"
                    ))

                    if not g_pred.empty:
                        y_pred = g_pred["pred_temp"].astype(float).clip(lower=TEMP_MIN, upper=TEMP_MAX)
                        fig.add_trace(go.Scatter(
                            x=g_pred["datetime"], y=y_pred, mode="lines",
                            name=f"{d}m 予測", legendgroup=lg, showlegend=False,
                            line=dict(color=base_col, width=1.2, dash="dot"), opacity=0.35,
                            hovertemplate="%{x}<br>水深: " + f"{d}m" + "<br>予測水温: %{y:.2f} °C<extra></extra>"
                        ))

                    if not g_obs.empty:
                        fig.add_trace(go.Scatter(
                            x=g_obs["datetime"], y=g_obs["obs_temp"], mode="markers",
                            name=f"{d}m 実測", legendgroup=lg, showlegend=True,
                            marker=dict(size=6, color=emphasize_color(base_col), line=dict(color="black", width=0.1)),
                            opacity=0.80,
                            hovertemplate="%{x}<br>水深: " + f"{d}m" + "<br>実測水温: %{y:.2f} °C<extra></extra>"
                        ))

                else:
                    if not g_pred.empty:
                        x = g_pred["datetime"]; y_pred = g_pred["pred_temp"].astype(float)
                        fig.add_trace(go.Scatter(
                            x=x, y=y_pred, mode="lines",
                            name=f"{d}m 予測", legendgroup=lg, showlegend=True,
                            line=dict(color=base_col, width=2.0), opacity=1.0,
                            hovertemplate="%{x}<br>水深: " + f"{d}m" + "<br>水温: %{y:.2f} °C"
                        ))

                    if not g_obs.empty:
                        fig.add_trace(go.Scatter(
                            x=g_obs["datetime"], y=g_obs["obs_temp"], mode="markers",
                            name=f"{d}m 実測", legendgroup=lg, showlegend=True,
                            marker=dict(size=4, color=emphasize_color(base_col), line=dict(color="black", width=0.1)),
                            opacity=0.40,
                            hovertemplate="%{x}<br>水深: " + f"{d}m" + "<br>実測水温: %{y:.2f} °C<extra></extra>"
                        ))
        
            fig.update_layout(
                title={"text": f"{selected_file} 水温{title_suffix}", "y": 0.98, "x": 0.01, "xanchor": "left", "font": {"size": 16}},
                margin=dict(l=10, r=10, t=50, b=10),
                height=550, template="plotly_white",
                legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="right", x=1, font=dict(size=12))
            )

            x_range = [pd.Timestamp(start_day), pd.Timestamp(end_day) + pd.Timedelta(days=1)]
            fig.update_xaxes(type="date", range=x_range, title_text="日時（JST）", tickfont=dict(size=11))
            fig.update_yaxes(title_text="水温 (℃)", tickfont=dict(size=11))
            st.plotly_chart(fig, use_container_width=True)
   
    else:
        use_corr_bg = (corr_available and (not df_corr_period.empty) and ("corr_temp" in df_corr_period.columns))
        if use_corr_bg:
            bg_name = "補正"
            bg_df = df_corr_period[["datetime","depth_m","corr_temp"]].rename(columns={"corr_temp":"bg_temp"}).copy()
        else:
            bg_name = "予測"
            bg_df = df_period[["datetime","depth_m","pred_temp"]].rename(columns={"pred_temp":"bg_temp"}).copy()
    
        if bg_df.empty:
            st.warning("コンター表示できるデータがありません")
            st.stop()
    
        bg_df["depth_m"] = pd.to_numeric(bg_df["depth_m"], errors="coerce").round(0).astype("Int64")
        bg_df["bg_temp"] = pd.to_numeric(bg_df["bg_temp"], errors="coerce").astype(float).clip(lower=TEMP_MIN, upper=TEMP_MAX)
        bg_df["datetime"] = pd.to_datetime(bg_df["datetime"], errors="coerce")
        bg_df = bg_df.dropna(subset=["datetime","depth_m","bg_temp"]).copy()
        bg_df["time_bin"] = bg_df["datetime"].dt.floor(snap_freq)
        bg_df = bg_df.groupby(["depth_m","time_bin"], as_index=False)["bg_temp"].mean()
    
        depths_all = sorted(set(bg_df["depth_m"].dropna().astype(int).tolist()))
        if not depths_all:
            st.warning("深度情報がありません")
            st.stop()
   
        t0 = pd.Timestamp(start_day)
        t1 = pd.Timestamp(end_day)
        if snap_freq == "1h":
            time_grid = pd.date_range(t0, t1 + pd.Timedelta(days=1) - pd.Timedelta(hours=1), freq="1h")
        else:
            time_grid = pd.date_range(t0, t1, freq="1D")
   
        pv = (
            bg_df.pivot_table(index="depth_m", columns="time_bin", values="bg_temp", aggfunc="mean")
            .reindex(index=depths_all, columns=time_grid)
        )
        z = pv.values
        if z.size == 0 or (not np.isfinite(np.nanmax(z))):
            st.warning("コンター表示できる値がありません")
            st.stop()
    
        zmin = float(np.nanmin(z)); zmax = float(np.nanmax(z))
        if not np.isfinite(zmin) or not np.isfinite(zmax) or zmin == zmax:
            zmin, zmax = TEMP_MIN, TEMP_MAX
    
   
    if graph_style != "折れ線":
        site_id = os.path.splitext(selected_file)[0] if selected_file else ""
        df_thetao = load_cmem_thetao(site_id, fp="") if site_id else pd.DataFrame()
        thetao_ok = (isinstance(df_thetao, pd.DataFrame) and (not df_thetao.empty))
        diff_candidates = []
        if corr_available and (not df_corr_period.empty) and ("corr_temp" in df_corr_period.columns):
            diff_candidates.append("実測 − 補正")
        diff_candidates.append("実測 − 予測")
        if thetao_ok:
            diff_candidates.append("実測 − CMEM(thetao)")
        default_diff = ("実測 − 補正" if "実測 − 補正" in diff_candidates else "実測 − 予測")
        diff_mode = st.session_state.get("graph_diff_mode", default_diff)
        if diff_mode not in diff_candidates:
            diff_mode = default_diff
            st.session_state["graph_diff_mode"] = diff_mode

        tab_wt, tab_cum, tab_thr = st.tabs(["水温", "積算水温", "22℃基準"])
        def _render_wt_contour(_contour_value: str):
            contour_value = _contour_value
            if graph_style == "折れ線":
                diff_freq = "1h"
            else:
                diff_freq = (
                    "1h"
                    if ("graph_contour_agg" in st.session_state
                        and st.session_state.get("graph_contour_agg") == "1時間")
                    else "1D"
                )
        
            from plotly.subplots import make_subplots
            start_ts = pd.Timestamp(start_day)
            end_ts   = pd.Timestamp(end_day) + pd.Timedelta(days=1)
        
            try:
                full_times = pd.date_range(start_ts, end_ts, freq=diff_freq, inclusive='left')
            except TypeError:
                step = pd.Timedelta(hours=1) if diff_freq == "1h" else pd.Timedelta(days=1)
                full_times = pd.date_range(start_ts, end_ts - step, freq=diff_freq)
        
            use_corr_bg = (corr_available and not df_corr_period.empty and "corr_temp" in df_corr_period.columns)
            if use_corr_bg:
                bg_name = "補正"
                bg = df_corr_period.rename(columns={"corr_temp":"bg_temp"})[["datetime","depth_m","bg_temp"]].copy()
            else:
                bg_name = "予測"
                bg = df_period.rename(columns={"pred_temp":"bg_temp"})[["datetime","depth_m","bg_temp"]].copy()
        
            bg["time_bin"] = bg["datetime"].dt.floor(diff_freq)
            depths_bg = sorted(bg["depth_m"].dropna().astype(int).unique())
            pv_bg = (
                bg.pivot_table(index="depth_m", columns="time_bin", values="bg_temp", aggfunc="mean")
                  .reindex(index=depths_bg, columns=full_times)
            )
        
            # --- 下段：差分（選択） ---
            site_id = os.path.splitext(selected_file)[0] if selected_file else ""
            df_thetao = load_cmem_thetao(site_id, fp="") if site_id else pd.DataFrame()
            thetao_ok = (isinstance(df_thetao, pd.DataFrame) and (not df_thetao.empty))
            diff_candidates = []
            if corr_available and (not df_corr_period.empty) and ("corr_temp" in df_corr_period.columns):
                diff_candidates.append("実測 − 補正")
            diff_candidates.append("実測 − 予測")
            if thetao_ok:
                diff_candidates.append("実測 − CMEM(thetao)")
            default_diff = ("実測 − 補正" if "実測 − 補正" in diff_candidates else "実測 − 予測")
            diff_mode = st.session_state.get("graph_diff_mode", default_diff)
            if diff_mode not in diff_candidates:
                diff_mode = default_diff
                st.session_state["graph_diff_mode"] = diff_mode
            
            def _bin_series(df_src: pd.DataFrame, value_col: str, agg: str, out_col: str) -> pd.DataFrame:
                if not (isinstance(df_src, pd.DataFrame) and (not df_src.empty)):
                    return pd.DataFrame(columns=["depth_m", "time_bin", out_col])
                tmp = df_src.copy()
                if "datetime" not in tmp.columns:
                    return pd.DataFrame(columns=["depth_m", "time_bin", out_col])
                tmp["datetime"] = pd.to_datetime(tmp["datetime"], errors="coerce")
                tmp["depth_m"] = pd.to_numeric(tmp.get("depth_m"), errors="coerce").round(0).astype("Int64")
                tmp[value_col] = pd.to_numeric(tmp.get(value_col), errors="coerce")
                tmp = tmp.dropna(subset=["datetime", "depth_m", value_col]).copy()
                tmp = tmp[(tmp["datetime"] >= start_ts) & (tmp["datetime"] < end_ts)].copy()
                if tmp.empty:
                    return pd.DataFrame(columns=["depth_m", "time_bin", out_col])
                tmp["time_bin"] = tmp["datetime"].dt.floor(diff_freq)
                if agg == "median":
                    out = tmp.groupby(["depth_m", "time_bin"], as_index=False)[value_col].median()
                else:
                    out = tmp.groupby(["depth_m", "time_bin"], as_index=False)[value_col].mean()
                out = out.rename(columns={value_col: out_col})
                return out
            
            obs_bin = _bin_series(df_obs, "obs_temp", ("median" if diff_freq == "1h" else "mean"), "obs")
            pred_bin = _bin_series(df_period, "pred_temp", "mean", "pred")
            corr_bin = _bin_series(df_corr_period, "corr_temp", "mean", "corr")
            thetao_bin = _bin_series(df_thetao, "thetao", "mean", "thetao") if thetao_ok else pd.DataFrame(columns=["depth_m","time_bin","thetao"])
            
            if diff_mode == "実測 − 補正":
                A, B = obs_bin, corr_bin; a_col, b_col = "obs", "corr"; a_lbl, b_lbl = "実測", "補正"
            elif diff_mode == "実測 − 予測":
                A, B = obs_bin, pred_bin; a_col, b_col = "obs", "pred"; a_lbl, b_lbl = "実測", "予測"
            elif diff_mode == "実測 − CMEM(thetao)":
                A, B = obs_bin, thetao_bin; a_col, b_col = "obs", "thetao"; a_lbl, b_lbl = "実測", "CMEM(thetao)"
            else:
                A, B = obs_bin, pred_bin; a_col, b_col = "obs", "pred"; a_lbl, b_lbl = "実測", "予測"
            diff_title = f"{a_lbl} − {b_lbl}"
            diff_title = f"{a_lbl} − {b_lbl}"
            
            _depths_for_grid = depths_bg
            _times_for_grid = list(full_times)
            grid = pd.DataFrame({
                "depth_m": np.repeat(_depths_for_grid, len(_times_for_grid)),
                "time_bin": np.tile(_times_for_grid, len(_depths_for_grid)),
            })
            mrg = grid.merge(A[["depth_m","time_bin",a_col]], on=["depth_m","time_bin"], how="left")
            mrg = mrg.merge(B[["depth_m","time_bin",b_col]], on=["depth_m","time_bin"], how="left")
            mrg["delta"] = mrg[a_col] - mrg[b_col]
        
            mrg2 = mrg.dropna(subset=["depth_m","time_bin"]).copy()
            mrg2["depth_m"] = mrg2["depth_m"].astype(int)
            depths_d = sorted(mrg2["depth_m"].unique())
            pv_d = (
                mrg2.pivot_table(index="depth_m", columns="time_bin", values="delta", aggfunc="mean")
                    .reindex(index=depths_d, columns=full_times)
            )
        
            z_bg = pv_bg.values
            z_d  = pv_d.values
        
            absmax = float(np.nanmax(np.abs(z_d))) if np.isfinite(np.nanmax(np.abs(z_d))) else 1.0
            absmax = max(absmax, 0.3)
        
            zmin_bg = float(np.nanmin(z_bg)) if np.isfinite(np.nanmin(z_bg)) else TEMP_MIN
            zmax_bg = float(np.nanmax(z_bg)) if np.isfinite(np.nanmax(z_bg)) else TEMP_MAX
            if (not np.isfinite(zmin_bg)) or (not np.isfinite(zmax_bg)) or (zmin_bg == zmax_bg):
                zmin_bg, zmax_bg = TEMP_MIN, TEMP_MAX
        
            cb1 = dict(title="℃", x=1.02, y=0.78, len=0.42, thickness=12)
            cb2 = dict(title="Δ℃", x=1.02, y=0.22, len=0.42, thickness=12)
        
            contour_agg_label = (
                contour_agg if "contour_agg" in locals() else ("1時間" if ("diff_freq" in locals() and diff_freq == "1h") else "日平均")
            )

            z_plot = z_bg
            zmin_plot, zmax_plot = zmin_bg, zmax_bg
            bg_title_name = "水温コンター"
            cb1_title = "℃"
            bg_colorscale = "Turbo"
            hover_bg = "日時=%{x|%Y-%m-%d %H:%M}<br>水深=%{y}m<br>T=%{z:.2f}℃<extra></extra>"

            if 'contour_value' in locals() and contour_value == "積算水温":
                if len(full_times) >= 2:
                    dt_days = (full_times[1] - full_times[0]).total_seconds() / 86400.0
                else:
                    dt_days = 1.0

                z_fill = np.where(np.isfinite(z_bg), z_bg, 0.0)
                z_plot = np.cumsum(z_fill * dt_days, axis=1)

                zmin_plot = 0.0
                zmax_plot = float(np.nanmax(z_plot)) if np.isfinite(np.nanmax(z_plot)) else 1.0
                bg_title_name = "積算水温コンター"
                cb1_title = "℃・day"
                hover_bg = "日時=%{x|%Y-%m-%d %H:%M}<br>水深=%{y}m<br>積算=%{z:.2f}℃・day<extra></extra>"
            
            elif 'contour_value' in locals() and contour_value == "22℃基準":
                z_plot = np.maximum(z_bg - HIGH_TEMP_TH, 0.0)
                zmin_plot = 0.0
                try:
                    zmax_plot = float(np.nanquantile(z_plot, 0.98))
                except Exception:
                    zmax_plot = float(np.nanmax(z_plot)) if np.isfinite(np.nanmax(z_plot)) else 0.5
                if (not np.isfinite(zmax_plot)) or (zmax_plot <= 0):
                    zmax_plot = 0.5
                zmax_plot = max(zmax_plot, 0.5)
                bg_title_name = "22℃基準温度コンター"
                cb1_title = "超過℃"
                bg_colorscale = "Reds"
                hover_bg = "日時=%{x|%Y-%m-%d %H:%M}<br>水深=%{y}m<br>基準超過=%{z:.2f}℃<extra></extra>"

            try:
                cb1['title'] = cb1_title
            except Exception:
                pass

            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                row_heights=[0.56, 0.44], vertical_spacing=0.14,
                subplot_titles=(f"{bg_title_name}（{bg_name}・{contour_agg_label}）", f"差分（{diff_title}・{contour_agg_label}）")
            )
            fig.layout.annotations[1].update(yshift=20)
        
            fig.add_trace(go.Heatmap(
                x=full_times, y=depths_bg, z=z_plot,
                colorscale=bg_colorscale, zmin=zmin_plot, zmax=zmax_plot,
                zsmooth="best",  
                colorbar=cb1,
                hovertemplate=hover_bg
            ), row=1, col=1)

            if contour_value == "水温":
                zmin_iso = float(np.floor(np.nanmin(z_bg))) if np.isfinite(np.nanmin(z_bg)) else zmin_bg
                zmax_iso = float(np.ceil(np.nanmax(z_bg))) if np.isfinite(np.nanmax(z_bg)) else zmax_bg
                
                fig.add_trace(go.Contour(
                    x=full_times, y=depths_bg, z=z_bg,
                    contours=dict(
                        start=zmin_iso,
                        end=zmax_iso,
                        size=1.0,          # 1℃固定
                        coloring="none"
                    ),
                    line=dict(
                        color="rgba(0,0,0,0.35)",
                        width=1
                    ),
                    showscale=False,
                    hoverinfo="skip",
                    name="等温線（1℃）"
                ), row=1, col=1)

            if contour_value == "積算水温":
                zmax_cum = float(np.nanmax(z_plot)) if np.isfinite(np.nanmax(z_plot)) else 0.0
                if zmax_cum > 0:
                    fig.add_trace(go.Contour(
                        x=full_times, y=depths_bg, z=z_plot,
                        contours=dict(start=0.0, end=zmax_cum, size=100.0, coloring="none"),
                        line=dict(color="rgba(0,0,0,0.35)", width=1),
                        showscale=False, hoverinfo="skip",
                        name="等積算線（100℃・day）"
                    ), row=1, col=1)
        
            fig.add_trace(go.Heatmap(
                x=full_times, y=depths_d, z=z_d,
                colorscale="RdBu_r", zmin=-absmax, zmax=absmax,
                zsmooth="best",  
                colorbar=cb2,
                hovertemplate="日時=%{x|%Y-%m-%d %H:%M}<br>水深=%{y}m<br>Δ=%{z:.2f}℃<extra></extra>"
            ), row=2, col=1)
        
            fig.update_xaxes(type="date", range=[start_ts, end_ts], showticklabels=True, title_text=None, tickfont=dict(size=10), row=1, col=1)
            fig.update_xaxes(type="date", range=[start_ts, end_ts], title_text="日時（JST）", tickfont=dict(size=10), row=2, col=1)
        
            fig.update_yaxes(title_text="水深 (m)", autorange="reversed", row=1, col=1)
            fig.update_yaxes(title_text="水深 (m)", autorange="reversed", row=2, col=1)

            if getattr(fig.layout, "annotations", None):
                for a in fig.layout.annotations:
                    a.update(x=0.01, xanchor="left", font=dict(size=13), yshift=(-8 if "差分" in (a.text or "") else 0))
        
            fig.update_layout(
                height=760, template="plotly_white",
                margin=dict(l=55, r=95, t=55, b=40)
            )
        
            st.plotly_chart(fig, use_container_width=True)


        with tab_wt:
            _render_wt_contour("水温")
        with tab_cum:
            _render_wt_contour("積算水温")
        with tab_thr:
            _render_wt_contour("22℃基準")

        if isinstance(diff_candidates, list) and (len(diff_candidates) > 0):
            try:
                _cur = st.session_state.get("graph_diff_mode", diff_mode)
                _idx = diff_candidates.index(_cur) if _cur in diff_candidates else 0
            except Exception:
                _idx = 0
            st.selectbox("", diff_candidates, index=_idx, key="graph_diff_mode", label_visibility="collapsed")
def render_cmem_mode():
    selected_file = None
    sites = list_cmem_sites()
    if not sites:
        st.warning("data/cmem/thetao と data/cmem/chl にサイトCSVがありません")
        st.stop()

    if selected_file:
        site_guess = os.path.splitext(selected_file)[0]
        if site_guess not in sites:
            st.warning(f"CMEMに site '{site_guess}' がありません（data/cmem/thetao, chl を確認）")
            st.stop()
        sel_site = site_guess
    else:
        sel_site = st.selectbox("", sites, key="cmem_site", label_visibility="collapsed")

    path_t = pjoin(BASE_DIR, CMEM_DIR, CMEM_THETAO_DIR, f"thetao_{sel_site}.csv")
    path_c = pjoin(BASE_DIR, CMEM_DIR, CMEM_CHL_DIR, f"chl_{sel_site}.csv")
    fp_t = file_fingerprint(path_t)
    fp_c = file_fingerprint(path_c)
    df_t = load_cmem_thetao(sel_site, fp_t)
    df_c = load_cmem_chl(sel_site, fp_c)

    if df_t.empty and df_c.empty:
        st.warning("CMEMデータが読み込めませんでした")
        st.stop()

    show_thetao = (not df_t.empty)
    show_chl = (not df_c.empty)

    depths_t = sorted([int(d) for d in df_t['depth_m'].dropna().astype(int).unique()]) if (show_thetao and (not df_t.empty)) else []
    depths_c = sorted([int(d) for d in df_c['depth_m'].dropna().astype(int).unique()]) if (show_chl and (not df_c.empty)) else []
    depths_all = sorted(set(depths_t + depths_c))
    if not depths_all:
        st.warning("深度情報がありません")
        st.stop()
    selected_depths = depths_all
    depths_sorted = depths_all

    dt_col = 'datetime'
    depths_int = [int(d) for d in selected_depths]
    df_t2_raw = df_t[df_t['depth_m'].isin(depths_int)].copy() if (show_thetao and not df_t.empty) else pd.DataFrame()
    df_c2_raw = df_c[df_c['depth_m'].isin(depths_int)].copy() if (show_chl and not df_c.empty) else pd.DataFrame()

    def _cmem_dt_minmax(df_a: pd.DataFrame, df_b: pd.DataFrame):
        s = pd.Series(dtype='datetime64[ns]')
        if not df_a.empty: s = pd.concat([s, pd.to_datetime(df_a[dt_col], errors='coerce')])
        if not df_b.empty: s = pd.concat([s, pd.to_datetime(df_b[dt_col], errors='coerce')])
        s = s.dropna()
        if len(s) == 0: return None, None
        return s.min(), s.max()

    try:
        cmem_period = safe_segmented_control("", options=["日別", "月別"], default="日別", key="cmem_period")
    except Exception:
        cmem_period = st.radio("", ["日別", "月別"], index=0, horizontal=True, key="cmem_period_radio", label_visibility="collapsed")

    tab_cmem_ts, tab_cmem_md = st.tabs(["時系列", "同月日比較"])

    def _render_cmem(cmem_view: str, df_t2: pd.DataFrame, df_c2: pd.DataFrame):
        def _available_years():
            s = pd.Series(dtype='datetime64[ns]')
            if not df_t2.empty:
                s = pd.concat([s, pd.to_datetime(df_t2[dt_col], errors='coerce')])
            if not df_c2.empty:
                s = pd.concat([s, pd.to_datetime(df_c2[dt_col], errors='coerce')])
            s = s.dropna()
            return sorted(s.dt.year.unique().tolist()) if len(s) else []

        years_all = _available_years()

        base_year = None
        comp_years = []
        if cmem_view == "同月日比較":
            if not years_all:
                st.warning("年情報がありません")
                st.stop()

            years_sorted = sorted([int(y) for y in years_all])
            base_year = st.selectbox("", years_sorted, index=len(years_sorted)-1, key="cmem_base_year", label_visibility="collapsed")
            cand = [y for y in years_sorted if y != int(base_year)]
            default_comp = cand[-2:] if len(cand) >= 2 else cand
            comp_years = st.multiselect("", cand, default=default_comp, key="cmem_comp_years", label_visibility="collapsed")
            if not comp_years:
                st.info("比較する年を選択してください")
                st.stop()

        base_colors = px.colors.qualitative.Dark24

        if cmem_view == "時系列":
            def _prep_grid(df_in: pd.DataFrame, value_col: str):
                if df_in.empty:
                    return None, None, None
                dfw = df_in.copy()
                dts = pd.to_datetime(dfw[dt_col], errors='coerce')
                if cmem_period == "月別":
                    dfw['t'] = dts.dt.to_period('M').dt.to_timestamp()
                else:
                    dfw['t'] = dts.dt.floor('D')
                dfw = dfw.dropna(subset=['depth_m','t', value_col]).copy()
                if dfw.empty:
                    return None, None, None
                dfw['depth_m'] = pd.to_numeric(dfw['depth_m'], errors='coerce').round(0).astype('Int64')
                dfw[value_col] = pd.to_numeric(dfw[value_col], errors='coerce')
                dfw = dfw.dropna(subset=['depth_m','t', value_col]).copy()
                dfw = dfw.groupby(['depth_m','t'], as_index=False)[value_col].mean()

                depths = sorted([int(d) for d in dfw['depth_m'].dropna().astype(int).unique().tolist()])
                times = sorted(pd.to_datetime(dfw['t']).dropna().unique().tolist())
                if (not depths) or (not times):
                    return None, None, None

                piv = dfw.pivot(index='depth_m', columns='t', values=value_col)
                piv = piv.reindex(index=depths, columns=times)
                z = piv.values
                return times, depths, z

            rows = 0
            titles = []
            grids = []
            if show_thetao and (not df_t2.empty):
                x, y, z = _prep_grid(df_t2, 'thetao')
                if x is not None:
                    rows += 1
                    titles.append("thetao（水温）")
                    grids.append(('thetao', x, y, z))
            if show_chl and (not df_c2.empty):
                df_c2 = df_c2.copy()
                df_c2['chl_log'] = np.log10(np.maximum(pd.to_numeric(df_c2['chl'], errors='coerce'), 0.01))
                x, y, z = _prep_grid(df_c2, 'chl_log')
                if x is not None:
                    rows += 1
                    titles.append("log10(chl)")
                    grids.append(('chl_log', x, y, z))

            if rows == 0:
                st.warning("CMEMデータが表示できません（空）")
                st.stop()

            fig = make_subplots(
                rows=rows, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.14,
                subplot_titles=titles
            )

            def _cbar_y(r: int) -> float:
                return 1.0 - (r - 0.5) / rows

            for r, (vname, x, y, z) in enumerate(grids, start=1):
                if vname == 'thetao':
                    colorscale = 'RdBu_r'
                    colorbar_title = '℃'
                else:
                    colorscale = 'Viridis'
                    colorbar_title = 'log10(chl)'

                tr = go.Contour(
                    x=x, y=y, z=z,
                    colorscale=colorscale,
                    contours=dict(coloring='heatmap', showlines=False),
                    ncontours=20,
                    colorbar=dict(title=colorbar_title, x=1.02, y=_cbar_y(r), yanchor='middle', len=0.75/rows),
                    hovertemplate="%{x}<br>Depth: %{y} m<br>Value: %{z:.4g}<extra></extra>"
                )
                fig.add_trace(tr, row=r, col=1)
                fig.update_yaxes(autorange='reversed', title_text="水深 (m)", row=r, col=1)

            # --- MY / ANFC 境界線（Source列がある日付/年月で切替が起きる場所に縦線） ---

            def _cmem_boundary_x(df_a: pd.DataFrame, df_b: pd.DataFrame) -> List[pd.Timestamp]:
                if (df_a is None) and (df_b is None):
                    return []
                frames = []
                for df0 in (df_a, df_b):
                    if df0 is None or df0.empty:
                        continue
                    if 'Source' not in df0.columns:
                        continue
                    d = df0[[dt_col, 'Source']].copy()
                    d[dt_col] = pd.to_datetime(d[dt_col], errors='coerce')
                    d = d.dropna(subset=[dt_col])
                    if d.empty:
                        continue
                    if cmem_period == '月別':
                        d['_t'] = d[dt_col].dt.to_period('M').dt.to_timestamp()
                    else:
                        d['_t'] = d[dt_col].dt.floor('D')
                    d['_anfc'] = d['Source'].astype(str).str.strip().str.upper().eq('ANFC')
                    frames.append(d[['_t', '_anfc']])
                if not frames:
                    return []
                u = pd.concat(frames, ignore_index=True)
                if u.empty:
                    return []
                g = u.groupby('_t', as_index=False)['_anfc'].any().sort_values('_t')
                if g.empty:
                    return []
                xs = g['_t'].tolist()
                flags = g['_anfc'].tolist()
                out = []
                for i in range(1, len(xs)):
                    if flags[i] != flags[i - 1]:
                        out.append(xs[i])
                return out

            boundary_x = _cmem_boundary_x(df_t2, df_c2)
            if boundary_x:
                for bx in boundary_x:
                    for rr in range(1, rows + 1):
                        try:
                            fig.add_vline(x=bx, row=rr, col=1, line_width=6, line_dash='solid', line_color='white', opacity=0.75)
                            fig.add_vline(x=bx, row=rr, col=1, line_width=2, line_dash='dot', line_color='black', opacity=0.85)
                        except Exception:
                            fig.add_shape(type='line', x0=bx, x1=bx, y0=0, y1=1, xref='x', yref='paper',
                                          line=dict(color='white', width=6, dash='solid'), opacity=0.75)
                            fig.add_shape(type='line', x0=bx, x1=bx, y0=0, y1=1, xref='x', yref='paper',
                                          line=dict(color='black', width=2, dash='dot'), opacity=0.85)

            title_suffix = "（時系列・月平均）" if cmem_period == "月別" else "（時系列・日別）"
            fig.update_layout(
                title={"text": f"CMEM {sel_site}{title_suffix}", "y": 0.98, "x": 0.01, "xanchor": "left", "font": {"size": 16}},
                margin=dict(l=10, r=120, t=70, b=10),
                height=260 + 280 * rows,
                template="plotly_white",
            )
            fig.update_xaxes(title_text="日付" if cmem_period == "日別" else "月", row=rows, col=1)
            st.plotly_chart(fig, use_container_width=True)

        else:
            base_y = int(base_year)
            comp_sorted = sorted([int(y) for y in comp_years if int(y) != base_y])
            if not comp_sorted:
                st.info("比較する年を選択してください")
                st.stop()

            if cmem_period == "月別":
                m_start, m_end = st.slider(
                    "", min_value=1, max_value=12, value=(1, 12),
                    key="cmem_month_window", label_visibility="collapsed"
                )
                months = list(range(int(m_start), int(m_end) + 1)) 
                month_order = {m: i for i, m in enumerate(months)}
                xname = "x_idx"
            else:
                m_start, m_end = st.slider(
                    "", min_value=1, max_value=12, value=(1, 12),
                    key="cmem_md_month_window", label_visibility="collapsed"
                )
                start_dt = pd.Timestamp(year=2000, month=int(m_start), day=1)
                end_dt   = (pd.Timestamp(year=2000, month=int(m_end), day=1) + pd.offsets.MonthEnd(0))
                md_list  = [d.strftime("%m-%d") for d in pd.date_range(start_dt, end_dt, freq="D")]
                md_order = {m: k for k, m in enumerate(md_list)}
                xname = "x_idx"

            def prep_thetao(df):
                if df.empty:
                    return pd.DataFrame()
                dts = pd.to_datetime(df[dt_col])
                df = df.assign(y=dts.dt.year)

                if cmem_period == "月別":
                    df = df.assign(
                        m=dts.dt.month,
                        x_idx=dts.dt.month.map(month_order),
                        x_label=dts.dt.month.apply(lambda v: f"{int(v)}月")
                    )
                    df = df[(df["m"] >= m_start) & (df["m"] <= m_end)]
                else:
                    df = df.assign(md=dts.dt.strftime("%m-%d"))
                    df = df[df["md"].isin(md_order)]
                    df = df.assign(x_idx=df["md"].map(md_order), x_label=df["md"])
                return df.groupby(
                    ["depth_m", "y", xname, "x_label"],
                    as_index=False
                )["thetao"].mean()

            def prep_chl(df):
                if df.empty:
                    return pd.DataFrame()
                dts = pd.to_datetime(df[dt_col])
                df = df.assign(
                    y=dts.dt.year,
                    chl_log=np.log10(np.maximum(pd.to_numeric(df["chl"], errors="coerce"), 0.01))
                )

                if cmem_period == "月別":
                    df = df.assign(
                        m=dts.dt.month,
                        x_idx=dts.dt.month.map(month_order),
                        x_label=dts.dt.month.apply(lambda v: f"{int(v)}月")
                    )
                    df = df[(df["m"] >= m_start) & (df["m"] <= m_end)]
                else:
                    df = df.assign(md=dts.dt.strftime("%m-%d"))
                    df = df[df["md"].isin(md_order)]
                    df = df.assign(x_idx=df["md"].map(md_order), x_label=df["md"])
                return df.groupby(
                    ["depth_m", "y", xname, "x_label"],
                    as_index=False
                )["chl_log"].mean()

            df_tg = prep_thetao(df_t2) if show_thetao else pd.DataFrame()
            df_cg = prep_chl(df_c2)    if show_chl else pd.DataFrame()

            def diff_base(df, valcol):
                if df.empty:
                    return pd.DataFrame()
                base = df[df["y"] == base_y][["depth_m", xname, "x_label", valcol]].rename(columns={valcol: "base"})
                cmp = df[df["y"].isin(comp_sorted)][["depth_m", xname, "x_label", valcol]]
                cmp_mean = cmp.groupby(["depth_m", xname, "x_label"], as_index=False)[valcol].mean()
                cmp_mean = cmp_mean.rename(columns={valcol: "cmp"})
                out = pd.merge(base, cmp_mean, on=["depth_m", xname, "x_label"])
                out["diff"] = out["base"] - out["cmp"]   
                return out

            df_tdiff = diff_base(df_tg, "thetao")
            df_cdiff = diff_base(df_cg, "chl_log")

            if cmem_period == "月別":
                x_grid = list(range(len(months)))
                x_labels = [f"{m}月" for m in months]
                tickvals = x_grid
                ticktext = x_labels

            else:
                x_grid = list(range(len(md_list)))
                x_labels = md_list[:]
                dt_tmp = pd.to_datetime([f"2000-{s}" for s in md_list], errors="coerce")
                tickvals, ticktext = [], []
                for i, d in enumerate(dt_tmp):
                    if pd.notna(d) and d.day == 1:
                        tickvals.append(i)
                        ticktext.append(d.strftime("%m/%d"))
                if len(tickvals) == 0:
                    step = max(1, len(md_list) // 12)
                    tickvals = list(range(0, len(md_list), step))
                    ticktext = [md_list[i] for i in tickvals]
        
            def _pivot_z(df_diff: pd.DataFrame) -> np.ndarray:
                """depths_sorted × x_grid の z 行列を作る（欠損は NaN）。"""
                if df_diff.empty:
                    return np.full((len(depths_sorted), len(x_grid)), np.nan)
                pv = (
                    df_diff.pivot_table(index="depth_m", columns=xname, values="diff", aggfunc="mean")
                    .reindex(index=depths_sorted, columns=x_grid)
                )
                return pv.values
        
            def _sym_zrange(z: np.ndarray, fallback: float = 1.0) -> float:
                """0中心の対称レンジ用 maxabs を返す。"""
                if z.size == 0:
                    return fallback
                m = np.nanmax(np.abs(z))
                if (not np.isfinite(m)) or (m <= 0):
                    return fallback
                return float(m)
        
            def _custom_xlabels_2d() -> np.ndarray:
                return np.tile(np.array(x_labels, dtype=object), (len(depths_sorted), 1))
        
            show_t = (show_thetao and (not df_tdiff.empty))
            show_c = (show_chl and (not df_cdiff.empty))
            if (not show_t) and (not show_c):
                st.warning("差分を描画できるデータがありません（基準年・比較年・期間・深度を確認）")
                st.stop()
        
            nrows = (1 if (show_t ^ show_c) else 2)
            titles = []
            if show_t:
                titles.append("thetao（水温）差：基準 − 比較（平均）")
            if show_c:
                titles.append("chl（log10）差：基準 − 比較（平均）")
        
            if nrows == 1:
                fig = make_subplots(rows=1, cols=1, shared_xaxes=True, subplot_titles=titles)
            else:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.14, subplot_titles=titles)
        
            row_i = 1
        
            if show_t:
                zt = _pivot_z(df_tdiff)
                maxabs_t = _sym_zrange(zt, fallback=0.5)
                fig.add_trace(
                    go.Contour(
                        x=x_grid, y=depths_sorted, z=zt,
                        colorscale="RdBu_r", zmin=-maxabs_t, zmax=maxabs_t,
                        contours=dict(coloring="heatmap"), connectgaps=False,
                        colorbar=dict(title="℃", x=1.02, y=(0.78 if nrows == 2 else 0.50), len=(0.42 if nrows == 2 else 0.85)),
                        customdata=_custom_xlabels_2d(),
                        hovertemplate="時点:%{customdata}<br>水深:%{y} m<br>差:%{z:.2f} ℃<extra></extra>"
                    ),
                    row=row_i, col=1
                )

                fig.update_yaxes(autorange="reversed", title_text="水深 (m)", row=row_i, col=1)
                row_i += 1
        
            if show_c:
                zc = _pivot_z(df_cdiff)
                maxabs_c = _sym_zrange(zc, fallback=0.3)
                fig.add_trace(
                    go.Contour(
                        x=x_grid, y=depths_sorted, z=zc,
                        colorscale="RdBu_r", zmin=-maxabs_c, zmax=maxabs_c,
                        contours=dict(coloring="heatmap"), connectgaps=False,
                        colorbar=dict(title="log10", x=1.02, y=(0.22 if nrows == 2 else 0.50), len=(0.42 if nrows == 2 else 0.85)),
                        customdata=_custom_xlabels_2d(),
                        hovertemplate="時点:%{customdata}<br>水深:%{y} m<br>差:%{z:.2f} (log10)<extra></extra>"
                    ),
                    row=row_i, col=1
                )

                fig.update_yaxes(autorange="reversed", title_text="水深 (m)", row=row_i, col=1)
       
            # --- MY / ANFC 境界線（同月日比較：基準年データにANFCが含まれる期間の切替点に縦線） ---

            def _cmem_boundary_xidx_base(df_a: pd.DataFrame, df_b: pd.DataFrame) -> List[int]:
                frames = []
                for df0 in (df_a, df_b):
                    if df0 is None or df0.empty:
                        continue
                    if 'Source' not in df0.columns:
                        continue
                    dts = pd.to_datetime(df0[dt_col], errors='coerce')
                    base_mask = (dts.dt.year == base_y)
                    if not base_mask.any():
                        continue
                    d = df0.loc[base_mask, [dt_col, 'Source']].copy()
                    d[dt_col] = pd.to_datetime(d[dt_col], errors='coerce')
                    d = d.dropna(subset=[dt_col])
                    if d.empty:
                        continue

                    if cmem_period == '月別':
                        d['_m'] = d[dt_col].dt.month
                        d = d[d['_m'].isin(month_order)]
                        d['_x'] = d['_m'].map(month_order)
                    else:
                        d['_md'] = d[dt_col].dt.strftime('%m-%d')
                        d = d[d['_md'].isin(md_order)]
                        d['_x'] = d['_md'].map(md_order)

                    d['_anfc'] = d['Source'].astype(str).str.strip().str.upper().eq('ANFC')
                    frames.append(d[['_x', '_anfc']])

                if not frames:
                    return []
                u = pd.concat(frames, ignore_index=True)
                if u.empty:
                    return []
                g = u.groupby('_x', as_index=False)['_anfc'].any().sort_values('_x')
                xs = g['_x'].tolist()
                flags = g['_anfc'].tolist()
                out = []
                for i in range(1, len(xs)):
                    if flags[i] != flags[i - 1]:
                        out.append(xs[i])
                return out

            boundary_x = _cmem_boundary_xidx_base(df_t2, df_c2)
            if boundary_x:
                for bx in boundary_x:
                    for rr in range(1, nrows + 1):
                        fig.add_vline(x=bx, row=rr, col=1, line_width=6, line_dash='solid', line_color='white', opacity=0.75)
                        fig.add_vline(x=bx, row=rr, col=1, line_width=2, line_dash='dot', line_color='black', opacity=0.85)

            fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext)
            fig.update_xaxes(title_text=("月" if cmem_period == "月別" else "月日"), row=nrows, col=1)
            fig.update_layout(
                title=f"CMEM {sel_site} 同月比較差分（{base_y} − 平均[{','.join(map(str, comp_sorted))}]）",
                height=(520 if nrows == 1 else 760),
                margin=dict(l=60, r=110, t=60, b=50),
                template="plotly_white"
            )

            st.plotly_chart(fig, use_container_width=True)

    with tab_cmem_ts:
        dt_min, dt_max = _cmem_dt_minmax(df_t2_raw, df_c2_raw)
        if dt_min is None or dt_max is None:
            st.warning("日時情報がありません")
            st.stop()
        d0 = pd.to_datetime(dt_min).date()
        d1 = pd.to_datetime(dt_max).date()
        try:
            _dflt_start = (pd.Timestamp(d1) - pd.DateOffset(years=2)).date()
        except Exception:
            _dflt_start = (pd.Timestamp(d1) - pd.Timedelta(days=365*2)).date()
        _dflt_start = max(d0, _dflt_start)
        d_start, d_end = st.slider("", min_value=d0, max_value=d1, value=(_dflt_start, d1), key="cmem_dt_range_ts_slider", label_visibility="collapsed")
        t_start = pd.Timestamp(d_start)
        t_end = pd.Timestamp(d_end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        df_t2_ts = df_t2_raw.copy()
        df_c2_ts = df_c2_raw.copy()
        if not df_t2_ts.empty:
            _dt = pd.to_datetime(df_t2_ts[dt_col], errors='coerce')
            df_t2_ts = df_t2_ts[(_dt >= t_start) & (_dt <= t_end)].copy()
        if not df_c2_ts.empty:
            _dt = pd.to_datetime(df_c2_ts[dt_col], errors='coerce')
            df_c2_ts = df_c2_ts[(_dt >= t_start) & (_dt <= t_end)].copy()
        _render_cmem("時系列", df_t2_ts, df_c2_ts)
    with tab_cmem_md:
        _render_cmem("同月日比較", df_t2_raw, df_c2_raw)

def render_calendar_mode():
    """wt_test と同等の予測カレンダー（corr/<name>_corr.csv 優先）。"""
    # pred ファイル選択
    pred_folder = pjoin(BASE_DIR, PRED_DIR)
    if not os.path.exists(pred_folder):
        st.error(f"フォルダが見つかりません: {pred_folder}")
        st.stop()
    pred_files = [f for f in os.listdir(pred_folder) if f.lower().endswith('.csv')]
    if not pred_files:
        st.warning("pred に CSV がありません")
        st.stop()

    selected_file = st.selectbox("対象エリアを選択", sorted(pred_files), key='cal_selected_file', label_visibility="collapsed")

    # 指紋（キャッシュキー）
    pred_path = pjoin(BASE_DIR, PRED_DIR, selected_file)
    name, ext = os.path.splitext(selected_file)
    corr_path = pjoin(BASE_DIR, CORR_DIR, f"{name}_corr{ext}")
    obs_path  = pjoin(BASE_DIR, OBS_DIR, selected_file)
    fp_pred = file_fingerprint(pred_path)
    fp_corr = file_fingerprint(corr_path)
    fp_obs  = file_fingerprint(obs_path)

    # 読み込み
    df_pred = load_pred(selected_file, fp_pred)
    df_corr = load_corr_for(selected_file, fp_corr)
    df_obs  = load_obs_for(selected_file, fp_obs)
    corr_available = not df_corr.empty

    if df_pred.empty:
        st.warning("予測データが読み込めませんでした")
        st.stop()

    # UI（wt_test 準拠）
    today_jst = pd.Timestamp.now(tz="Asia/Tokyo").date()
    latest_day = df_pred["date_day"].max()
    available_days = sorted(df_pred["date_day"].unique())
    min_day = min(available_days) if available_days else latest_day
    max_day = max(available_days) if available_days else latest_day

    try:
        cal_choice = safe_segmented_control(
            "", options=["週間表示", "選択日"],
            default="週間表示", key='cal_choice'
        )
    except Exception:
        cal_choice = st.radio(
            "", ["週間表示", "選択日"],
            index=0, horizontal=True, key='cal_choice_radio',
            label_visibility='collapsed'
        )

    # ===== 週間（昼頃） =====
    if cal_choice == "週間表示":
        base_day_week = min(max(today_jst, min_day), max_day)
        selected_day = st.date_input(
            "", value=base_day_week,
            min_value=min_day, max_value=max_day,
            key='cal_week_base_day', label_visibility='collapsed'
        )

        if WEEK_WINDOW_FORWARD:
            start_day = pd.Timestamp(selected_day)
            end_day = start_day + pd.Timedelta(days=7)
        else:
            end_day = pd.Timestamp(selected_day)
            start_day = end_day - pd.Timedelta(days=7)

        day_list = list(pd.date_range(start_day, end_day, freq='D'))
        df_period = df_pred[df_pred["date_day"].isin([d.date() for d in day_list])].copy()

        # corr 付与
        if corr_available:
            df_corr_period = df_corr[df_corr["date_day"].isin([d.date() for d in day_list])].copy()
            df_period = add_corr(df_period, df_corr_period)

        # obs 温度（週コメント判定用）
        if (not df_obs.empty) and (not df_period.empty):
            df_obs_week = df_obs[df_obs["date_day"].between(day_list[0].date(), day_list[-1].date())].copy()
            tol_obs = pd.Timedelta(minutes=OBS_MATCH_TOL_MIN)
            left = df_period.sort_values(["depth_m","datetime"]).copy()
            right = df_obs_week.sort_values(["depth_m","datetime"])[["datetime","depth_m","obs_temp"]].copy()
            df_period = safe_merge_asof_by_depth_keep_left(
                left, right, tolerance=tol_obs,
                right_value_cols=["obs_temp"], suffixes=("", "")
            )

        depths_all = sorted([int(d) for d in df_pred["depth_m"].dropna().unique()])
        # 週コメント（表層/中層/底層）

        with st.expander(f'コメント（{start_day:%m/%d}～{end_day:%m/%d}の推移）', expanded=False):
                    layers = make_layer_groups(depths_all)
                    any_line = False
                    for lname, ldepths in layers.items():
                        line = summarize_weekly_layer_temp(lname, ldepths, df_period)
                        if line:
                            any_line = True
                            st.markdown(line)
                    if not any_line:
                        st.caption("（特筆すべき変化はありません）")

        table_html = build_weekly_table_html(df_period, day_list, depths_all, corr_on=corr_available)
        styles = get_calendar_css(65)
        full_html = f"<!doctype html><html><head><meta charset='utf-8'>{styles}</head><body>{table_html}</body></html>"
        st_html(full_html, height=650, scrolling=True)

    # ===== 選択日 =====
    else:
        base_day_day = min(max(today_jst, min_day), max_day)
        selected_day = st.date_input(
            "", value=base_day_day,
            min_value=min_day, max_value=max_day,
            key='cal_day_sel', label_visibility='collapsed'
        )

        df_day = df_pred[df_pred["date_day"] == selected_day].copy()
        if corr_available:
            df_corr_sel = df_corr[df_corr["date_day"] == selected_day].copy()
            df_day = add_corr(df_day, df_corr_sel)

        if (not df_obs.empty) and (not df_day.empty):
            df_obs_sel = df_obs[df_obs["date_day"] == selected_day].copy()
            tol_obs = pd.Timedelta(minutes=OBS_MATCH_TOL_MIN)
            left = df_day.sort_values(["depth_m","datetime"]).copy()
            right = df_obs_sel.sort_values(["depth_m","datetime"])[["datetime","depth_m","obs_temp"]].copy()
            df_day = safe_merge_asof_by_depth_keep_left(
                left, right, tolerance=tol_obs,
                right_value_cols=["obs_temp"], suffixes=("", "")
            )

        depths_all = sorted([int(d) for d in df_pred["depth_m"].dropna().unique()])
        with st.expander('コメント（朝(4～6時)、昼(11～13時)、夕(16～18時)）', expanded=False):
                    any_line = False
                    layers = make_layer_groups(depths_all)
                    for lname, ldepths in layers.items():
                        line = summarize_daily_layer_flow(lname, ldepths, df_day)
                        if line:
                            any_line = True
                            st.markdown(line)
                    if not any_line:
                        st.caption("（特筆すべき変化はありません）")

        table_html = build_daily_table_html(df_day, depths_all, corr_on=corr_available)
        styles = get_calendar_css(65)
        full_html = f"<!doctype html><html><head><meta charset='utf-8'>{styles}</head><body>{table_html}</body></html>"
        st_html(full_html, height=650, scrolling=True)
def metrics_gsi(df_gsi, area: str, year: int, week: int, eps: float = 0.2):
    import pandas as pd
    import math
    g_week = df_gsi[(df_gsi['Area'].astype(str).str.strip() == str(area).strip()) & (df_gsi['Year'] == year) & (df_gsi['week'] == week)].copy()
    if g_week.empty:
        return float('nan'), 'データ不足', float('nan'), 'データ不足'
    latest_date = pd.to_datetime(g_week['Date'], errors='coerce').max()
    g_area_year = df_gsi[(df_gsi['Area'].astype(str).str.strip() == str(area).strip()) & (df_gsi['Year'] == year)].copy()
    g_area_year['Date'] = pd.to_datetime(g_area_year['Date'], errors='coerce')
    g_area_year = g_area_year.dropna(subset=['Date'])
    cur = g_area_year[g_area_year['Date'] == latest_date]
    prev_candidates = g_area_year[g_area_year['Date'] < latest_date]
    prev_date = prev_candidates['Date'].max() if not prev_candidates.empty else None
    prev = g_area_year[g_area_year['Date'] == prev_date] if prev_date is not None else pd.DataFrame()
    cur_F = float(pd.to_numeric(cur.loc[cur.get('Sex','').astype(str).str.upper().str.strip()=='F','GSI'], errors='coerce').mean()) if not cur.empty else float('nan')
    cur_M = float(pd.to_numeric(cur.loc[cur.get('Sex','').astype(str).str.upper().str.strip()=='M','GSI'], errors='coerce').mean()) if not cur.empty else float('nan')
    prev_F = float(pd.to_numeric(prev.loc[prev.get('Sex','').astype(str).str.upper().str.strip()=='F','GSI'], errors='coerce').mean()) if not prev.empty else float('nan')
    prev_M = float(pd.to_numeric(prev.loc[prev.get('Sex','').astype(str).str.upper().str.strip()=='M','GSI'], errors='coerce').mean()) if not prev.empty else float('nan')
    def trend(cur_v, base_v):
        if math.isnan(cur_v) or math.isnan(base_v): return 'データ不足'
        d = cur_v - base_v
        if d > eps:  return '上昇'
        if d < -eps: return '下降'
        return '変化なし'
    return cur_F, trend(cur_F, prev_F), cur_M, trend(cur_M, prev_M)


def metrics_larvae(df_larv, area: str, year: int, week: int):
    import pandas as pd
    import numpy as np
    sub_week = df_larv[(df_larv['Area'].astype(str).str.strip() == str(area).strip()) & (df_larv['Year'] == year) & (df_larv['week'] == week)].copy()
    if sub_week.empty:
        return 0.0, 0.0, 0.0, float('nan'), float('nan')
    sub_week['Date'] = pd.to_datetime(sub_week['Date'], errors='coerce')
    latest_date = sub_week['Date'].max()
    cur = sub_week[sub_week['Date'] == latest_date].copy()
    size_cols = [c for c in cur.columns if str(c).isdigit()]
    for c in size_cols: cur[c] = pd.to_numeric(cur[c], errors='coerce').fillna(0.0)
    qty_200_259 = float(cur[[c for c in size_cols if 200 <= int(c) <= 259]].sum().sum()) if size_cols else 0.0
    qty_ge260   = float(cur[[c for c in size_cols if int(c) >= 260]].sum().sum()) if size_cols else 0.0
    qty_ge200   = qty_200_259 + qty_ge260
    area_year = df_larv[(df_larv['Area'].astype(str).str.strip() == str(area).strip()) & (df_larv['Year'] == year)].copy()
    area_year['Date'] = pd.to_datetime(area_year['Date'], errors='coerce')
    prev_candidates = area_year[area_year['Date'] < latest_date]
    prev_date = prev_candidates['Date'].max() if not prev_candidates.empty else None
    diff_prev = float('nan')
    if prev_date is not None:
        prev = area_year[area_year['Date'] == prev_date].copy()
        size_cols_prev = [c for c in prev.columns if str(c).isdigit()]
        for c in size_cols_prev: prev[c] = pd.to_numeric(prev[c], errors='coerce').fillna(0.0)
        prev_ge200 = float(prev[[c for c in size_cols_prev if int(c)>=200]].sum().sum()) if size_cols_prev else 0.0
        diff_prev = qty_ge200 - prev_ge200
    py = df_larv[(df_larv['Area'].astype(str).str.strip() == str(area).strip()) & (df_larv['Year'] == year-1) & (df_larv['week'] == week)].copy()
    if not py.empty:
        for c in [c for c in py.columns if str(c).isdigit()]: py[c] = pd.to_numeric(py[c], errors='coerce').fillna(0.0)
        py_ge200 = float(py[[c for c in py.columns if c.isdigit() and int(c)>=200]].sum().sum())
        diff_prevY = qty_ge200 - py_ge200
    else:
        diff_prevY = float('nan')
    return qty_ge200, qty_200_259, qty_ge260, diff_prev, diff_prevY


def metrics_temp10m(base_dir: str, area: str, year: int, week: int, eps_t: float=0.5):
    import pandas as pd
    import math
    try:
        df_dr = load_dr_single_file(base_dir, area)
    except Exception:
        return float('nan'), 'データ不足', 'データ不足'
    if df_dr.empty:
        return float('nan'), 'データ不足', 'データ不足'
    dt = pd.to_datetime(df_dr['datetime'], errors='coerce')
    df_dr['week'] = dt.dt.isocalendar().week.astype(int)
    df_dr['year'] = dt.dt.year
    df_10 = df_dr[(df_dr['depth_m']==10) & (df_dr['year']==year) & (df_dr['week']==week)].copy()
    cur = float(pd.to_numeric(df_10.get('pred_temp'), errors='coerce').mean()) if not df_10.empty else float('nan')
    prev_week = max(int(week)-1, 1)
    p10 = df_dr[(df_dr['depth_m']==10) & (df_dr['year']==year) & (df_dr['week']==prev_week)].copy()
    prev = float(pd.to_numeric(p10.get('pred_temp'), errors='coerce').mean()) if not p10.empty else float('nan')
    y10 = df_dr[(df_dr['depth_m']==10) & (df_dr['year']==(year-1)) & (df_dr['week']==week)].copy()
    prevY = float(pd.to_numeric(y10.get('pred_temp'), errors='coerce').mean()) if not y10.empty else float('nan')
    def trend(cur_v, base_v):
        if math.isnan(cur_v) or math.isnan(base_v): return 'データ不足'
        d = cur_v - base_v
        if d > eps_t:  return '上昇'
        if d < -eps_t: return '下降'
        return '変化なし'
    return cur, trend(cur, prev), trend(cur, prevY)

def render_map_mode():
    import pandas as pd
    import numpy as np
    import streamlit as st
    from datetime import datetime
    import html as _html

    AREA_FILE   = pjoin(base_dir, "file_summary.csv")
    GSI_FILE    = MATURITY_PATH
    LARVAE_FILE = LARVAE_PATH

    def _read(path):
        for enc in ("utf-8", "utf-8-sig", "cp932"):
            try:
                df = pd.read_csv(path, encoding=enc)
                df.columns = [c.strip() for c in df.columns]
                return df
            except Exception:
                continue
        return None

    try:
        df_area = read_csv_path(AREA_FILE)
        df_gsi  = read_csv_path(GSI_FILE)
        df_larv = read_csv_path(LARVAE_FILE)
    except NameError:
        df_area = _read(AREA_FILE)
        df_gsi  = _read(GSI_FILE)
        df_larv = _read(LARVAE_FILE)

    if df_area is None or df_gsi is None or df_larv is None:
        st.warning("file_summary.csv / maturity.csv / larvae.csv を配置してください。")
        return

    df_area["Area"] = df_area.get("Area", "").astype(str).str.strip()
    for col in ["Laf", "Lof"]:
        if col in df_area.columns:
            df_area[col] = pd.to_numeric(df_area[col], errors="coerce")

    # GSI
    df_gsi["Date"] = pd.to_datetime(df_gsi.get("Date"), errors="coerce")
    df_gsi["Area"] = df_gsi.get("Area", "").astype(str).str.strip()
    df_gsi["GSI"] = (
        df_gsi.get("GSI", np.nan)
        .astype(str)
        .str.replace("%", "", regex=False)
        .replace("", np.nan)
    )
    df_gsi["GSI"] = pd.to_numeric(df_gsi["GSI"], errors="coerce")
    iso_g = df_gsi["Date"].dt.isocalendar()
    df_gsi["ISOYear"] = iso_g.year.astype("Int64")
    df_gsi["week"]    = iso_g.week.astype("Int64")

    # Larvae
    df_larv["Date"] = pd.to_datetime(df_larv.get("Date"), errors="coerce")
    df_larv["Area"] = df_larv.get("Area", "").astype(str).str.strip()
    iso_l = df_larv["Date"].dt.isocalendar()
    df_larv["ISOYear"] = iso_l.year.astype("Int64")
    df_larv["week"]    = iso_l.week.astype("Int64")

    years_all = sorted(
        set(df_gsi["ISOYear"].dropna().astype(int).unique()).union(
            set(df_larv["ISOYear"].dropna().astype(int).unique())
        )
    )
    if not years_all:
        st.info("年度データがありません。")
        return

    c0, c1, c2, c3 = st.columns([1.2, 1.0, 1.2, 2.0])

    with c0:
        mode = st.radio(
            "",
            ["GSI", "ラーバ"],
            index=0,
            key="map_mode",
            horizontal=True,
            label_visibility="collapsed",
        )

    with c1:
        base_year = st.selectbox(
            "",
            years_all,
            index=len(years_all) - 1,
            key="map_base_year",
            label_visibility="collapsed",
        )

    with c2:
        # ラーバ通常デフォルト=週集計
        norm_data_mode = st.radio(
            "",
            ["生データ", "週集計"],
            index=1,
            key="map_norm_data_mode",
            horizontal=True,
            label_visibility="collapsed",
        )

    with c3:
        candidates = [y for y in years_all if int(y) != int(base_year)]
        default_comp = candidates[-2:] if len(candidates) >= 2 else candidates
        comp_years = st.multiselect(
            "",
            candidates,
            default=default_comp,
            key="map_comp_years",
            label_visibility="collapsed",
        )

    tab_norm, tab_cmp = st.tabs(["通常", "比較"])


    colors_gsi    = ["#d62728", "#ff7f0e", "#1f77b4"]  # ≥25 / 20–24.9 / <20
    colors_larvae = ["#1f77b4", "#ff7f0e", "#d62728"]  # <200 / 200–259 / ≥260

    EMPH_GSI_IDX = [2]       # values=[ge25, mid, lt20] の lt20
    EMPH_LARV_IDX = [1, 2]   # values=[lt200, mid, ge260] の mid, ge260

    def _hex_to_rgb(h: str):
        h = (h or "").lstrip("#")
        if len(h) == 3:
            h = "".join([c * 2 for c in h])
        try:
            return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        except Exception:
            return 150, 150, 150

    def _hex_to_rgba(h: str, a: float):
        r, g, b = _hex_to_rgb(h)
        return f"rgba({r},{g},{b},{a:.3f})"

    def _donut_sector_path(cx, cy, r_out, r_in, start_deg, end_deg):
        a0 = np.radians(start_deg)
        a1 = np.radians(end_deg)

        x0o = cx + r_out * np.cos(a0)
        y0o = cy + r_out * np.sin(a0)
        x1o = cx + r_out * np.cos(a1)
        y1o = cy + r_out * np.sin(a1)

        x0i = cx + r_in * np.cos(a0)
        y0i = cy + r_in * np.sin(a0)
        x1i = cx + r_in * np.cos(a1)
        y1i = cy + r_in * np.sin(a1)

        span = (end_deg - start_deg) % 360.0
        large = 1 if span > 180.0 else 0

        d = (
            f"M{x0o},{y0o} "
            f"A{r_out},{r_out} 0 {large},1 {x1o},{y1o} "
            f"L{x1i},{y1i} "
            f"A{r_in},{r_in} 0 {large},0 {x0i},{y0i} "
            f"Z"
        )
        return d

    def _pie_wedge_path(cx, cy, r, start_deg, end_deg):
        a0 = np.radians(start_deg)
        a1 = np.radians(end_deg)

        x1 = cx + r * np.cos(a0)
        y1 = cy + r * np.sin(a0)
        x2 = cx + r * np.cos(a1)
        y2 = cy + r * np.sin(a1)

        span = (end_deg - start_deg) % 360.0
        large = 1 if span > 180.0 else 0

        return f"M{cx},{cy} L{x1},{y1} A{r},{r} 0 {large},1 {x2},{y2} Z"

    def svg_pie(
        values, colors, size=60, labels=None, alpha=0.60, center_text=None, hover_text=None,
        stroke_width=2,
        emphasize_idxs=None,
        emphasize_alpha_boost=0.0,
        emphasize_stroke="rgba(0,0,0,0.35)",
        emphasize_stroke_width=2.0
    ):
        """単一円（通常用）: total<=0 なら空文字（数字だけ出る事故を防ぐ）"""
        total = float(np.nansum(values))
        if (not np.isfinite(total)) or total <= 0:
            return ""

        emphasize_set = set(emphasize_idxs or [])
        cx, cy, r = size / 2, size / 2, size / 2 - 2
        hover_text = (hover_text or "").strip()

        def _title(i, v):
            return hover_text if hover_text else (f"{labels[i]}: {v}" if labels else f"値: {v}")

        def _center_text_svg():
            if center_text is None:
                return ""
            s = str(center_text).strip()
            if s == "":
                return ""
            fz = max(8, int(size * 0.20))
            return (
                f"<text x='{cx}' y='{cy+4}' text-anchor='middle' "
                f"font-size='{fz}' font-weight='700' "
                f"fill='rgba(0,0,0,0.78)' "
                f"stroke='rgba(255,255,255,0.85)' stroke-width='2' paint-order='stroke'>"
                f"{_html.escape(s)}</text>"
            )

        def _slice_style(i, base_alpha):
            if i in emphasize_set:
                a = min(1.0, float(base_alpha) + float(emphasize_alpha_boost))
                return a, emphasize_stroke, float(emphasize_stroke_width)
            return float(base_alpha), "rgba(255,255,255,0.90)", float(stroke_width)

        nonzeros = [(i, float(v)) for i, v in enumerate(values)
                    if (v is not None and np.isfinite(v) and float(v) > 1e-12)]
        if len(nonzeros) == 1:
            i, v = nonzeros[0]
            c = colors[i] if colors and i < len(colors) else "#888888"
            a, stc, stw = _slice_style(i, alpha)
            fill = _hex_to_rgba(c, a) if isinstance(c, str) and c.startswith("#") else c

            svg = f"<svg width='{size}' height='{size}' viewBox='0 0 {size} {size}'>"
            svg += (
                f"<circle cx='{cx}' cy='{cy}' r='{r}' fill='{fill}' "
                f"stroke='{stc}' stroke-width='{stw}'>"
                f"<title>{_html.escape(_title(i, v))}</title></circle>"
            )
            ct = _center_text_svg()
            if ct:
                svg += ct
            svg += "</svg>"
            return svg

        svg = f"<svg width='{size}' height='{size}' viewBox='0 0 {size} {size}'>"
        start = 0.0
        EPS = 1e-9
        for i, v in enumerate(values):
            v = float(v) if v is not None else 0.0
            if (not np.isfinite(v)) or v <= EPS:
                continue
            ang = (v / total) * 360.0
            end = start + ang
            c = colors[i] if colors and i < len(colors) else "#888888"
            a, stc, stw = _slice_style(i, alpha)
            fill = _hex_to_rgba(c, a) if isinstance(c, str) and c.startswith("#") else c
            d = _pie_wedge_path(cx, cy, r, start, end)
            svg += f"<path d='{d}' fill='{fill}' stroke='{stc}' stroke-width='{stw}'>"
            svg += f"<title>{_html.escape(_title(i, v))}</title></path>"
            start = end

        ct = _center_text_svg()
        if ct:
            svg += ct
        svg += "</svg>"
        return svg

    def svg_ring_with_pie_core(
        core_values, ring_values, colors, size=60, labels=None,
        core_alpha=0.92, ring_alpha=0.25,
        center_text=None, hover_text=None,
        draw_ring=True,
        core_stroke_width=3,
        ring_stroke_width=2,
        emphasize_core_idxs=None,
        emphasize_ring_idxs=None,
        emphasize_alpha_boost=0.0,
        emphasize_stroke="rgba(0,0,0,0.35)",
        emphasize_stroke_width=2.0
    ):
        """
        比較用:
          - 基準(core)は必ず描く（ただし core_total<=0 なら空を返す）
          - 比較(ring)は draw_ring=True かつ ring_total>0 のときだけ描く
          - GSI比較は center_text=None（中心数字なし）
          - ラーバ比較は center_text=Δtotal（両方揃う点のみ）
          - ★100%（単一カテゴリ）でも必ず描画する
        """
        core_total = float(np.nansum(core_values))
        if (not np.isfinite(core_total)) or core_total <= 0:
            return ""  # 「数字だけ」を防ぐ

        hover_text = (hover_text or "").strip()
        emph_core = set(emphasize_core_idxs or [])
        emph_ring = set(emphasize_ring_idxs or [])

        cx, cy = size / 2, size / 2

        r_out = size / 2 - 2
        ring_thick = max(7, int(size * 0.15))
        gap = max(2, int(size * 0.03))
        r_ring_in = r_out - ring_thick
        r_core = max(2, r_ring_in - gap)

        def _title(i, v):
            if hover_text:
                return hover_text
            return (f"{labels[i]}: {v}" if labels else f"値: {v}")

        def _center_text_svg():
            if center_text is None:
                return ""
            s = str(center_text).strip()
            if s == "":
                return ""
            fz = max(8, int(size * 0.20))
            return (
                f"<text x='{cx}' y='{cy+4}' text-anchor='middle' "
                f"font-size='{fz}' font-weight='700' "
                f"fill='rgba(0,0,0,0.82)' "
                f"stroke='rgba(255,255,255,0.90)' stroke-width='2' paint-order='stroke'>"
                f"{_html.escape(s)}</text>"
            )

        def _slice_style(i, base_alpha, is_emph: bool):
            if is_emph:
                a = min(1.0, float(base_alpha) + float(emphasize_alpha_boost))
                return a, emphasize_stroke, float(emphasize_stroke_width)
            return float(base_alpha), None, None

        def _draw_ring(values):
            if not draw_ring:
                return ""
            total = float(np.nansum(values))
            if (not np.isfinite(total)) or total <= 0:
                return ""

            # ★リングが単一カテゴリ100%なら「太いstroke円」で描く（強調も対応）
            nonz = [(i, float(v)) for i, v in enumerate(values)
                    if (v is not None and np.isfinite(v) and float(v) > 1e-12)]
            if len(nonz) == 1:
                i, v = nonz[0]
                c = colors[i] if colors and i < len(colors) else "#888888"
                a, stc, stw = _slice_style(i, ring_alpha, (i in emph_ring))
                stroke_col = _hex_to_rgba(c, a) if isinstance(c, str) and c.startswith("#") else c
                r_mid = (r_out + r_ring_in) / 2.0
                w_ring = (r_out - r_ring_in)
                svg = (
                    f"<circle cx='{cx}' cy='{cy}' r='{r_mid}' fill='none' "
                    f"stroke='{stroke_col}' stroke-width='{w_ring}' stroke-linecap='butt'>"
                    f"<title>{_html.escape(_title(i, v))}</title></circle>"
                )
                if i in emph_ring:
                    svg += (
                        f"<circle cx='{cx}' cy='{cy}' r='{r_mid}' fill='none' "
                        f"stroke='{stc}' stroke-width='{max(2.4, float(stw))}' stroke-linecap='round' />"
                    )
                return svg

            parts = []
            start = 0.0
            EPS = 1e-9
            for i, v in enumerate(values):
                v = float(v) if v is not None else 0.0
                if (not np.isfinite(v)) or v <= EPS:
                    continue
                ang = (v / total) * 360.0
                end = start + ang
                c = colors[i] if colors and i < len(colors) else "#888888"
                a, stc, stw = _slice_style(i, ring_alpha, (i in emph_ring))
                fill = _hex_to_rgba(c, a) if isinstance(c, str) and c.startswith("#") else c
                d = _donut_sector_path(cx, cy, r_out, r_ring_in, start, end)

                if i in emph_ring:
                    use_stroke = stc
                    use_stw = stw
                else:
                    use_stroke = "rgba(255,255,255,0.90)"
                    use_stw = ring_stroke_width

                parts.append(
                    f"<path d='{d}' fill='{fill}' stroke='{use_stroke}' stroke-width='{use_stw}'>"
                    f"<title>{_html.escape(_title(i, v))}</title></path>"
                )
                start = end
            return "".join(parts)

        def _draw_core(values):
            total = float(np.nansum(values))
            if (not np.isfinite(total)) or total <= 0:
                return ""

            # ★コアが単一カテゴリ100%なら circle で描く（強調も対応）
            nonz = [(i, float(v)) for i, v in enumerate(values)
                    if (v is not None and np.isfinite(v) and float(v) > 1e-12)]
            if len(nonz) == 1:
                i, v = nonz[0]
                c = colors[i] if colors and i < len(colors) else "#888888"
                a, stc, stw = _slice_style(i, core_alpha, (i in emph_core))
                fill = _hex_to_rgba(c, a) if isinstance(c, str) and c.startswith("#") else c

                if i in emph_core:
                    stroke = stc
                    sw = stw
                else:
                    stroke = "rgba(255,255,255,0.95)"
                    sw = core_stroke_width

                return (
                    f"<circle cx='{cx}' cy='{cy}' r='{r_core}' fill='{fill}' "
                    f"stroke='{stroke}' stroke-width='{sw}'>"
                    f"<title>{_html.escape(_title(i, v))}</title></circle>"
                )

            parts = []
            start = 0.0
            EPS = 1e-9
            for i, v in enumerate(values):
                v = float(v) if v is not None else 0.0
                if (not np.isfinite(v)) or v <= EPS:
                    continue
                ang = (v / total) * 360.0
                end = start + ang
                c = colors[i] if colors and i < len(colors) else "#888888"
                a, stc, stw = _slice_style(i, core_alpha, (i in emph_core))
                fill = _hex_to_rgba(c, a) if isinstance(c, str) and c.startswith("#") else c
                d = _pie_wedge_path(cx, cy, r_core, start, end)

                if i in emph_core:
                    use_stroke = stc
                    use_stw = stw
                else:
                    use_stroke = "rgba(255,255,255,0.95)"
                    use_stw = core_stroke_width

                parts.append(
                    f"<path d='{d}' fill='{fill}' stroke='{use_stroke}' stroke-width='{use_stw}'>"
                    f"<title>{_html.escape(_title(i, v))}</title></path>"
                )
                start = end
            return "".join(parts)

        svg = f"<svg width='{size}' height='{size}' viewBox='0 0 {size} {size}'>"
        svg += _draw_ring(ring_values)
        svg += _draw_core(core_values)
        ct = _center_text_svg()
        if ct:
            svg += ct
        svg += "</svg>"
        return svg

    def _areas_sorted():
        tmp = df_area.copy()
        tmp["Area"] = tmp.get("Area", "").astype(str).str.strip()
        ok = tmp.dropna(subset=["Laf", "Lof"]).copy()
        if not ok.empty:
            ok = ok.sort_values(["Laf", "Lof"], ascending=[False, True])
            return ok["Area"].dropna().astype(str).str.strip().unique().tolist()
        return sorted(tmp["Area"].dropna().astype(str).str.strip().unique().tolist())

    def _render_xy(points, mondays, week_to_idx, data_mode, legend_html):
        areas_all = _areas_sorted()

        X_STEP = 54
        ROW_H  = 66
        LEFT_W = 130
        TOP_H  = 46
        DAY_STEP = X_STEP / 7.0

        x_labels = []
        prev_m = None
        for i, m in enumerate(mondays):
            show = (i % 2 == 0)
            if prev_m is None or m.month != prev_m:
                show = True
            prev_m = m.month
            x_labels.append((i, m, show))

        month_lines = []
        prev_m = None
        for i, m in enumerate(mondays):
            if prev_m is None:
                prev_m = m.month
                continue
            if m.month != prev_m:
                month_lines.append(i)
            prev_m = m.month

        n_x = len(mondays)
        n_y = len(areas_all)
        plot_w = (n_x - 1) * X_STEP + 2 * X_STEP
        plot_h = n_y * ROW_H

        def x_from_date(dt: pd.Timestamp):
            if pd.isna(dt):
                return None
            iso = pd.Timestamp(dt).isocalendar()
            w = int(iso.week)
            if w not in week_to_idx:
                return None
            i = week_to_idx[w]
            if data_mode == "週集計":
                return i * X_STEP + X_STEP
            dow = int(pd.Timestamp(dt).dayofweek)
            return i * X_STEP + X_STEP + dow * DAY_STEP

        css = f"""
        <style>
          .xywrap {{ border:1px solid #e5e5e5; border-radius:12px; background:#fff; padding:6px; }}
          .xyframe {{ display:grid; grid-template-columns:{LEFT_W}px 1fr; grid-template-rows:{TOP_H}px {plot_h}px; gap:0; }}
          .corner {{ grid-column:1; grid-row:1; background:#fafafa; border-right:1px solid #eee; border-bottom:1px solid #eee; }}
          .ylabels {{ grid-column:1; grid-row:2; background: rgba(255,255,255,0.98); border-right:1px solid #eee; }}
          .ylabel {{ height:{ROW_H}px; display:flex; align-items:center; justify-content:flex-start; padding-left:10px; font-weight:700; font-size:13px; color:#333;
                    box-sizing:border-box; border-bottom:1px solid #f3f3f3; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
          .xscroll {{ grid-column:2; grid-row:1 / span 2; overflow-x:auto; overflow-y:hidden; -webkit-overflow-scrolling:touch; }}
          .canvas {{ position:relative; width:{plot_w}px; height:{TOP_H + plot_h}px; }}
          .xaxis {{ position:sticky; top:0; height:{TOP_H}px; background: rgba(255,255,255,0.98); border-bottom:1px solid #eee; z-index:10; }}
          .xtick {{ position:absolute; bottom:6px; transform:translateX(-50%); font-size:12px; color:#444; white-space:nowrap; }}
          .xminor {{ position:absolute; top:0; width:1px; height:{TOP_H}px; background:#f0f0f0; }}
          .vline {{ position:absolute; left:0; top:{TOP_H}px; width:1px; height:{plot_h}px; background:#f0f0f0; z-index:2; }}
          .vline.month {{ background:#d9d9d9; }}
          .hline {{ position:absolute; left:0; top:0; width:{plot_w}px; height:1px; background:#f3f3f3; z-index:1; }}
          .plot {{ position:absolute; left:0; top:{TOP_H}px; width:{plot_w}px; height:{plot_h}px; z-index:5; }}
          .dot {{ position:absolute; transform:translate(-50%,-50%); overflow:visible; pointer-events:auto; }}
        </style>
        """

        html = "<div class='xywrap'>"
        html += "<div class='xyframe'>"
        html += "<div class='corner'></div>"

        html += "<div class='ylabels'>"
        for a in areas_all:
            html += f"<div class='ylabel'>{_html.escape(str(a))}</div>"
        html += "</div>"

        html += "<div class='xscroll'><div class='canvas'>"

        html += "<div class='xaxis'>"
        for i, m, show in x_labels:
            x = i * X_STEP + X_STEP
            html += f"<div class='xminor' style='left:{x}px;'></div>"
            if show:
                html += f"<div class='xtick' style='left:{x}px;'>{m:%m/%d}</div>"
        html += "</div>"

        for i in range(n_x):
            x = i * X_STEP + X_STEP
            cls = "vline month" if i in month_lines else "vline"
            html += f"<div class='{cls}' style='left:{x}px;'></div>"

        for j in range(n_y + 1):
            y = j * ROW_H
            html += f"<div class='hline' style='top:{TOP_H + y}px;'></div>"

        html += "<div class='plot'>"
        area_to_y = {a: (i * ROW_H + ROW_H / 2) for i, a in enumerate(areas_all)}
        jitter_counter = {}

        for area, dt, svg, hover in points:
            if area not in area_to_y:
                continue
            y = area_to_y[area]
            x = x_from_date(pd.Timestamp(dt))
            if x is None:
                continue

            key = (area, pd.Timestamp(dt).date())
            k = jitter_counter.get(key, 0)
            jitter_counter[key] = k + 1
            if k == 0:
                xj = x
            else:
                step = 4 * (k // 2 + 1)
                sign = 1 if (k % 2 == 0) else -1
                xj = x + sign * step

            safe_tip = _html.escape(hover, quote=True)
            html += f"<div class='dot' title='{safe_tip}' style='left:{xj:.1f}px; top:{y}px;'>{svg}</div>"

        html += "</div>"  # plot
        html += "</div></div>"  # canvas, xscroll
        html += "</div>"  # xyframe
        html += legend_html
        html += "</div>"  # xywrap

        iframe_h = max(420, min(1100, TOP_H + plot_h + 120))
        st.components.v1.html(
            f"<!doctype html><html><head><meta charset='utf-8'>{css}</head><body>{html}</body></html>",
            height=iframe_h,
            scrolling=False
        )

    def _weekly_gsi_sum(df):
        d = df.copy()
        d = d.dropna(subset=["Date", "Area", "ISOYear", "week"])
        v = pd.to_numeric(d["GSI"], errors="coerce")
        d["ge25"] = (v >= 25).astype(int)
        d["mid"]  = ((v >= 20) & (v < 25)).astype(int)
        d["lt20"] = (v < 20).astype(int)
        d["n"]    = v.notna().astype(int)
        out = d.groupby(["Area", "ISOYear", "week"], as_index=False)[["ge25","mid","lt20","n"]].sum()
        return out

    def _weekly_larvae_mean(df):
        d = df.copy()
        d = d.dropna(subset=["Date", "Area", "ISOYear", "week"])
        d["Date"] = pd.to_datetime(d["Date"], errors="coerce")

        size_cols = [c for c in d.columns if str(c).isdigit()]
        for c in size_cols:
            d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)

        cols_lt200   = [c for c in size_cols if int(c) < 200]
        cols_200_259 = [c for c in size_cols if 200 <= int(c) <= 259]
        cols_ge260   = [c for c in size_cols if int(c) >= 260]

        d["lt200"] = d[cols_lt200].sum(axis=1) if cols_lt200 else 0.0
        d["mid"]   = d[cols_200_259].sum(axis=1) if cols_200_259 else 0.0
        d["ge260"] = d[cols_ge260].sum(axis=1) if cols_ge260 else 0.0
        d["total"] = d["lt200"] + d["mid"] + d["ge260"]

        day = (
            d.groupby(["Area","ISOYear","week","Date"], as_index=False)[["lt200","mid","ge260","total"]]
            .sum()
        )
        out = (
            day.groupby(["Area","ISOYear","week"], as_index=False)[["lt200","mid","ge260","total"]]
            .mean()
        )
        return out

    with tab_norm:
        base_df = df_gsi if mode == "GSI" else df_larv
        weeks_all = sorted(
            base_df[base_df["ISOYear"] == int(base_year)]["week"].dropna().astype(int).unique().tolist()
        )
        if not weeks_all:
            st.info("選択年に週データがありません。")
            return

        mondays = [datetime.fromisocalendar(int(base_year), int(w), 1) for w in weeks_all]
        week_to_idx = {w: i for i, w in enumerate(weeks_all)}
        points = []

        if mode == "GSI":
            d = df_gsi[df_gsi["ISOYear"] == int(base_year)].copy().dropna(subset=["Date"])
            d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
            d["Area"] = d["Area"].astype(str).str.strip()

            if norm_data_mode == "生データ":
                for (area, dt), g in d.groupby(["Area","Date"]):
                    area = str(area).strip()
                    if area == "" or pd.isna(dt):
                        continue
                    v = pd.to_numeric(g["GSI"], errors="coerce").dropna()
                    if v.empty:
                        continue
                    n_ge25 = int((v >= 25).sum())
                    n_mid  = int(((v >= 20) & (v < 25)).sum())
                    n_lt20 = int((v < 20).sum())
                    tot = n_ge25 + n_mid + n_lt20
                    if tot <= 0:
                        continue
                    hover = f"{pd.Timestamp(dt):%Y-%m-%d}"
                    svg = svg_pie(
                        [n_ge25, n_mid, n_lt20], colors_gsi, size=50,
                        labels=["≥25","20–24.9","<20"], alpha=0.60,
                        center_text=None, hover_text=hover, stroke_width=2,
                        emphasize_idxs=EMPH_GSI_IDX
                    )
                    if svg:
                        points.append((area, pd.Timestamp(dt), svg, hover))
            else:
                for (area, w), g in d.groupby(["Area","week"]):
                    area = str(area).strip()
                    if area == "" or pd.isna(w):
                        continue
                    v = pd.to_numeric(g["GSI"], errors="coerce").dropna()
                    if v.empty:
                        continue
                    n_ge25 = int((v >= 25).sum())
                    n_mid  = int(((v >= 20) & (v < 25)).sum())
                    n_lt20 = int((v < 20).sum())
                    tot = n_ge25 + n_mid + n_lt20
                    if tot <= 0:
                        continue
                    dt = pd.Timestamp(datetime.fromisocalendar(int(base_year), int(w), 1))
                    hover = f"{dt:%Y-%m-%d}"
                    svg = svg_pie(
                        [n_ge25, n_mid, n_lt20], colors_gsi, size=50,
                        labels=["≥25","20–24.9","<20"], alpha=0.60,
                        center_text=None, hover_text=hover, stroke_width=2,
                        emphasize_idxs=EMPH_GSI_IDX
                    )
                    if svg:
                        points.append((area, dt, svg, hover))

            legend_html = """
            <div style="display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-top:6px;font-size:13px;">
              <div style="font-weight:700;">凡例（GSI）</div>
              <div style="color:#666;">注目：&lt;20 を強調（太枠+濃色）</div>
              <div><span style="display:inline-block;width:12px;height:12px;background:#d62728;margin-right:6px;border:1px solid #fff;"></span>≥25</div>
              <div><span style="display:inline-block;width:12px;height:12px;background:#ff7f0e;margin-right:6px;border:1px solid #fff;"></span>20–24.9</div>
              <div><span style="display:inline-block;width:12px;height:12px;background:#1f77b4;margin-right:6px;border:1px solid #fff;"></span>&lt;20</div>
            </div>
            """
            _render_xy(points, mondays, week_to_idx, norm_data_mode, legend_html)

        else:
            if norm_data_mode == "生データ":
                d = df_larv[df_larv["ISOYear"] == int(base_year)].copy().dropna(subset=["Date"])
                d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
                d["Area"] = d["Area"].astype(str).str.strip()

                size_cols = [c for c in d.columns if str(c).isdigit()]
                for c in size_cols:
                    d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)

                cols_lt200   = [c for c in size_cols if int(c) < 200]
                cols_200_259 = [c for c in size_cols if 200 <= int(c) <= 259]
                cols_ge260   = [c for c in size_cols if int(c) >= 260]

                def row_sums(r):
                    lt200 = float(r[cols_lt200].sum()) if cols_lt200 else 0.0
                    mid   = float(r[cols_200_259].sum()) if cols_200_259 else 0.0
                    ge260 = float(r[cols_ge260].sum()) if cols_ge260 else 0.0
                    total = lt200 + mid + ge260
                    return lt200, mid, ge260, total

                totals = []
                for _, rr in d.iterrows():
                    *_, total = row_sums(rr)
                    if total > 0:
                        totals.append(total)
                t95 = float(np.nanpercentile(np.array(totals, dtype=float), 95)) if totals else 1.0
                t95 = max(t95, 1.0)
                MIN_S, MAX_S = 32, 78

                def size_from_total(total: float):
                    s = np.sqrt(max(float(total), 0.0)) / np.sqrt(t95) * MAX_S
                    return int(min(MAX_S, max(MIN_S, s)))

                for _, r in d.iterrows():
                    area = str(r["Area"]).strip()
                    dt = r["Date"]
                    if area == "" or pd.isna(dt):
                        continue
                    lt200, mid, ge260, total = row_sums(r)
                    if total <= 0:
                        continue
                    size = size_from_total(total)
                    center_txt = str(int(round(total)))  # ★常に表示
                    hover = f"{pd.Timestamp(dt):%Y-%m-%d}"
                    svg = svg_pie(
                        [lt200, mid, ge260], colors_larvae, size=size,
                        labels=["<200","200–259","≥260"], alpha=0.55,
                        center_text=center_txt, hover_text=hover, stroke_width=2,
                        emphasize_idxs=EMPH_LARV_IDX
                    )
                    if svg:
                        points.append((area, pd.Timestamp(dt), svg, hover))

            else:
                wk = _weekly_larvae_mean(df_larv)
                base = wk[wk["ISOYear"] == int(base_year)].copy()
                if base.empty:
                    st.info("週集計データがありません。")
                    return

                totals = base["total"].dropna().astype(float).tolist()
                t95 = float(np.nanpercentile(np.array(totals, dtype=float), 95)) if totals else 1.0
                t95 = max(t95, 1.0)
                MIN_S, MAX_S = 32, 78

                def size_from_total(total: float):
                    s = np.sqrt(max(float(total), 0.0)) / np.sqrt(t95) * MAX_S
                    return int(min(MAX_S, max(MIN_S, s)))

                for _, r in base.iterrows():
                    area = str(r["Area"]).strip()
                    w = int(r["week"])
                    lt200 = float(r["lt200"]); mid = float(r["mid"]); ge260 = float(r["ge260"])
                    total = float(r["total"])
                    if area == "" or (not np.isfinite(total)) or total <= 0:
                        continue
                    dt = pd.Timestamp(datetime.fromisocalendar(int(base_year), int(w), 1))
                    size = size_from_total(total)
                    center_txt = str(int(round(total)))  # ★常に表示（週平均の整数）
                    hover = f"{dt:%Y-%m-%d}"
                    svg = svg_pie(
                        [lt200, mid, ge260], colors_larvae, size=size,
                        labels=["<200","200–259","≥260"], alpha=0.55,
                        center_text=center_txt, hover_text=hover, stroke_width=2,
                        emphasize_idxs=EMPH_LARV_IDX
                    )
                    if svg:
                        points.append((area, dt, svg, hover))

            legend_html = """
            <div style="display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-top:6px;font-size:13px;">
              <div style="font-weight:700;">凡例（ラーバ）</div>
              <div style="color:#666;">注目：200–259 と ≥260 を強調（太枠+濃色）</div>
              <div><span style="display:inline-block;width:12px;height:12px;background:#d62728;margin-right:6px;border:1px solid #fff;"></span>≥260</div>
              <div><span style="display:inline-block;width:12px;height:12px;background:#ff7f0e;margin-right:6px;border:1px solid #fff;"></span>200–259</div>
              <div><span style="display:inline-block;width:12px;height:12px;background:#1f77b4;margin-right:6px;border:1px solid #fff;"></span>&lt;200</div>
            </div>
            """
            _render_xy(points, mondays, week_to_idx, norm_data_mode, legend_html)

    with tab_cmp:
        base_df = df_gsi if mode == "GSI" else df_larv
        weeks_all = sorted(
            base_df[base_df["ISOYear"] == int(base_year)]["week"].dropna().astype(int).unique().tolist()
        )
        if not weeks_all:
            st.info("基準年に週データがありません。")
            return

        mondays = [datetime.fromisocalendar(int(base_year), int(w), 1) for w in weeks_all]
        week_to_idx = {w: i for i, w in enumerate(weeks_all)}
        data_mode = "週集計"

        CMP_SIZE = 54
        points = []

        if mode == "GSI":
            agg = _weekly_gsi_sum(df_gsi)
            base = agg[agg["ISOYear"] == int(base_year)].copy()
            bmap = {(str(r["Area"]).strip(), int(r["week"])): r for _, r in base.iterrows()}

            cmap = {}
            if comp_years:
                comp_raw = agg[agg["ISOYear"].isin([int(y) for y in comp_years])].copy()
                comp = comp_raw.groupby(["Area","week"], as_index=False)[["ge25","mid","lt20","n"]].mean()
                cmap = {(str(r["Area"]).strip(), int(r["week"])): r for _, r in comp.iterrows()}

            for (area, w), br in bmap.items():
                if w not in week_to_idx:
                    continue
                dt = pd.Timestamp(datetime.fromisocalendar(int(base_year), int(w), 1))
                hover = f"{dt:%Y-%m-%d}"

                b_ge25 = float(br["ge25"]); b_mid = float(br["mid"]); b_lt20 = float(br["lt20"])

                cr = cmap.get((area, w), None)
                if cr is None:
                    c_ge25 = c_mid = c_lt20 = 0.0
                    draw_ring = False
                else:
                    c_ge25 = float(cr["ge25"]); c_mid = float(cr["mid"]); c_lt20 = float(cr["lt20"])
                    draw_ring = True

                svg = svg_ring_with_pie_core(
                    core_values=[b_ge25, b_mid, b_lt20],
                    ring_values=[c_ge25, c_mid, c_lt20],
                    colors=colors_gsi,
                    size=CMP_SIZE,
                    labels=["≥25","20–24.9","<20"],
                    core_alpha=0.92,
                    ring_alpha=0.25,
                    center_text=None,
                    hover_text=hover,
                    draw_ring=draw_ring,
                    core_stroke_width=3,
                    ring_stroke_width=2,
                    emphasize_core_idxs=EMPH_GSI_IDX,
                    emphasize_ring_idxs=EMPH_GSI_IDX
                )
                if svg:
                    points.append((area, dt, svg, hover))

            legend_html = """
            <div style="display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-top:6px;font-size:13px;">
              <div style="font-weight:700;">凡例（GSI）</div>
              <div style="color:#666;">基準=円 / 比較=外リング（注目：&lt;20 を強調）</div>
              <div><span style="display:inline-block;width:12px;height:12px;background:#d62728;margin-right:6px;border:1px solid #fff;"></span>≥25</div>
              <div><span style="display:inline-block;width:12px;height:12px;background:#ff7f0e;margin-right:6px;border:1px solid #fff;"></span>20–24.9</div>
              <div><span style="display:inline-block;width:12px;height:12px;background:#1f77b4;margin-right:6px;border:1px solid #fff;"></span>&lt;20</div>
            </div>
            """
            _render_xy(points, mondays, week_to_idx, data_mode, legend_html)

        else:
            wk = _weekly_larvae_mean(df_larv)
            base = wk[wk["ISOYear"] == int(base_year)].copy()
            bmap = {(str(r["Area"]).strip(), int(r["week"])): r for _, r in base.iterrows()}

            cmap = {}
            if comp_years:
                comp_raw = wk[wk["ISOYear"].isin([int(y) for y in comp_years])].copy()
                comp = comp_raw.groupby(["Area","week"], as_index=False)[["lt200","mid","ge260","total"]].mean()
                cmap = {(str(r["Area"]).strip(), int(r["week"])): r for _, r in comp.iterrows()}

            for (area, w), br in bmap.items():
                if w not in week_to_idx:
                    continue
                dt = pd.Timestamp(datetime.fromisocalendar(int(base_year), int(w), 1))
                hover = f"{dt:%Y-%m-%d}"

                b_lt200 = float(br["lt200"]); b_mid = float(br["mid"]); b_ge260 = float(br["ge260"])
                b_total = float(br["total"])
                if (not np.isfinite(b_total)) or b_total <= 0:
                    continue

                cr = cmap.get((area, w), None)
                if cr is None:
                    c_lt200 = c_mid = c_ge260 = 0.0
                    c_total = np.nan
                    draw_ring = False
                else:
                    c_lt200 = float(cr["lt200"]); c_mid = float(cr["mid"]); c_ge260 = float(cr["ge260"])
                    c_total = float(cr["total"])
                    draw_ring = True

                # ★差分（中心）は「基準と比較が揃う点だけ」
                if draw_ring and np.isfinite(c_total):
                    d_total = b_total - c_total
                    center_txt = f"{d_total:+.0f}"
                else:
                    center_txt = None

                svg = svg_ring_with_pie_core(
                    core_values=[b_lt200, b_mid, b_ge260],
                    ring_values=[c_lt200, c_mid, c_ge260],
                    colors=colors_larvae,
                    size=CMP_SIZE,
                    labels=["<200","200–259","≥260"],
                    core_alpha=0.92,
                    ring_alpha=0.25,
                    center_text=center_txt,
                    hover_text=hover,
                    draw_ring=draw_ring,
                    core_stroke_width=3,
                    ring_stroke_width=2,
                    emphasize_core_idxs=EMPH_LARV_IDX,
                    emphasize_ring_idxs=EMPH_LARV_IDX
                )
                if svg:
                    points.append((area, dt, svg, hover))

            legend_html = """
            <div style="display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-top:6px;font-size:13px;">
              <div style="font-weight:700;">凡例（ラーバ）</div>
              <div style="color:#666;">中心=Δtotal（基準−比較平均：両方ある点のみ） / 比較=外リング（注目：200–259, ≥260 を強調）</div>
              <div><span style="display:inline-block;width:12px;height:12px;background:#d62728;margin-right:6px;border:1px solid #fff;"></span>≥260</div>
              <div><span style="display:inline-block;width:12px;height:12px;background:#ff7f0e;margin-right:6px;border:1px solid #fff;"></span>200–259</div>
              <div><span style="display:inline-block;width:12px;height:12px;background:#1f77b4;margin-right:6px;border:1px solid #fff;"></span>&lt;200</div>
            </div>
            """
            _render_xy(points, mondays, week_to_idx, data_mode, legend_html)


def reset_sidebar_state_for(prefix_keep: str):
    import streamlit as st
    prefixes = ("map_", "sc_", "larv_", "yc_", "water_", "cal_", "cmem_")
    for k in list(st.session_state.keys()):
        if k.startswith(prefixes) and not k.startswith(prefix_keep):
            try:
                del st.session_state[k]
            except KeyError:
                pass


def main():
    import streamlit as st
    require_password_gate()
    #try:
    #    inject_compact_css()
    #except Exception:
    #    pass

    #st_html("""
    #<script>
    #const id = "trial-note-fixed";
    #if (!window.parent.document.getElementById(id)) {
    #  const div = window.parent.document.createElement("div");
    #  div.id = id;
    #  div.innerText = "※試験・関係者限定";
    #  div.style.position = "fixed";
    #  div.style.top = "6px";
    #  div.style.left = "10px";
    #  div.style.fontSize = "15px";
    #  div.style.color = "rgba(120,120,120,0.8)";
    #  div.style.zIndex = "999999";
    #  div.style.pointerEvents = "none";
    #  window.parent.document.body.appendChild(div);
    #}
    #</script>
    #""", height=0)


    OPTIONS = ["ガイダンス", "水温図", "CMEM", "テスト", "ラーバ", "経年比較"]

    if "main_mode_value" not in st.session_state:
        st.session_state["main_mode_value"] = "ガイダンス"

    try:
        mode = safe_segmented_control(
            "",
            options=OPTIONS,
            key="main_mode_seg",
            default=st.session_state["main_mode_value"],
            label_visibility="collapsed",
        )
    except Exception:
        idx = OPTIONS.index(st.session_state["main_mode_value"]) if st.session_state["main_mode_value"] in OPTIONS else 0
        mode = st.radio(
            "",
            options=OPTIONS,
            index=idx,
            horizontal=True,
            key="main_mode_radio",
            label_visibility="collapsed",
        )

    st.session_state["main_mode_value"] = mode

    sel_areas = None
    with st.sidebar:
        pass

    if mode == "水温図":
        reset_sidebar_state_for("water_")
        render_water_mode()
    elif mode == "CMEM":
        reset_sidebar_state_for("cmem_")
        render_cmem_mode()
    elif mode == "ラーバ":
        reset_sidebar_state_for("larv_")
        render_larvae_mode(sel_areas)
    elif mode == "経年比較":
        reset_sidebar_state_for("yc_")
        render_yearly_compare_mode()
    elif mode == "テスト":
        reset_sidebar_state_for("map_")
        render_map_mode()
    else:
        reset_sidebar_state_for("cal_")
        render_calendar_mode()

if __name__ == "__main__":
    main()