import os
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

# =========================
# 基本設定／パス
# =========================
st.set_page_config(page_title="デモ版", layout="centered")
ANCHOR_YEAR = 2000


DEFAULT_BASE_DIR ="data" 
base_dir = os.environ.get("APP_BASE_DIR", DEFAULT_BASE_DIR)

def pjoin(*parts: str) -> str:
    return os.path.normpath(os.path.join(*parts))


# CSVパス base_dir 直下のファイル
MATURITY_PATH = pjoin(base_dir, "maturity.csv")
LARVAE_PATH   = pjoin(base_dir, "larvae.csv")
NUM_PATH      = pjoin(base_dir, "collector_number.csv")
SIZE_PATH     = pjoin(base_dir, "collector_size.csv") 

TITLE_SIZE = 18
TEMP_MIN, TEMP_MAX = -2.0, 40.0

# =========================
# 共通ユーティリティ（時刻変換・読込・asof・Area系）
# =========================

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

@st.cache_data(show_spinner=False)
def utc_to_jst_naive(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    dt = dt.dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)
    return dt

@st.cache_data(show_spinner=False)
def jst_to_naive(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", utc=False)
    try:
        if getattr(dt.dt, "tz", None) is not None:
            dt = dt.dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)
    except Exception:
        pass
    return dt

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
def read_csv_path(path: str, try_encodings=("utf-8", "utf-8-sig", "cp932")):
    last_err = None
    for enc in try_encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            df.columns = [c.strip() for c in df.columns]
            return df
        except Exception as e:
            last_err = e
            continue
    st.error(f"CSV読み込みに失敗: {path}\n{last_err}")
    return None

def load_all_areas():
    areas = set()
    for path in [LARVAE_PATH, NUM_PATH, SIZE_PATH]:
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
    """
    depth_m ごとに asof マージ（datetime近傍）を安全に実施。
    """
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
    """
    深さ別の線形回帰（obs ≈ α + β*pred）を asof マージで学習。
    """
    dr_path  = pjoin(base_dir, "pred", train_filename)
    obs_path = pjoin(base_dir, "obs",  train_filename)
    if not (os.path.exists(dr_path) and os.path.exists(obs_path)):
        return None, None
    try:
        pred = pd.read_csv(dr_path)
        obs  = pd.read_csv(obs_path)
    except Exception as e:
        st.warning(f"補正用ファイルの読み込みに失敗: {e}")
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


# =========================
# GSI集計（重ね合わせ用）
# =========================
@st.cache_data(show_spinner=False)
def get_gsi_agg(selected_areas: List[str], years_sel: List[int]) \
        -> Tuple[Dict[str, Dict[int, Dict[str, pd.DataFrame]]], List[str]]:
    """
    エリア×年×Sexごとの MMDD 順 mean/std DataFrame。
    戻り値: (area_year_sex_dict, 全体MMDD順リスト)
    - area_year_sex_dict[area][year][sex] = DataFrame(columns=["MMDD","mean","std","sort"])
    """
    df = read_csv_path(MATURITY_PATH)
    if df is None:
        return {}, []

    # 前処理
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Year"] = df["Date"].dt.year
    df["MMDD"] = df["Date"].dt.strftime("%m-%d")
    if "GSI" in df.columns:
        df["GSI"] = pd.to_numeric(df["GSI"], errors="coerce")
    df = df.dropna(subset=["Date", "GSI"]).copy()

    if "Sex" not in df.columns:
        df["Sex"] = "Unknown"
    else:
        df["Sex"] = df["Sex"].fillna("Unknown").astype(str)

    base = f"{ANCHOR_YEAR}-"
    all_mmdd = sorted(
        df["MMDD"].unique(),
        key=lambda s: pd.to_datetime(base + s).day_of_year
    )

    out: Dict[str, Dict[int, Dict[str, pd.DataFrame]]] = {}
    for area in selected_areas:
        dfa = filter_by_areas(df, [area])
        if dfa.empty:
            continue
        out[area] = {}
        for y in years_sel:
            d = dfa[dfa["Year"] == y]
            if d.empty:
                continue
            out[area][y] = {}
            # ▼ Sex 別に MMDD 集計
            for sex, g in d.groupby("Sex"):
                agg = g.groupby("MMDD")["GSI"].agg(["mean", "std"]).reset_index()
                agg["sort"] = agg["MMDD"].apply(lambda s: pd.to_datetime(base + s).day_of_year)
                agg = agg.sort_values("sort")
                out[area][y][str(sex)] = agg
    return out, all_mmdd

# =========================
# 水温グラフ
# =========================
from typing import List
import os
import datetime as dt
import pandas as pd
import numpy as np
import streamlit as st
import re
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.express as px


def render_water_with_optional_gsi_overlay(selected_areas_for_gsi: List[str]):
    parent_folder_dr = pjoin(base_dir, "pred")
    if not os.path.exists(parent_folder_dr):
        st.error(f"フォルダが見つかりません: {parent_folder_dr}")
        st.stop()
    dr_files = list_dr_files_safe(base_dir)
    if not dr_files:
        st.warning("pred に CSV がありません")
        st.stop()

    # ======================
    # ユーティリティ
    # ======================
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

    # ======================
    # サイドバー
    # ======================
    with st.sidebar:
        selected_file = st.selectbox(
            "エリアを選択",
            sorted(dr_files),
            key="sb_selected_file"
        )

        df_dr_preview = load_dr_single_file(base_dir, selected_file)
        if df_dr_preview.empty:
            st.warning("データが読み込めませんでした")
            st.stop()

        years_dr = sorted(pd.to_datetime(df_dr_preview["datetime"]).dt.year.dropna().unique().tolist())
        df_gsi_pre = read_csv_path(MATURITY_PATH)
        years_gsi: list = []
        if df_gsi_pre is not None and "Date" in df_gsi_pre.columns:
            years_gsi = sorted(pd.to_datetime(df_gsi_pre["Date"]).dt.year.dropna().unique().tolist())
        years_all = sorted(set(years_dr) | set(years_gsi))
        latest_year = years_all[-1] if years_all else None

        # 期間（MM/DD）
        st.markdown("**期間指定（MM/DD）**")
        latest_dt_dr = pd.to_datetime(df_dr_preview["datetime"]).max()
        default_end_anchor = pd.Timestamp(f"{ANCHOR_YEAR}-{latest_dt_dr:%m-%d}")
        default_start_anchor = default_end_anchor - pd.Timedelta(days=29)
        start_mmdd = st.text_input("期間開始 (MM/DD)", value=f"{default_start_anchor:%m/%d}")
        end_mmdd   = st.text_input("期間終了 (MM/DD)", value=f"{default_end_anchor:%m/%d}")

        # 積算水温の起算（MM/DD）
        sekisan_mmdd = st.text_input("積算水温の開始 (MM/DD)", value="01/01")

        # 年選択（共通：水温/GSI）
        selected_years = st.multiselect(
            "表示年（水温/GSI）",
            years_all,
            default=[latest_year] if latest_year else [],
            key="main_years"
        )

        overlay_gsi   = st.checkbox("GSIを重ねる", value=False, key="sb_overlay_gsi")
        use_correction = st.checkbox("実測ベース補正(回帰)", value=False, key="sb_use_correction")
        show_sekisan   = st.checkbox("積算水温を表示", value=False, key="show_sekisan")

    # 入力検証（MM/DD）
    start_anchor_date    = parse_mmdd(start_mmdd)
    end_anchor_date      = parse_mmdd(end_mmdd)
    sekisan_anchor_date  = parse_mmdd(sekisan_mmdd)

    if start_anchor_date is None or end_anchor_date is None:
        st.warning("期間の月日は MM/DD 形式で入力してください（例：03/15）")
        st.stop()
    if show_sekisan and sekisan_anchor_date is None:
        st.warning("積算水温の開始は MM/DD 形式で入力してください（例：01/01）")
        st.stop()

    title_suffix = f"（{start_anchor_date:%m-%d}〜{end_anchor_date:%m-%d}）"

    # 固定・表示方針
    tolerance_min = 35
    show_obs_points = True
    only_depths_with_obs_when_correct = True

    # --- DR 読み込み（本処理用） ---
    df_dr = load_dr_single_file(base_dir, selected_file)
    if df_dr.empty:
        st.warning("データが読み込めませんでした")
        st.stop()
    df_dr["datetime"] = pd.to_datetime(df_dr["datetime"], errors="coerce")
    df_dr = df_dr.dropna(subset=["datetime"]).copy()
    df_dr["date_day"] = df_dr["datetime"].dt.date
    df_dr["year"]     = df_dr["datetime"].dt.year
    if "depth_m" in df_dr.columns:
        df_dr["depth_m"] = pd.to_numeric(df_dr["depth_m"], errors="coerce").round(0).astype("Int64")

    # 選択年フィルタ
    if selected_years:
        df_dr = df_dr[df_dr["year"].isin(selected_years)].copy()

    # 深度一覧
    depths_all = sorted(set(df_dr["depth_m"].dropna().astype(int).tolist())) if not df_dr.empty else []
    default_depths = depths_all[:min(1, len(depths_all))]
    selected_depths = st.multiselect("表示する水深", depths_all, default=default_depths, key="main_depths")

    # アンカーTimestamp
    start_anchor_ts   = pd.Timestamp(start_anchor_date)
    end_anchor_ts     = pd.Timestamp(end_anchor_date)
    sekisan_anchor_ts = pd.Timestamp(sekisan_anchor_date if sekisan_anchor_date else start_anchor_date)

    # OBS 読み込み（期間＋年フィルタ）
    parent_folder_obs = pjoin(base_dir, "obs")
    df_obs_period = pd.DataFrame()
    if show_obs_points:
        obs_path = pjoin(parent_folder_obs, selected_file)
        if os.path.exists(obs_path):
            try:
                df_obs = pd.read_csv(obs_path)
                df_obs["datetime"] = jst_to_naive(df_obs.get("Date"))
                df_obs["depth_m"]  = pd.to_numeric(df_obs.get("Depth"), errors="coerce").round(0).astype("Int64")
                df_obs = df_obs.rename(columns={"Temp": "obs_temp"})
                df_obs = df_obs.dropna(subset=["datetime", "depth_m"]).copy()
                df_obs["year"] = df_obs["datetime"].dt.year
                df_obs = df_obs[df_obs["year"].isin(selected_years)] if selected_years else df_obs
                mask_obs = mmdd_mask(df_obs["datetime"], start_anchor_ts, end_anchor_ts)
                df_obs_period = df_obs[mask_obs].copy()
            except Exception as e:
                st.warning(f"OBSの読み込みに失敗: {obs_path} ({e})")
                df_obs_period = pd.DataFrame()
        else:
            st.info("obs フォルダに同名CSVがありません。実測点は表示されません。")

    # 補正（選択年の最新日時を基準に直近30日）
    reg_depthwise, n_match_reg = None, None
    if use_correction:
        if df_dr.empty:
            st.warning("補正用のデータがありません")
        else:
            mask_period = mmdd_mask(df_dr["datetime"], start_anchor_ts, end_anchor_ts)
            df_dr_period = df_dr[mask_period].copy()
            if df_dr_period.empty:
                st.warning("選択期間内にデータがありません。最新時刻からの基準に戻ります。")
                period_end_max = pd.to_datetime(df_dr["datetime"]).max()
            else:
                period_end_max = pd.to_datetime(df_dr_period["datetime"]).max()
                
            train_end_dt   = period_end_max - pd.Timedelta(days=10)
            train_start_dt = train_end_dt - pd.Timedelta(days=30)

            data_min = pd.to_datetime(df_dr["datetime"]).min()
            if train_start_dt < data_min:
                train_start_dt = data_min
                
            with st.spinner(  
                f"回帰補正パラメータ算出中({selected_file}・{train_start_dt:%Y-%m-%d}〜{train_end_dt:%Y-%m-%d})..."
            ):

                reg_depthwise, n_match_reg = compute_depthwise_regression(
                    base_dir, selected_file, tolerance_min,
                    start_dt=train_start_dt, end_dt=train_end_dt, min_pairs=5
                )

    # GSI年選択
    gsi_years_sel = selected_years if overlay_gsi else []
    area_year_sex_dict, all_mmdd = ({}, [])
    if overlay_gsi:
        area_year_sex_dict, all_mmdd = get_gsi_agg(selected_areas_for_gsi, gsi_years_sel)
    sex_style = {
        "F": {"dash": "dash", "alpha_band": 0.18, "label": "F", "color": "#d62728"},
        "M": {"dash": "solid",  "alpha_band": 0.18, "label": "M", "color": "#1f77b4"},
        "Unknown": {"dash": "dot", "alpha_band": 0.18, "label": "Unknown", "color": "#7f7f7f"},
    }

    # カラーマップ
    base_colors = px.colors.qualitative.Dark24
    color_map = {}
    for i, d in enumerate(selected_depths):
        for idx_y, y in enumerate(selected_years):
            color_map[(int(d), int(y))] = base_colors[(i * len(selected_years) + idx_y) % len(base_colors)]
    year_color_map = {}
    for (d, y), c in color_map.items():
        if y not in year_color_map:
            year_color_map[y] = c

    dash_styles = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
        row_heights=[0.6, 0.4]
    )

    # 上段：水温（年別・深度別）＋補正＋積算水温
    merged_for_points = pd.DataFrame()
    if selected_depths and (not df_dr.empty):
        traces_added = 0
        for d in selected_depths:
            if use_correction and only_depths_with_obs_when_correct and not df_obs_period.empty:
                obs_depths = set(df_obs_period["depth_m"].dropna().astype(int).tolist())
                if int(d) not in obs_depths:
                    continue
            for idx_y, y in enumerate(selected_years if selected_years else sorted(df_dr["year"].unique().tolist())):
                df_dy = df_dr[(df_dr["depth_m"] == d) & (df_dr["year"] == y)].copy()
                if df_dy.empty:
                    continue
                if "pred_temp" in df_dy.columns and not df_dy.empty:
                    df_dy = df_dy.groupby(["depth_m", "datetime"], as_index=False).agg({"pred_temp": "median"})
                    df_dy = (
                        df_dy.sort_values("datetime")
                        .groupby("depth_m", group_keys=False)
                        .apply(lambda g: (
                            g.drop(columns=["depth_m"]).set_index("datetime")
                            .resample("1H").median(numeric_only=True)
                            .interpolate(method="time", limit=2)
                            .reset_index()
                            .assign(depth_m=int(g["depth_m"].iloc[0]))
                        ))
                    )
                # 月日フィルタ
                mask_dy = mmdd_mask(df_dy["datetime"], start_anchor_ts, end_anchor_ts)
                df_dy = df_dy[mask_dy].copy()
                if df_dy.empty:
                    continue
                # アンカーx・hover
                df_dy["anchored_dt"] = to_anchor_ts(df_dy["datetime"])
                custom_hover = df_dy["datetime"].dt.strftime("%m-%d %H:%M")
                line_color = color_map[(int(d), int(y))]
                dash = dash_styles[idx_y % len(dash_styles)]
                y_raw = df_dy["pred_temp"].astype(float)

                # 予測
                pred_line_width  = 1 if use_correction else 2
                pred_line_opacity = 0.6 if use_correction else 1.0
                fig.add_trace(go.Scatter(
                    x=df_dy["anchored_dt"], y=y_raw, mode="lines",
                    name=f"{d}m 予測 {y}",
                    line=dict(color=line_color, width=pred_line_width, dash=dash),
                    opacity=pred_line_opacity,
                    customdata=custom_hover,
                    hovertemplate="予測: %{y:.2f} ℃<extra></extra>",
                    legendgroup=f"{d}-{y}"
                ), row=1, col=1, secondary_y=False)
                traces_added += 1

                # 実測
                if not df_obs_period.empty:
                    df_obs_dy = df_obs_period[(df_obs_period["depth_m"] == d) & (df_obs_period["year"] == y)].copy()
                    if not df_obs_dy.empty:
                        df_obs_dy = df_obs_dy.sort_values("datetime")
                        obs_color = color_map.get((int(d), int(y)), "#666666")
                        fig.add_trace(go.Scatter(
                            x=pd.to_datetime(df_obs_dy["datetime"].dt.strftime(f"{ANCHOR_YEAR}-%m-%d %H:%M:%S")),
                            y=df_obs_dy["obs_temp"], mode="markers",
                            name=f"{d}m 実測水温 {y}",
                            marker=dict(size=6, color=obs_color, symbol="circle", line=dict(color="#9e9e9e", width=0.6)),
                            opacity=0.70,
                            customdata=custom_hover,
                            hovertemplate="実測: %{y:.2f} ℃<extra></extra>",
                            legendgroup=f"{d}-{y}"
                        ), row=1, col=1, secondary_y=False)
                           
                # 補正水温
                if use_correction and (reg_depthwise is not None) and (int(d) in (reg_depthwise or {})):
                    alpha, beta = reg_depthwise[int(d)]
                    y_corr = np.clip(alpha + beta * y_raw.astype(float), TEMP_MIN, TEMP_MAX)
                    fig.add_trace(go.Scatter(
                        x=df_dy["anchored_dt"], y=y_corr, mode="lines",
                        name=f"{d}m 補正水温 {y}",
                        line=dict(color=line_color, width=3, dash="solid"),
                        customdata=custom_hover,
                        hovertemplate="補正: %{y:.2f} ℃<extra></extra>",
                        legendgroup=f"{d}-{y}"
                    ), row=1, col=1, secondary_y=False)
                    

                # 積算水温
                if show_sekisan:
                    df_dy_full = df_dr[(df_dr["depth_m"] == d) & (df_dr["year"] == y)].copy()
                    if "pred_temp" in df_dy_full.columns and not df_dy_full.empty:
                        df_dy_full = (
                            df_dy_full.sort_values("datetime")
                            .groupby("depth_m", group_keys=False)
                            .apply(lambda g: (
                                g.drop(columns=["depth_m"]).set_index("datetime")
                                 .resample("1H").median(numeric_only=True)
                                 .interpolate(method="time", limit=2)
                                 .reset_index()
                                 .assign(depth_m=int(g["depth_m"].iloc[0]))
                            ))
                        )
                        # ② 年内（日平均）を作成
                        df_daily_full = (
                            df_dy_full.assign(date_day=pd.to_datetime(df_dy_full["datetime"]).dt.date)
                            .groupby("date_day")["pred_temp"].mean().reset_index()
                            .sort_values("date_day")
                        )
                        df_daily_full["dt"] = pd.to_datetime(df_daily_full["date_day"])
                        # ③ 積算の“計算”は「起算～期間終了」で実施
                        mask_calc = mmdd_mask(df_daily_full["dt"], sekisan_anchor_ts, end_anchor_ts)
                        df_daily_calc = df_daily_full[mask_calc].copy()
                        if not df_daily_calc.empty:
                            # ④ “表示”は「期間開始～期間終了」でスライス（値は起算からの通算）
                            mask_show = mmdd_mask(df_daily_calc["dt"], start_anchor_ts, end_anchor_ts)
                            df_daily_show = df_daily_calc[mask_show].copy()
                            if not df_daily_show.empty:
                                x_sekisan = df_daily_show["dt"].map(lambda d0: pd.Timestamp(f"{ANCHOR_YEAR}-{d0:%m-%d}"))
                                # 予測積算
                                y_pred_accum = df_daily_calc["pred_temp"].cumsum()
                                # 表示区間に合わせた値だけ抽出（dtで join）
                                y_pred_accum_show = y_pred_accum.loc[df_daily_calc.index].reindex(df_daily_show.index).values
                                fig.add_trace(go.Scatter(
                                    x=x_sekisan, y=y_pred_accum_show, mode="lines",
                                    name=f"{d}m 積算水温（予測） {y}",
                                    line=dict(color=line_color, width=2, dash="dot"),
                                    opacity=0.70
                                ), row=1, col=1, secondary_y=False)
                                # 補正積算：起算から通算→表示区間へ
                                if use_correction and (reg_depthwise is not None) and (int(d) in reg_depthwise):
                                    alpha, beta = reg_depthwise[int(d)]
                                    y_corr_daily_calc = np.clip(alpha + beta * df_daily_calc["pred_temp"].astype(float),
                                                                TEMP_MIN, TEMP_MAX)
                                    y_corr_accum = y_corr_daily_calc.cumsum()
                                    y_corr_accum_show = y_corr_accum.loc[df_daily_calc.index].reindex(df_daily_show.index).values
                                    fig.add_trace(go.Scatter(
                                        x=x_sekisan, y=y_corr_accum_show, mode="lines",
                                        name=f"{d}m 積算水温（補正） {y}",
                                        line=dict(color=line_color, width=3, dash="dot"),
                                        opacity=1.0
                                    ), row=1, col=1, secondary_y=False)


                # 実測マージ用に保持
                merged_for_points = pd.concat(
                    [merged_for_points, df_dy[["datetime", "anchored_dt", "depth_m"]].copy()], axis=0
                )

    # GSI重ね合わせ（右軸）：平均±1σ帯＋平均線
    if overlay_gsi and area_year_sex_dict:
        def mmdd_to_anchor(mmdd: str) -> pd.Timestamp:
            return pd.to_datetime(f"{ANCHOR_YEAR}-{mmdd}")

        for area, by_year in area_year_sex_dict.items():
            for y, by_sex in by_year.items():
                if not by_sex:
                    continue
                base_color_hex = year_color_map.get(int(y), "#1f77b4")
                h = base_color_hex.lstrip('#')
                r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))

                for sex, agg in by_sex.items():
                    if agg is None or agg.empty:
                        continue
                
                    x_dt_full = agg["MMDD"].apply(mmdd_to_anchor)
                    mask_gsi = (mmdd_mask(x_dt_full, start_anchor_ts, end_anchor_ts))
                    x_dt = x_dt_full[mask_gsi]
                    agg_r = agg[mask_gsi].copy()
                    if agg_r.empty:
                        continue
                    lower = agg_r["mean"] - agg_r["std"].fillna(0.0)
                    upper = agg_r["mean"] + agg_r["std"].fillna(0.0)

                    style = sex_style.get(sex, sex_style["Unknown"])
                    fill_alpha = style["alpha_band"]
                    dash = style["dash"]
                    sex_lab = style["label"]


                    fig.add_trace(go.Scatter(
                        x=x_dt, y=lower, mode="lines", line=dict(width=0), hoverinfo="skip",
                        showlegend=False
                    ), row=1, col=1, secondary_y=True)
                    fig.add_trace(go.Scatter(
                        x=x_dt, y=upper, mode="lines", line=dict(width=0), fill="tonexty", hoverinfo="skip",
                        fillcolor=f"rgba({r},{g},{b},0.15)",  # 年度色の薄色
                        showlegend=False
                    ), row=1, col=1, secondary_y=True)
                    # 平均線（年度色）
                    fig.add_trace(go.Scatter(
                        x=x_dt, y=agg_r["mean"], mode="lines",
                        name=f"{area}-{y} GSI平均({sex_lab})",
                        line=dict(color=base_color_hex, width=2, dash=dash),
                        customdata=agg_r["MMDD"],
                        hovertemplate="%{customdata}<br>GSI平均: %{y:.2f}<extra></extra>",                      
                        legendgroup=f"GSI-{area}-{y}-{sex_lab}",
                    ), row=1, col=1, secondary_y=True)

    # レイアウト
    show_legend = st.checkbox("凡例", value=True, key="main_show_legend")
    legend_cfg = dict(orientation="h", yanchor="top", y=1.02, xanchor="right", x=1,
                      font=dict(size=12), itemsizing="constant")
    fig.update_layout(
        title={"text": f"{selected_file} 水温 {title_suffix}", "y": 0.98, "x": 0.01,
               "xanchor": "left", "font": {"size": 16}, "pad": {"t": 8}},
        margin=dict(l=10, r=10, t=50, b=10),
        height=700, template="plotly_white",
        showlegend=bool(show_legend), legend=legend_cfg if show_legend else dict()
    )
    fig.update_layout(hovermode="x unified")

    # X軸
    total_days = anchored_day_span(start_anchor_ts, end_anchor_ts)
    if total_days <= 14:
        dtick = "D1"
    elif total_days <= 60:
        dtick = "D7"
    elif total_days <= 180:
        dtick = "M1"
    else:
        dtick = "M2"

    tick0 = None
    if dtick == "D7":
        first_anchor = start_anchor_ts
        offset_days = (0 - first_anchor.weekday()) % 7
        tick0 = first_anchor + pd.Timedelta(days=offset_days)

    y_start = pd.Timestamp(f"{ANCHOR_YEAR}-01-01")
    y_end = pd.Timestamp(f"{ANCHOR_YEAR}-12-31") + pd.Timedelta(days=1)
    if start_anchor_ts <= end_anchor_ts:
        x_range = [start_anchor_ts, end_anchor_ts + pd.Timedelta(days=1)]
    else:
        x_range = [y_start, y_end]

    fig.update_xaxes(
        type="date",
        range=x_range,
        tickformat="%m-%d",
        dtick=dtick,
        tick0=tick0 if tick0 is not None else None,
        showticklabels=True,
        ticks="outside",
        showline=True,
        mirror=True,
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
        hoverformat="%m-%d %H:%M",
        title_text="月日(JST)",
        row=1, col=1
    )
    fig.update_xaxes(
        tickformat="%m-%d",
        dtick=dtick,
        tick0=tick0 if tick0 is not None else None,
        showticklabels=True,
        ticks="outside",
        showline=True,
        mirror=True,
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
        hoverformat="%m-%d %H:%M",
        row=2, col=1
    )

    fig.update_yaxes(title_text="水温 (℃)", secondary_y=False, tickfont=dict(size=11), row=1, col=1)
    if overlay_gsi:
        fig.update_yaxes(title_text="GSI", secondary_y=True, tickfont=dict(size=11), row=1, col=1)
    else:
        fig.update_yaxes(title_text="任意列(右軸)", secondary_y=True, tickfont=dict(size=11), row=1, col=1)

    st.plotly_chart(fig, use_container_width=True)

# =========================
# ラーバ
# =========================
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
        st.sidebar.caption(f"※日数が多いので最初の {max_days} 日のみ表示。期間や最大日数を調整してください。")
        days_all = days_all[:max_days]

    years_to_show = sorted(q["Year"].unique().tolist(), reverse=True)
    if not years_to_show:
        st.info("選択年のデータがありません。")
        return

    # 帯定義（20 μm刻み）
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

    # サブプロット構成
    n_days = len(days_all)
    titles = []
    for _ in years_to_show:
        titles += [""] + [mmdd_to_md(md) for md in days_all]

    fig = make_subplots(
        rows=len(years_to_show), cols=n_days + 1,
        shared_yaxes=False, shared_xaxes=True,
        horizontal_spacing=0.02, vertical_spacing=0.08,
        subplot_titles=titles,
        column_widths=[0.06] + [(1.0 - 0.06) / max(1, n_days)] * n_days
    )

    # 年列の軸は非表示
    for r, _ in enumerate(years_to_show, start=1):
        fig.update_xaxes(visible=False, row=r, col=1)
        fig.update_yaxes(visible=False, row=r, col=1)

    # 本体：横棒（3分類色）
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
            text=f"{yr}年", xref="paper", yref="paper", x=-0.12, y=y_paper_mid,
            showarrow=False, align="center", yanchor="middle",
            font=dict(size=12, color="#222"), textangle=-90
        )

    fig.update_layout(
        xaxis_title="", yaxis_title="サイズ（μm）",
        plot_bgcolor="white", paper_bgcolor="white",
        height=max(260, 240 * len(years_to_show)),
        margin=dict(l=110, r=10, t=60, b=10),
        font=dict(size=13, color="#222"),
        legend=dict(orientation="h", y=-0.12)
    )
    st.plotly_chart(fig, use_container_width=True)


def render_larvae_mode(selected_areas: List[str]):
    import plotly.graph_objects as go
    from datetime import date

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

    with st.sidebar:
        years_all = sorted(df["Year"].dropna().unique().tolist())
        latest    = years_all[-1] if years_all else None
        years_sel = st.multiselect("表示年", years_all, default=[latest] if latest else [])

        # 採苗数モードと同じレンジ・既定（3/1〜7/31）
        min_md    = date(ANCHOR_YEAR, 3, 1)
        max_md    = date(ANCHOR_YEAR, 7, 31)
        def_start = date(ANCHOR_YEAR, 3, 1)
        def_end   = date(ANCHOR_YEAR, 7, 31)
        sel_md_start, sel_md_end = st.slider(
            "対象期間（MM-DD）", min_value=min_md, max_value=max_md,
            value=(def_start, def_end), format="MM-DD"
        )

        mode_b    = st.radio("グラフ種類", ["横棒（日別／実値）", "縦棒（期間／比率）"], index=0, horizontal=False)
        max_days  = st.slider("横棒の最大表示（日数）", min_value=5, max_value=40, value=5)

    # スライダー入力を day-of-year へ
    def to_doy(d: date) -> int:
        return pd.Timestamp(d).day_of_year

    s_doy = to_doy(sel_md_start)
    e_doy = to_doy(sel_md_end)

    def in_window(md_doy: int, s: int, e: int) -> bool:
        return (s <= e and s <= md_doy <= e) or (s > e and (md_doy >= s or md_doy <= e))

    if not size_ints:
        st.info("サイズ列が見つかりません。")
        return

    first_bin = size_ints[0]
    def bin_low(s: int) -> int:
        return first_bin + ((s - first_bin) // 20) * 20

    band_labels = sorted({f"{bin_low(si)}-{bin_low(si)+20}" for si in size_ints},
                        key=lambda t: int(t.split("-")[0]))

    category_colors = {
        "<200":    "#1f77b4",  # blue
        "200-259": "#ff7f0e",  # orange
        ">=260":   "#d62728",  # red
    }
    def band_to_category(band_label: str) -> str:
        try:
            low = int(band_label.split("-")[0])
        except Exception:
            return "<200"
        if low < 200:            return "<200"
        elif 200 <= low <= 259:  return "200-259"
        else:                    return ">=260"

    auto_max_global = 0.0
    for area in selected_areas:
        df_area = filter_by_areas(df, [area])
        q_test  = df_area.copy()
        if years_sel:
            q_test = q_test[q_test["Year"].isin(years_sel)]
        q_test = q_test[q_test["md_doy"].apply(lambda d: in_window(d, s_doy, e_doy))]
        if q_test.empty:
            continue
        for md in q_test["MMDD"].astype(str).unique():
            gmd = q_test[q_test["MMDD"].astype(str) == md]
            if gmd.empty:
                continue
            bins_sum = {}
            for si in size_ints:
                b = bin_low(si)
                bins_sum[b] = bins_sum.get(b, 0.0) + gmd[str(si)].sum()
            local_max = max(bins_sum.values()) if bins_sum else 0.0
            auto_max_global = max(auto_max_global, float(local_max))

    with st.sidebar:
        x_max_global = st.slider(
            "横棒のX軸（最大値｜ラーバ全体）",
            min_value=0.0, max_value=max(1.0, auto_max_global * 1.5),
            value=auto_max_global, step=0.1
        )

    tables_to_show: list[tuple[str, pd.DataFrame]] = []

    for i, area in enumerate(selected_areas):

        df_area = filter_by_areas(df, [area])

        q_area = df_area.copy()
        if years_sel:
            q_area = q_area[q_area["Year"].isin(years_sel)]
        q_area = q_area[q_area["md_doy"].apply(lambda d: in_window(d, s_doy, e_doy))]

        if q_area.empty or not size_cols:
            st.info("選択条件に該当するデータがありません。")
            if i < len(selected_areas) - 1:
                st.markdown("---")  
            continue

        # ===== 縦棒（期間／比率）=====
        if mode_b == "縦棒（期間／比率）":
            rows = []
            for yr, g in q_area.groupby("Year"):
                total = g[size_cols].sum().sum()
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
                years_sorted = sorted(bars_df["Year"].unique())  # 古い→新しい

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

        # ===== 横棒（日別／実値）=====
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

        # ✅ グラフ下のキャプションは残す
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

        priority = {"200μm未満": 0, "200-259μm": 1, "260μm以上": 2}
        summary_df = pd.DataFrame(summary_rows, columns=["サイズ", "年", "合計", "割合"])
        if not summary_df.empty:
            summary_df["__order"] = summary_df["サイズ"].map(priority)
            summary_df = summary_df.sort_values(["__order", "年"]).drop(columns="__order")
            tables_to_show.append((area, summary_df))

        if i < len(selected_areas) - 1:
            st.markdown("---")

    if tables_to_show:
        place_bg = {"200μm未満": "#e6f3ff", "200-259μm": "#fff3e0", "260μm以上": "#ffe6e6"}

        def color_by_size(row: pd.Series):
            bg = place_bg.get(row.get("サイズ"), "")
            return [f"background-color: {bg}" if bg else "" for _ in row]

        for area, summary_df in tables_to_show:
            with st.expander(f"Area: {area}", expanded=False):  # ← タイトルは「Area: xxx」を維持
                styled = (
                    summary_df.style
                    .apply(color_by_size, axis=1)
                    .set_properties(**{"border-color": "#ddd"})
                    .format({"合計": "{:.1f}", "割合": "{:.1f}"})
                )
                st.dataframe(styled, use_container_width=True)

        if i < len(selected_areas) - 1:
            st.markdown("---")

# =========================
# 採苗数モード
# =========================
def render_scallop_mode(selected_areas: List[str]):
    df_num  = read_csv_path(NUM_PATH)
    df_size = read_csv_path(SIZE_PATH)
    if (df_num is None) or (df_size is None):
        st.warning("CSVが読み込めませんでした。")
        st.stop()

    def normalize_place_col(df_):
        if df_ is None or df_.empty:
            return df_
        if "Place" in df_.columns:
            df_["Place"] = (
                df_["Place"]
                .fillna("unknown")
                .astype(str)
                .str.strip()
                .str.lower()
            )
        return df_

    df_num  = normalize_place_col(df_num)
    df_size = normalize_place_col(df_size)

    for d in (df_num, df_size):
        for col in ["Drop_Date", "Monitoring_Date"]:
            if col in d.columns:
                d[col] = pd.to_datetime(d[col], errors="coerce")
    for c in ["Scallop", "Mussel", "Akazara", "Other"]:
        if c in df_num.columns:
            df_num[c] = pd.to_numeric(df_num[c], errors="coerce")
    for c in ["No", "Shell(mm)"]:
        if c in df_size.columns:
            df_size[c] = pd.to_numeric(df_size[c], errors="coerce")

    with st.sidebar:
        # 年候補（Drop_Date基準）
        if df_num is None or df_num.empty or ("Drop_Date" not in df_num.columns):
            years_all = []
            latest    = None
        else:
            years_all = sorted(
                pd.to_datetime(df_num["Drop_Date"], errors="coerce")
                .dt.year.dropna().unique().tolist()
            )
            latest    = years_all[-1] if years_all else None

        df_num_place  = normalize_place_col(read_csv_path(NUM_PATH))
        df_size_place = normalize_place_col(read_csv_path(SIZE_PATH))
        places_all = set()
        if df_num_place is not None and ("Place" in df_num_place.columns):
            places_all.update(df_num_place["Place"].dropna().unique().tolist())
        if df_size_place is not None and ("Place" in df_size_place.columns):
            places_all.update(df_size_place["Place"].dropna().unique().tolist())
        preferred_order = ["unknown", "lower", "middle", "upper"]
        places_all = sorted(
            places_all,
            key=lambda x: (
                preferred_order.index(x) if x in preferred_order else len(preferred_order),
                x,
            ),
        )

        sel_years    = st.multiselect("表示年", options=years_all, default=[latest] if latest else [])
        min_md       = date(ANCHOR_YEAR, 3, 1)
        max_md       = date(ANCHOR_YEAR, 7, 31)
        def_start    = date(ANCHOR_YEAR, 3, 1)
        def_end      = date(ANCHOR_YEAR, 7, 31)
        sel_md_start, sel_md_end = st.slider(
            "対象期間", min_value=min_md, max_value=max_md,
            value=(def_start, def_end), format="MM-DD"
        )
        sel_places   = st.multiselect("Place（フィルタ）", options=places_all, default=["upper","middle","lower"])
        place_stack  = st.checkbox("グラフを Place 別に積み上げる", value=True)

    def apply_filters(df, sel_years, sel_md_start=None, sel_md_end=None):
        if df is None or len(df) == 0:
            return df
        if "Drop_Date" in df.columns and sel_years:
            yy = pd.to_datetime(df["Drop_Date"], errors="coerce").dt.year
            df = df[yy.isin(sel_years)]
        if ("Drop_Date" in df.columns) and (sel_md_start is not None) and (sel_md_end is not None):
            anchored = anchored_md_series(df["Drop_Date"])
            s, e = sel_md_start, sel_md_end
            if s <= e:
                df = df[(anchored >= s) & (anchored <= e)]
            else:
                df = df[(anchored >= s) | (anchored <  e)]
        return df

    tier_color  = {"upper":"#1f77b4", "middle":"#ff7f0e", "lower":"#2ca02c", "unknown":"#7f7f7f"}
    place_bg    = {"upper": "#e6f3ff", "middle": "#fff3e0", "lower": "#ffe6e6", "unknown": "#f0f0f0"}
    stack_order = ["unknown","lower","middle","upper"]

    for i, area in enumerate(selected_areas):
        df_num_f0  = filter_by_areas(df_num.copy(),  [area])
        df_size_f0 = filter_by_areas(df_size.copy(), [area])

        if sel_places:
            if ("Place" in df_num_f0.columns):
                df_num_f0  = df_num_f0[df_num_f0["Place"].isin(sel_places)]
            if ("Place" in df_size_f0.columns):
                df_size_f0 = df_size_f0[df_size_f0["Place"].isin(sel_places)]

        df_num_f  = apply_filters(df_num_f0.copy(),  sel_years, sel_md_start, sel_md_end)
        df_size_f = apply_filters(df_size_f0.copy(), sel_years, sel_md_start, sel_md_end)
        if df_num_f.empty:
            st.info("選択条件に該当する number データがありません。")
            if i < len(selected_areas) - 1:
                st.markdown("---")
            continue

        n_for_max = df_num_f.dropna(subset=["Drop_Date", "Monitoring_Date"]).copy()
        n_for_max["Drop_Date_dt"]       = pd.to_datetime(n_for_max["Drop_Date"], errors="coerce")
        n_for_max["Monitoring_Date_dt"] = pd.to_datetime(n_for_max["Monitoring_Date"], errors="coerce")

        s_for_max = df_size_f.dropna(subset=["Drop_Date","Monitoring_Date","Shell(mm)"]).copy()
        s_for_max["Drop_Date_dt"]       = pd.to_datetime(s_for_max["Drop_Date"], errors="coerce")
        s_for_max["Monitoring_Date_dt"] = pd.to_datetime(s_for_max["Monitoring_Date"], errors="coerce")

        x2_global_max = float(
            n_for_max.groupby(["Drop_Date_dt","Monitoring_Date_dt"])["Scallop"].sum().max()
        ) if ("Scallop" in n_for_max.columns and not n_for_max.empty) else 0.0
        x1_global_max = float(
            s_for_max.groupby(["Drop_Date_dt","Monitoring_Date_dt"])["Shell(mm)"].mean().max()
        ) if not s_for_max.empty else 0.0
        qty_range_max  = max(1.0, x2_global_max * 1.10)
        size_range_max = max(1.0, x1_global_max * 1.10)

        with st.expander(f"投入日別｜Area: {area}", expanded=True):
            drops_all = sorted(
                pd.to_datetime(pd.concat([n_for_max["Drop_Date_dt"], s_for_max["Drop_Date_dt"]], ignore_index=True).dropna()).unique(),
                reverse=True
            )
            if not drops_all:
                st.info("Drop_Date がありません。")
                continue

            first_figure = True
            for d_drop in drops_all:
                n_g = n_for_max[n_for_max["Drop_Date_dt"] == d_drop].copy()
                s_g = s_for_max[s_for_max["Drop_Date_dt"] == d_drop].copy()

                drop_ts = pd.to_datetime(d_drop).normalize()
                dd_str  = drop_ts.strftime('%Y-%m-%d')

                md_all = pd.concat([n_g["Monitoring_Date_dt"], s_g["Monitoring_Date_dt"]], ignore_index=True).dropna()
                md_all_ts = pd.to_datetime(md_all).dt.normalize()     # ← Series なので dt.normalize()
                md_all_ts = md_all_ts.drop_duplicates().sort_values()
                md_index  = pd.Index(md_all_ts)                       # reindex 等で明示的に Index 化

                md_lab0   = [ts.strftime('%m-%d') for ts in md_index]
                md_days   = [(ts - drop_ts).days for ts in md_index]
                md_labels = [f"{lbl}（{dy}日）" for lbl, dy in zip(md_lab0, md_days)]

                y_labels = md_labels
                md_to_label = dict(zip(md_index, md_labels))

                if not n_g.empty:
                    n_g["Mon_norm"] = pd.to_datetime(n_g["Monitoring_Date_dt"]).dt.normalize()
                if not s_g.empty:
                    s_g["Mon_norm"] = pd.to_datetime(s_g["Monitoring_Date_dt"]).dt.normalize()

                fig = go.Figure()

                if ("Scallop" in n_g.columns) and (len(n_g) > 0):
                    if place_stack and ("Place" in n_g.columns):
                        n_agg = (
                            n_g.groupby(["Mon_norm","Place"])["Scallop"]
                            .sum()
                            .reset_index()
                        )
                        for place_key in stack_order:
                            g2 = n_agg[n_agg["Place"] == place_key]
                            if g2.empty:
                                continue
                            s_vals = g2.set_index("Mon_norm")["Scallop"]
                            x_vals = [float(v) for v in s_vals.reindex(md_index, fill_value=0.0).values]
                            fig.add_trace(go.Bar(
                                x=x_vals,
                                y=y_labels,
                                orientation="h",
                                name=str(place_key),
                                legendgroup="qty",
                                marker=dict(
                                    color=tier_color.get(place_key, "#7f7f7f"),
                                    line=dict(color="#000", width=1)
                                ),
                                width=0.95,
                                opacity=0.6,
                                hovertemplate=(
                                    f"投入日: {dd_str}<br>"
                                    "日付: %{y}<br>"
                                    "数量: %{x:.0f}<br>"
                                    f"Place: {place_key}"
                                ),
                                xaxis="x2",
                                showlegend=first_figure
                            ))
                        fig.update_layout(barmode="stack", bargap=0.05, bargroupgap=0.0)
                    else:
                        n_agg = (
                            n_g.groupby(["Mon_norm"])["Scallop"]
                            .sum()
                            .reindex(md_index, fill_value=0.0)
                        )
                        x_vals = [float(v) for v in n_agg.values]
                        fig.add_trace(go.Bar(
                            x=x_vals,
                            y=y_labels,
                            orientation="h",
                            name="Scallop",
                            legendgroup="qty",
                            marker=dict(color="#1f77b4", line=dict(color="#000", width=1)),
                            width=0.95,
                            opacity=0.6,
                            hovertemplate=(
                                f"投入日: {dd_str}<br>"
                                "日付: %{y}<br>"
                                "数量: %{x:.0f}"
                            ),
                            xaxis="x2",
                            showlegend=first_figure
                        ))

                if len(s_g) > 0:
                    if place_stack and ("Place" in s_g.columns):
                        for place_key in stack_order:
                            s_place = s_g[s_g["Place"] == place_key]
                            if s_place.empty:
                                continue
                            s_mean = s_place.groupby("Mon_norm")["Shell(mm)"].mean()
                            x_for_line = [float(v) if pd.notna(v) else None
                                        for v in s_mean.reindex(md_index).values]
                            y_for_line = y_labels
                            fig.add_trace(go.Scatter(
                                x=x_for_line,
                                y=y_for_line,
                                mode="lines+markers",
                                name=f"Size_mean({place_key})",
                                legendgroup="size",
                                line=dict(color=tier_color.get(place_key, "#7f7f7f"), width=3),
                                marker=dict(color=tier_color.get(place_key, "#7f7f7f"), size=6),
                                hovertemplate=("サイズ: %{x:.1f} mm<br>日付: %{y}"),
                                xaxis="x",
                                cliponaxis=False,
                                showlegend=first_figure
                            ))
                    else:
                        s_mean = s_g.groupby("Mon_norm")["Shell(mm)"].mean()
                        x_for_line = [float(v) if pd.notna(v) else None
                                    for v in s_mean.reindex(md_index).values]
                        y_for_line = y_labels
                        fig.add_trace(go.Scatter(
                            x=x_for_line,
                            y=y_for_line,
                            mode="lines+markers",
                            name="Size_mean(mm)",
                            legendgroup="size",
                            line=dict(color="#d62728", width=3),
                            marker=dict(color="#d62728", size=6),
                            hovertemplate=("サイズ: %{x:.1f} mm<br>日付: %{y}"),
                            xaxis="x",
                            cliponaxis=False,
                            showlegend=first_figure
                        ))

                fig.update_layout(
                    plot_bgcolor="white", paper_bgcolor="white",
                    margin=dict(l=10, r=44, t=52, b=64),
                    font=dict(size=13, color="#222"),
                    legend=dict(orientation="h", y=-0.18, x=0.01, font=dict(size=11)),
                    showlegend=True,
                    title=dict(text=f"投入日: {dd_str}", font=dict(size=16), x=0.01),
                    hovermode="closest", legend_traceorder="normal"
                )
                fig.update_xaxes(
                    title="サイズ（mm）", range=[0, size_range_max],
                    showline=True, linecolor="#000", linewidth=1,
                    showgrid=False, zeroline=False, automargin=True, domain=[0.0, 0.985]
                )
                fig.update_layout(
                    xaxis2=dict(
                        title="付着数合計", range=[0, qty_range_max],
                        overlaying="x", side="top",
                        showline=True, linecolor="#000", linewidth=1,
                        showgrid=True, gridcolor="rgba(0,0,0,0.06)",
                        zeroline=False, automargin=True, domain=[0.0, 0.985]
                    )
                )
                fig.update_yaxes(
                    title="", type="category",
                    categoryorder="array", categoryarray=y_labels,
                    showline=True, linecolor="#000", linewidth=1,
                    gridcolor="rgba(0,0,0,0.06)", automargin=True
                )


                fig.update_layout(height=max(300, int(42 * len(y_labels) + 120)))

                st.plotly_chart(fig, use_container_width=True)
                first_figure = False

        # ------------- 表（Place背景色：常に生成、折りたたみ） -------------
        with st.expander("採苗数データ表（クリックで展開）", expanded=False):
            common_keys = ["Drop_Date", "Monitoring_Date"]
            if ("Place" in df_num_f.columns) and ("Place" in df_size_f.columns):
                common_keys = ["Place"] + common_keys
            common_keys = list(dict.fromkeys(common_keys))

            size_mean = (
                df_size_f.groupby(common_keys, as_index=False)["Shell(mm)"]
                .mean()
                .rename(columns={"Shell(mm)": "Size_mean(mm)"})
            )

            num_agg_df = (
                df_num_f.groupby(common_keys, as_index=False, dropna=False)
                        .agg({"Scallop": "sum"})
            )
            num_agg_df["Scallop"] = pd.to_numeric(num_agg_df["Scallop"], errors="coerce").astype(float)

            final_df = num_agg_df.merge(size_mean, on=common_keys, how="left")

            final_df["Drop_Date"]       = pd.to_datetime(final_df["Drop_Date"], errors="coerce")
            final_df["Monitoring_Date"] = pd.to_datetime(final_df["Monitoring_Date"], errors="coerce")
            final_df["Days"] = (final_df["Monitoring_Date"] - final_df["Drop_Date"]).dt.days

            for col in ["Drop_Date", "Monitoring_Date"]:
                final_df[col] = pd.to_datetime(final_df[col], errors="coerce").dt.strftime("%Y-%m-%d")

            preferred_cols = ["Place", "Drop_Date", "Monitoring_Date", "Days", "Scallop", "Size_mean(mm)"]
            ordered_cols = [c for c in preferred_cols if c in final_df.columns]
            other_cols   = [c for c in final_df.columns if c not in ordered_cols]
            final_df = final_df[ordered_cols + other_cols]

            def color_by_place(row: pd.Series):
                place = row.get("Place", "")
                color = place_bg.get(str(place).lower(), "")
                return [f"background-color: {color}" if color else "" for _ in row]

            styled_table = (
                final_df.style
                .apply(color_by_place, axis=1)
                .format({"Scallop": "{:.0f}", "Size_mean(mm)": "{:.1f}"})
            )
            st.dataframe(styled_table, use_container_width=True)

        if i < len(selected_areas) - 1:
            st.markdown("---")
            
# =========================
# カレンダー
# =========================
def render_calendar_mode():
    parent_folder_dr = pjoin(base_dir, "pred")
    if not os.path.exists(parent_folder_dr):
        st.error(f"フォルダが見つかりません: {parent_folder_dr}")
        st.stop()
    dr_files = list_dr_files_safe(base_dir)
    if not dr_files:
        st.warning("pred に CSV がありません")
        st.stop()
    selected_file = st.selectbox("対象エリア選択", sorted(dr_files))

    use_correction = st.sidebar.checkbox("実測ベース補正", value=False)
    tolerance_min = st.sidebar.slider("時刻差の許容範囲（分）", 5, 120, 35, step=5)
    train_days = st.sidebar.slider("補正学習期間（日数）", 7, 90, 30, step=1)
    bg_basis = st.sidebar.radio("セル背景の基準", ["予測に連動", "補正に連動"], index=0)
    max_h_vh = st.sidebar.slider("表の最大高さ (vh)", 40, 80, 65, step=5)
    recent_days = st.sidebar.slider("表示日数（直近）", 7, 10, 8, step=1)

    df_dr = load_dr_single_file(base_dir, selected_file)
    if df_dr.empty:
        st.warning("データが読み込めませんでした")
        st.stop()
    latest_dt = df_dr["datetime"].max()
    latest_day = latest_dt.date()
    depths_all = sorted([int(d) for d in df_dr["depth_m"].dropna().unique()])

    start_day = latest_day - pd.Timedelta(days=(recent_days - 1))
    end_day = latest_day
    day_list = list(pd.date_range(start_day, end_day, freq="D"))
    df_period = df_dr[df_dr["date_day"].isin([d.date() for d in day_list])].copy()

    parent_folder_obs = pjoin(base_dir, "obs")
    df_obs_period = pd.DataFrame()
    obs_path = pjoin(parent_folder_obs, selected_file)
    if os.path.exists(obs_path):
        try:
            df_obs = pd.read_csv(obs_path)
            df_obs["datetime"] = jst_to_naive(df_obs.get("Date"))
            df_obs["depth_m"] = pd.to_numeric(df_obs.get("Depth"), errors="coerce").round(0).astype("Int64")
            if ("U" in df_obs.columns) and ("V" in df_obs.columns):
                df_obs["U"] = pd.to_numeric(df_obs["U"], errors="coerce")
                df_obs["V"] = pd.to_numeric(df_obs["V"], errors="coerce")
                df_obs["Speed"] = np.sqrt(np.square(df_obs["U"]) + np.square(df_obs["V"]))
                df_obs["Direction_deg"] = (np.degrees(np.arctan2(df_obs["U"], df_obs["V"])) + 360.0) % 360.0
            df_obs = df_obs.dropna(subset=["datetime", "depth_m"]).copy()
            df_obs["date_day"] = df_obs["datetime"].dt.date
            df_obs_period = df_obs[df_obs["date_day"].isin([d.date() for d in day_list])].copy()
        except Exception as e:
            st.warning(f"読み込みに失敗: {obs_path} ({e})")
    else:
        st.info("実測データがありません。矢印は予測値です。")

    reg_depthwise, n_match_reg = None, None
    if use_correction:
        train_start_dt = pd.Timestamp(latest_day) - pd.Timedelta(days=train_days)
        train_end_dt = pd.Timestamp(latest_day) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        with st.spinner(
            f"回帰補正パラメータ算出中（{selected_file}, 終端={latest_day:%Y-%m-%d}, 遡り{train_days}日）..."
        ):
            reg_depthwise, n_match_reg = compute_depthwise_regression(
                base_dir, selected_file, tolerance_min,
                start_dt=train_start_dt, end_dt=train_end_dt, min_pairs=5
            )
        if reg_depthwise is None:
            st.warning("回帰係数の算出に失敗（一致データ不足など）。補正なしで表示。")
            use_correction = False
        else:
            with st.expander("補正の学習情報", expanded=False):
                st.write(f"期間: {train_start_dt:%Y-%m-%d} 〜 {train_end_dt:%Y-%m-%d}")
                st.write("一致ペア数:", {int(k): int(v) for k, v in (n_match_reg or {}).items()})
                st.write("学習済み水深（α, β）:", {int(k): (round(v[0],4), round(v[1],4)) for k, v in (reg_depthwise or {}).items()})

    # ---- CSS / 色 ----
    def get_calendar_css(max_h_vh: int) -> str:
        return f"""
<style>
.calendar-scroll-container {{
overflow: auto; max-height: {max_h_vh}vh; max-width: 100%;
-webkit-overflow-scrolling: touch; border:1px solid #e5e5e5; border-radius:8px; background:#fff;
}}
.calendar-table {{ border-collapse:collapse; width:max-content; min-width:640px; font-size:14px; font-family:'Noto Sans JP','Roboto',sans-serif; }}
.calendar-table th, .calendar-table td {{ padding:6px 10px; border-bottom:1px solid #eee; vertical-align:top; line-height:1.2; text-align:center; }}
.calendar-table thead th {{ position:sticky; top:0; background:#fafafa; z-index:2; white-space:nowrap; }}
.calendar-table td.depth-cell, .calendar-table thead th:first-child {{ position:sticky; left:0; background:#f7f7f7; z-index:3; min-width:40px; font-weight:600; }}
.pred-small {{ font-size:12px; color:#555; }}
@media (max-width:480px) {{ .calendar-table {{ font-size:13px; }} .calendar-table td.depth-cell {{ min-width:30px; }} }}
</style>
""".strip()

    def get_color(temp: float, t_min: float = 5, t_max: float = 25) -> str:
        if pd.isna(temp):
            return "rgba(220,220,220,0.6)"
        ratio = (float(temp) - t_min) / (t_max - t_min)
        ratio = max(0, min(1, ratio))
        if ratio < 0.5:
            r = int(240 * ratio * 2); g = int(240 * ratio * 2); b = 240
        else:
            r = 240; g = int(240 * (1 - (ratio - 0.5) * 2)); b = int(240 * (1 - (ratio - 0.5) * 2))
        return f"rgba({r},{g},{b},0.4)"

    def get_arrow_svg(direction_deg, speed_mps):
        if direction_deg is None or speed_mps is None or pd.isna(direction_deg) or pd.isna(speed_mps):
            return ""
        css_angle = (float(direction_deg) - 90.0) % 360.0
        speed_kt = float(speed_mps) * 1.94384
        if speed_kt < 1.0:
            size, color = 18, "#0000FF"
        elif speed_kt < 2.0:
            size, color = 22, "#FFC107"
        else:
            size, color = 26, "#FF0000"
        HEAD_LENGTH_RATIO = 0.55; HEAD_HALF_HEIGHT_RATIO = 0.35; SHAFT_WIDTH_PX = 4.0
        head_len = size * HEAD_LENGTH_RATIO
        head_half = size * HEAD_HALF_HEIGHT_RATIO
        line_end = size - head_len
        return (
            f"<svg width='{size}' height='{size}' style='display:block;margin:0 auto;transform:rotate({css_angle}deg);'>"
            f"<line x1='4' y1='{size/2}' x2='{line_end}' y2='{size/2}' stroke='{color}' stroke-width='{SHAFT_WIDTH_PX}' stroke-linecap='round'/>"
            f"<polygon points='{line_end},{size/2 - head_half} {size},{size/2} {line_end},{size/2 + head_half}' fill='{color}'/></svg>"
        )

    # レイヤ分け（表層/中層/底層）
    def make_layer_groups(depths: List[int]) -> Dict[str, List[int]]:
        if not depths:
            return {"表層": [], "中層": [], "底層": []}
        d = sorted(depths); n = len(d)
        if n <= 3:
            return {"表層": d[:1], "中層": d[1:2] if n>=2 else [], "底層": d[2:] if n>=3 else (d[-1:] if n>=1 else [])}
        if n in (4,5):
            return {"表層": d[:2], "中層": d[2:3], "底層": d[3:]}
        top, bot = d[:2], d[-2:]
        mid = [x for x in d if x not in top+bot]
        if len(mid) >= 3:
            c = len(mid)//2; mid = mid[c-1:c+1]
        return {"表層": top, "中層": mid, "底層": bot}

    # コメント（週間：水温／日：流向・流速）
    HIGH_TEMP_TH = 22.0
    DAY_BINS = [("朝", 4, 6), ("昼", 11, 13), ("夕", 16, 18)]

    def dir_to_8pt_jp(deg: float) -> str:
        if pd.isna(deg): return ""
        dirs = ["北","北東","東","南東","南","南西","西","北西"]
        idx = int(((float(deg)+22.5)%360)//45)
        return dirs[idx]

    def speed_class_from_mps(v_mps: Optional[float]) -> str:
        if v_mps is None or pd.isna(v_mps): return ""
        kt = float(v_mps) * 1.94384
        if kt >= 1.5: return "速"
        if kt >= 0.8: return "やや速"
        return "穏"

    def summarize_weekly_layer_temp(layer_name, layer_depths, df_period, df_all, selected_day,
                                    use_correction=False, reg_depthwise=None, stable_eps=0.4, outlier_th=None):
        PHYS_MIN, PHYS_MAX = -1.5, 35.0
        if not layer_depths or df_period is None or df_period.empty: return None
        msgs = []
        for depth in layer_depths:
            g = df_period[df_period["depth_m"] == depth].copy()
            if g.empty or "pred_temp" not in g.columns: continue
            g = g.sort_values("datetime")
            temps_pred = pd.to_numeric(g["pred_temp"], errors="coerce")
            if use_correction:
                if not (reg_depthwise and int(depth) in reg_depthwise): continue
                alpha, beta = reg_depthwise[int(depth)]
                temps_corr_raw = alpha + beta * temps_pred
                temps_corr_clipped = np.clip(temps_corr_raw, TEMP_MIN, TEMP_MAX)
                mask = (
                    pd.notna(temps_pred) & pd.notna(temps_corr_raw) &
                    (temps_corr_raw > PHYS_MIN) & (temps_corr_raw < PHYS_MAX) &
                    (temps_corr_clipped > TEMP_MIN) & (temps_corr_clipped < TEMP_MAX)
                )
                if outlier_th is not None:
                    diff = (temps_corr_raw - temps_pred).abs(); mask &= (diff < float(outlier_th))
                temps = pd.Series(temps_corr_raw)[mask]
            else:
                temps = temps_pred[(pd.notna(temps_pred)) & (temps_pred > PHYS_MIN) & (temps_pred < PHYS_MAX)]
            temps = pd.to_numeric(temps, errors="coerce").replace([np.inf,-np.inf], np.nan).dropna()
            if len(temps)==0: continue
            t_min, t_max = float(temps.min()), float(temps.max())
            if t_max >= HIGH_TEMP_TH:
                tag = f"高水温注意（{t_min:.1f}℃～{t_max:.1f}℃）"
            else:
                if len(temps) >= 2:
                    t_start, t_end = float(temps.iloc[0]), float(temps.iloc[-1])
                    delta = t_end - t_start
                    if delta > stable_eps: tag = f"上昇（{t_start:.1f}℃→{t_end:.1f}℃）"
                    elif delta < -stable_eps: tag = f"下降（{t_start:.1f}℃→{t_end:.1f}℃）"
                    else: tag = f"安定（{t_start:.1f}℃）"
                else:
                    t = float(temps.iloc[0]); tag = f"安定（{t:.1f}℃）"
            msgs.append(f"{depth}m{tag}")
        return (f"**{layer_name}**： " + "／".join(msgs)) if msgs else None

    def summarize_daily_layer_flow(layer_name, layer_depths, df_day):
        if not layer_depths: return None
        order = {"朝": 0, "昼": 1, "夕": 2}
        rows = []
        for label, h0, h1 in DAY_BINS:
            g = df_day[(df_day["depth_m"].isin(layer_depths)) & (df_day["datetime"].dt.hour.between(h0,h1))]
            if g.empty: continue
            U_mean = g["U"].mean() if "U" in g.columns else np.nan
            V_mean = g["V"].mean() if "V" in g.columns else np.nan
            if pd.notna(U_mean) and pd.notna(V_mean):
                speed = float(np.sqrt(U_mean**2 + V_mean**2)); deg = (np.degrees(np.arctan2(U_mean, V_mean)) + 360.0) % 360.0
            else:
                D = g["Direction_deg"].dropna() if "Direction_deg" in g.columns else pd.Series(dtype=float)
                if D.empty: continue
                rad = np.deg2rad(D.values); C = np.cos(rad).mean(); S = np.sin(rad).mean()
                deg = (np.degrees(np.arctan2(S, C)) + 360.0) % 360.0
                speed = g["Speed"].mean() if "Speed" in g.columns else np.nan
            d_txt = dir_to_8pt_jp(deg) if pd.notna(deg) else ""; v_cls = speed_class_from_mps(speed) if pd.notna(speed) else ""
            if d_txt or v_cls: rows.append((label, d_txt, v_cls))
        if not rows: return None
        rows = sorted(rows, key=lambda r: order.get(r[0], 99))
        seg = [f"{lbl}（{'・'.join([x for x in [d,v] if x])}）" for lbl,d,v in rows]
        return f"**{layer_name}**： " + "／".join(seg)

    def render_cell_html(temp, use_correction, corr_temp_raw, corr_temp_clip, bg_basis, is_invalid,
                        arrow_svg="", speed_kt_label="") -> str:
        if is_invalid:
            bg_color = "rgba(220,220,220,0.6)"
            return f"<td style='background:{bg_color}'><div style='text-align:center;'>-</div></td>"
        if (bg_basis == "補正に連動") and (corr_temp_clip is not None):
            bg_color = get_color(float(corr_temp_clip))
        else:
            bg_color = get_color(float(temp)) if (temp is not None and not pd.isna(temp)) else "rgba(220,220,220,0.6)"
        pred_label = f"{float(temp):.1f}°C" if (temp is not None and not pd.isna(temp)) else "NaN"
        pred_html = f"<span class='pred-small'>{pred_label}</span>"
        corr_html = f"<span style='color:#D32F2F;font-weight:700;font-size:16px;margin:0;padding:0;'>{corr_temp_raw:.1f}°C</span>" \
                    if use_correction and (corr_temp_raw is not None) else ""
        main = corr_html if corr_html else pred_html
        speed_html = f"<span style='font-size:12px;color:#444;'>{speed_kt_label}</span>" if speed_kt_label else ""
        arrow_html = f"<span style='display:block;line-height:1;margin:0;padding:0;'>{arrow_svg}</span>" if arrow_svg else ""
        content = ("<div style='display:flex;flex-direction:column;align-items:center;gap:2px;margin:0;padding:0;'>"
                + main + speed_html + arrow_html + "</div>")
        return f"<td style='background:{bg_color}'>{content}</td>"

    styles = get_calendar_css(max_h_vh)
    mode_view = st.radio("表示期間", ["週間予測（表示値は昼頃）", "選択日（1時間毎）"]) 
    layers = make_layer_groups(depths_all)

    # ===== 週間（昼頃） =====
    if mode_view == "週間予測（表示値は昼頃）":
        base_day = st.date_input("週間予測の基準日", value=end_day,
            min_value=min(df_dr["date_day"]) if not df_dr.empty else end_day,
            max_value=max(df_dr["date_day"]) if not df_dr.empty else end_day)
        start_day_hdr = pd.to_datetime(base_day) - pd.Timedelta(days=(recent_days - 1))
        end_day_hdr = pd.to_datetime(base_day)
        day_list = list(pd.date_range(start_day_hdr, end_day_hdr, freq="D"))
        df_period = df_dr[df_dr["date_day"].isin([d.date() for d in day_list])].copy()
        if 'df_obs' in locals():
            df_obs_period = df_obs[df_obs["date_day"].isin([d.date() for d in day_list])].copy()

        st.markdown(f"**{pd.to_datetime(start_day_hdr):%m/%d}～{pd.to_datetime(end_day_hdr):%m/%d}**")

        any_line = False
        for lname, ldepths in layers.items():
            line = summarize_weekly_layer_temp(lname, ldepths, df_period, df_dr, end_day_hdr,
                use_correction=use_correction, reg_depthwise=reg_depthwise, stable_eps=0.4, outlier_th=7.0)
            if line:
                any_line = True
            import re as _re
            def _highlight_high_temp(_s):
                return _re.sub(r'高水温注意（[^）]+）', lambda m: f"<span style='color:#D32F2F;font-weight:700;'>{m.group(0)}</span>", _s)
            st.markdown(_highlight_high_temp(line), unsafe_allow_html=True)
        if not any_line: st.caption("（特筆すべき変化はありません）")

        depths_for_table = list(depths_all)
        if use_correction and (reg_depthwise is not None):
            depths_for_table = [d for d in depths_for_table if int(d) in reg_depthwise]
        if not depths_for_table:
            st.caption("（補正係数が算出できた水深がありませんでした）")

        times = [d.strftime('%m/%d') for d in day_list]
        html = "<div class='calendar-scroll-container'><table class='calendar-table'>" \
            + "<thead><tr><th>水深</th>" + "".join([f"<th>{t}</th>" for t in times]) + "</tr></thead><tbody>"
        PHYS_MIN, PHYS_MAX = -1.5, 35.0
        for depth in depths_for_table:
            html += f"<tr><td class='depth-cell'>{depth}m</td>"
            for day in day_list:
                g = df_period[(df_period["date_day"] == day.date()) & (df_period["depth_m"] == depth)]
                if not g.empty:
                    target_dt = pd.Timestamp(day.date()) + pd.Timedelta(hours=12)
                    g2 = g.assign(_diff=(g["datetime"] - target_dt).abs()).sort_values("_diff")
                    row = g2.iloc[[0]].drop(columns=["_diff"])
                    temp = float(row["pred_temp"].values[0]) if "pred_temp" in row.columns else float(row.get("pred_temp", [np.nan])[0])
                    corr_raw = corr_clip = None
                    if use_correction and (reg_depthwise is not None) and (int(depth) in reg_depthwise) and not pd.isna(temp):
                        alpha, beta = reg_depthwise[int(depth)]; corr_raw = float(alpha + beta * float(temp))
                        corr_clip = float(np.clip(corr_raw, TEMP_MIN, TEMP_MAX))
                    is_invalid = False
                    if use_correction and (corr_raw is not None) and ((corr_raw <= PHYS_MIN) or (corr_raw >= PHYS_MAX)): is_invalid = True
                    if (not is_invalid) and (corr_clip is not None) and ((corr_clip <= TEMP_MIN) or (corr_clip >= TEMP_MAX)): is_invalid = True
                    if (not is_invalid) and (corr_raw is not None) and (abs(float(corr_raw)-float(temp)) >= 7.0): is_invalid = True
                    speed_mps = dir_deg = None
                    try:
                        tol = pd.Timedelta(minutes=int(tolerance_min))
                        obs_d = df_obs_period[df_obs_period["depth_m"] == depth]
                        if not obs_d.empty:
                            obs_d = obs_d.assign(_diff=(obs_d["datetime"] - target_dt).abs()).sort_values("_diff")
                            obs_near = obs_d.iloc[0]
                            if (obs_near["datetime"] >= target_dt - tol) and (obs_near["datetime"] <= target_dt + tol):
                                if {"U","V"}.issubset(set(df_obs_period.columns)):
                                    u = float(obs_near.get("U")) if pd.notna(obs_near.get("U")) else None
                                    v = float(obs_near.get("V")) if pd.notna(obs_near.get("V")) else None
                                    if (u is not None) and (v is not None):
                                        speed_mps = (u**2 + v**2) ** 0.5
                                        dir_deg = (np.degrees(np.arctan2(u, v)) + 360.0) % 360.0
                                if (speed_mps is None or dir_deg is None) and {"Direction_deg","Speed"}.issubset(set(df_obs_period.columns)):
                                    dd = float(obs_near.get("Direction_deg")) if pd.notna(obs_near.get("Direction_deg")) else None
                                    ss = float(obs_near.get("Speed")) if pd.notna(obs_near.get("Speed")) else None
                                    dir_deg = dd if dd is not None else dir_deg
                                    speed_mps = ss if ss is not None else speed_mps
                    except Exception:
                        pass
                    if (speed_mps is None or dir_deg is None):
                        try:
                            if {"U","V"}.issubset(set(row.columns)):
                                u = float(row["U"].values[0]) if not pd.isna(row["U"].values[0]) else None
                                v = float(row["V"].values[0]) if not pd.isna(row["V"].values[0]) else None
                                if (u is not None) and (v is not None):
                                    speed_mps = (u**2 + v**2) ** 0.5
                                    dir_deg = (np.degrees(np.arctan2(u, v)) + 360.0) % 360.0
                            elif {"Direction_deg","Speed"}.issubset(set(row.columns)):
                                dd = float(row["Direction_deg"].values[0]) if not pd.isna(row["Direction_deg"].values[0]) else None
                                ss = float(row["Speed"].values[0]) if not pd.isna(row["Speed"].values[0]) else None
                                dir_deg = dd if dd is not None else dir_deg
                                speed_mps = ss if ss is not None else speed_mps
                        except Exception:
                            pass
                    speed_kt_label = f"{(speed_mps*1.94384):.1f} kt" if (speed_mps is not None) else ""
                    arrow_svg = get_arrow_svg(dir_deg, speed_mps) if (speed_mps is not None and dir_deg is not None) else ""
                    html += render_cell_html(temp, use_correction, corr_raw, corr_clip, bg_basis, is_invalid,
                                            arrow_svg=arrow_svg, speed_kt_label=speed_kt_label)
                else:
                    html += "<td>-</td>"
            html += "</tr>\n"
        html += "</tbody></table></div>"
        full_html = f"<!doctype html><html><head><meta charset='utf-8'>{styles}</head><body>{html}</body></html>"
        iframe_height = int(max(400, min(1100, max_h_vh * 10)))
        st_html(full_html, height=iframe_height, scrolling=True)

    else:
        available_days = sorted(df_dr["date_day"].unique())
        min_day = min(available_days) if available_days else latest_day
        max_day = max(available_days) if available_days else latest_day
        selected_day = st.date_input("表示日（1時間毎）", value=max_day, min_value=min_day, max_value=max_day)
        df_day = df_dr[df_dr["date_day"] == selected_day].copy()
        hours_list = sorted(df_day["datetime"].dt.floor("h").unique())
        times_hr = [t.strftime('%H:%M') for t in hours_list]
        st.markdown("**朝(4～6時)、昼(11～13時)、夕(16～18時)**")
        any_line = False
        for lname, ldepths in layers.items():
            flow_line = summarize_daily_layer_flow(lname, ldepths, df_day)
            if flow_line:
                any_line = True
                st.markdown(flow_line, unsafe_allow_html=True)
        if not any_line: st.caption("（特筆すべき変化はありません）")

        if use_correction:
            sel_train_end_dt = pd.Timestamp(selected_day) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            sel_train_start_dt = pd.Timestamp(selected_day) - pd.Timedelta(days=train_days)
            with st.spinner(
                f"回帰補正パラメータ算出中（{selected_file}、終端={selected_day:%Y-%m-%d}、遡り{train_days}日）..."
            ):
                reg_depthwise_sel, n_match_reg_sel = compute_depthwise_regression(
                    base_dir, selected_file, tolerance_min,
                    start_dt=sel_train_start_dt, end_dt=sel_train_end_dt, min_pairs=5
                )
            if reg_depthwise_sel is None:
                st.warning("選択日終端での回帰係数算出に失敗（一致データ不足など）。補正なしで表示します。")
                use_correction = False
            else:
                reg_depthwise = reg_depthwise_sel
                total_pairs = sum((n_match_reg_sel or {}).values())
                st.caption(f"補正に使用したデータ数（選択日終端の一致ペア合計）: {total_pairs} 件")

        depths_for_table = list(depths_all)
        if use_correction and (reg_depthwise is not None):
            depths_for_table = [d for d in depths_for_table if int(d) in reg_depthwise]
        if not depths_for_table:
            st.caption("（補正係数が算出できた水深がありませんでした）")

        html = "<div class='calendar-scroll-container'><table class='calendar-table'>" \
            + "<thead><tr><th>水深</th>" + "".join([f"<th>{t}</th>" for t in times_hr]) + "</tr></thead><tbody>"
        PHYS_MIN, PHYS_MAX = -1.5, 35.0
        for depth in depths_for_table:
            html += f"<tr><td class='depth-cell'>{depth}m</td>"
            for t_obj in hours_list:
                row = df_day[(df_day["datetime"].dt.floor("h") == t_obj) & (df_day["depth_m"] == depth)]
                if not row.empty:
                    temp = float(row["pred_temp"].values[0]) if "pred_temp" in row.columns else float(row.get("pred_temp", [np.nan])[0])
                    corr_raw = corr_clip = None
                    if use_correction and (reg_depthwise is not None) and (int(depth) in reg_depthwise) and not pd.isna(temp):
                        alpha, beta = reg_depthwise[int(depth)]; corr_raw = float(alpha + beta * float(temp))
                        corr_clip = float(np.clip(corr_raw, TEMP_MIN, TEMP_MAX))
                    is_invalid = False
                    if use_correction and (corr_raw is not None) and ((corr_raw <= PHYS_MIN) or (corr_raw >= PHYS_MAX)): is_invalid = True
                    if (not is_invalid) and (corr_clip is not None) and ((corr_clip <= TEMP_MIN) or (corr_clip >= TEMP_MAX)): is_invalid = True
                    if (not is_invalid) and (corr_raw is not None) and (abs(float(corr_raw)-float(temp)) >= 7.0): is_invalid = True
                    speed_mps = dir_deg = None
                    try:
                        tol = pd.Timedelta(minutes=int(tolerance_min))
                        obs_d = df_obs_period[df_obs_period["depth_m"] == depth]
                        if not obs_d.empty:
                            obs_d = obs_d.assign(_diff=(obs_d["datetime"] - t_obj).abs()).sort_values("_diff")
                            obs_near = obs_d.iloc[0]
                            if (obs_near["datetime"] >= t_obj - tol) and (obs_near["datetime"] <= t_obj + tol):
                                if {"U","V"}.issubset(set(df_obs_period.columns)):
                                    u = float(obs_near.get("U")) if pd.notna(obs_near.get("U")) else None
                                    v = float(obs_near.get("V")) if pd.notna(obs_near.get("V")) else None
                                    if (u is not None) and (v is not None):
                                        speed_mps = (u**2 + v**2) ** 0.5
                                        dir_deg = (np.degrees(np.arctan2(u, v)) + 360.0) % 360.0
                                if (speed_mps is None or dir_deg is None) and {"Direction_deg","Speed"}.issubset(set(df_obs_period.columns)):
                                    dd = float(obs_near.get("Direction_deg")) if pd.notna(obs_near.get("Direction_deg")) else None
                                    ss = float(obs_near.get("Speed")) if pd.notna(obs_near.get("Speed")) else None
                                    dir_deg = dd if dd is not None else dir_deg
                                    speed_mps = ss if ss is not None else speed_mps
                    except Exception:
                        pass
                    if (speed_mps is None or dir_deg is None):
                        try:
                            if {"U","V"}.issubset(set(row.columns)):
                                u = float(row["U"].values[0]) if not pd.isna(row["U"].values[0]) else None
                                v = float(row["V"].values[0]) if not pd.isna(row["V"].values[0]) else None
                                if (u is not None) and (v is not None):
                                    speed_mps = (u**2 + v**2) ** 0.5
                                    dir_deg = (np.degrees(np.arctan2(u, v)) + 360.0) % 360.0
                            elif {"Direction_deg","Speed"}.issubset(set(row.columns)):
                                dd = float(row["Direction_deg"].values[0]) if not pd.isna(row["Direction_deg"].values[0]) else None
                                ss = float(row["Speed"].values[0]) if not pd.isna(row["Speed"].values[0]) else None
                                dir_deg = dd if dd is not None else dir_deg
                                speed_mps = ss if ss is not None else speed_mps
                        except Exception:
                            pass
                    speed_kt_label = f"{(speed_mps*1.94384):.1f} kt" if (speed_mps is not None) else ""
                    arrow_svg = get_arrow_svg(dir_deg, speed_mps) if (speed_mps is not None and dir_deg is not None) else ""
                    html += render_cell_html(temp, use_correction, corr_raw, corr_clip, bg_basis, is_invalid,
                                            arrow_svg=arrow_svg, speed_kt_label=speed_kt_label)
                else:
                    html += "<td>-</td>"
            html += "</tr>\n"
        html += "</tbody></table></div>"
        styles2 = get_calendar_css(max_h_vh)
        full_html = f"<!doctype html><html><head><meta charset='utf-8'>{styles2}</head><body>{html}</body></html>"
        iframe_height = int(max(400, min(1100, max_h_vh * 10)))
        st_html(full_html, height=iframe_height, scrolling=True)

# =========================
# メイン
# =========================
def reset_sidebar_state_for(prefix_keep: str):
    prefixes = ('sc_', 'larv_', 'water_', 'cal_')
    for k in list(st.session_state.keys()):
        if k.startswith(prefixes) and not k.startswith(prefix_keep):
            try:
                del st.session_state[k]
            except KeyError:
                pass

def main():
    st.set_page_config(layout="wide")

    try:
        mode = st.segmented_control(
            "モード選択",
            options=["カレンダー", "水温(GSI)", "ラーバ", "採苗数"],
            key="main_mode",
            default="カレンダー"
        )
    except Exception:
        mode = st.radio(
            "モード選択",
            options=["カレンダー", "水温(GSI)", "ラーバ", "採苗数"],
            index=0,
            horizontal=True,
            key="main_mode"
        )
    areas_all = load_all_areas()
    areas_all = load_all_areas()

    # ---- サイドバーUI（条件表示）----
    sel_areas = None
    with st.sidebar:
        if mode == "水温(GSI)":
            st.caption("")

        if mode in ("水温(GSI)", "ラーバ", "採苗数"):
            if mode == "水温(GSI)":
                key_areas = "water_areas"
            elif mode == "ラーバ":
                key_areas = "larv_areas"
            else:  # "採苗数"
                key_areas = "sc_areas"

            sel_areas = st.multiselect(
                "エリア選択（複数可）",
                options=areas_all,
                default=[],
                key=key_areas
            )

    if mode == "水温(GSI)":
        reset_sidebar_state_for('water_')
        render_water_with_optional_gsi_overlay(selected_areas_for_gsi=sel_areas)

    elif mode == "ラーバ":
        reset_sidebar_state_for('larv_')
        render_larvae_mode(sel_areas)

    elif mode == "採苗数":
        reset_sidebar_state_for('sc_')
        render_scallop_mode(sel_areas)

    elif mode == "地図":
        reset_sidebar_state_for('map_')
        render_map_mode()

    else:  # "カレンダー"
        reset_sidebar_state_for('cal_')
        # ★ ここが抜けていたため表示されない
        render_calendar_mode()

if __name__ == "__main__":
    main()