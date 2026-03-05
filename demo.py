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
from pathlib import Path

# =========================
# 基本設定
# =========================
st.set_page_config(page_title="統合版", layout="wide")
ANCHOR_YEAR = 2000


DEFAULT_BASE_DIR = "data"
base_dir = os.environ.get("APP_BASE_DIR", DEFAULT_BASE_DIR)

def pjoin(*parts: str) -> str:
    return os.path.normpath(os.path.join(*parts))


MATURITY_PATH = pjoin(base_dir, "maturity.csv")          
LARVAE_PATH   = pjoin(base_dir, "larvae.csv")            
COLLECTOR_NUMBER_PATH = pjoin(base_dir, "collector_number.csv")


TITLE_SIZE = 18
TEMP_MIN, TEMP_MAX = -2.0, 40.0

# =========================
# 共通ユーティリティ
# =========================

from typing import Optional
def inject_compact_css():
    st.markdown("""
    <style>
      .block-container { padding-top: 0.8rem; padding-bottom: 0.8rem; }
      [data-testid="stSidebar"] { width: 18rem; }
      .stPlotlyChart, .element-container { margin-bottom: 0.6rem; }
    </style>
    """, unsafe_allow_html=True)

def file_fingerprint(path: str) -> str:
    try:
        st = os.stat(path)
        return f"{st.st_size}-{int(st.st_mtime)}"
    except Exception:
        return ""

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


# =========================
# GSI集計（オーバーレイ用｜Sex別に常時分割）
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

    # Sex が無い or 欠損 → Unknown で補う
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
# 水温グラフ（MM/DD入力・年選択適用・GSI帯の弱色化 版）
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

# NOTE: 以下の変数/関数は環境に既存前提
# base_dir, pjoin, ANCHOR_YEAR, MATURITY_PATH, TEMP_MIN, TEMP_MAX
# load_dr_single_file, jst_to_naive, safe_merge_asof_by_depth, compute_depthwise_regression,
# read_csv_path, get_gsi_agg

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

    # ======================
    # ユーティリティ
    # ======================
    def parse_mmdd(s: str) -> dt.date:
        """MM/DD を dt.date(ANCHOR_YEAR-mm-dd) へ。エラー時は None。"""
        try:
            m, d = dt.datetime.strptime(s.strip(), "%m/%d").month, dt.datetime.strptime(s.strip(), "%m/%d").day
            return dt.date(ANCHOR_YEAR, m, d)
        except Exception:
            return None

    def to_anchor_ts(ts: pd.Series) -> pd.Series:
        """datetime -> ANCHOR_YEARへ年を差し替えたTimestamp（x軸用）"""
        d = pd.to_datetime(ts, errors="coerce")
        return pd.to_datetime(d.dt.strftime(f"{ANCHOR_YEAR}-%m-%d %H:%M:%S"))

    def mmdd_mask(series_dt: pd.Series, start_anchor: pd.Timestamp, end_anchor: pd.Timestamp) -> pd.Series:
        """月日範囲（年跨ぎ対応）でTrue/Falseを返す"""
        anchored = pd.to_datetime(series_dt.dt.strftime(f"{ANCHOR_YEAR}-%m-%d %H:%M:%S"))
        if start_anchor <= end_anchor:
            return (anchored >= start_anchor) & (anchored <= end_anchor)
        else:
            # wrap（例：12/15〜01/15）は「>= start」or「<= end」
            return (anchored >= start_anchor) | (anchored <= end_anchor)

    def anchored_day_span(start_anchor: pd.Timestamp, end_anchor: pd.Timestamp) -> int:
        """選択範囲の日数（wrap対応、両端含む）"""
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
            "対象エリアを選択",
            sorted(dr_files),
            key="sb_selected_file"
        )

        # DRプレビュー
        df_dr_preview = load_dr_single_file(base_dir, selected_file)
        if df_dr_preview.empty:
            st.warning("DRデータが読み込めませんでした")
            st.stop()

        # 利用可能年（DR/GSIから統合）
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

        overlay_gsi   = st.checkbox("GSIを右軸で重ねる", value=False, key="sb_overlay_gsi")
        use_correction = st.checkbox("実測ベース補正(回帰)", value=False, key="sb_use_correction")
        show_sekisan   = st.checkbox("積算水温を表示する", value=False, key="show_sekisan")

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
        st.warning("DRデータが読み込めませんでした")
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
    selected_depths = st.multiselect("表示する水深(複数選択可)", depths_all, default=default_depths, key="main_depths")

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
                st.warning(f"OBSの読み込みに失敗しました: {obs_path} ({e})")
                df_obs_period = pd.DataFrame()
        else:
            st.info("obs フォルダに同名CSVがありません。実測点は表示されません。")

    # 補正学習（選択年の最新日時を基準に直近30日）
    reg_depthwise, n_match_reg = None, None
    if use_correction:
        if df_dr.empty:
            st.warning("補正用のDRデータがありません")
        else:
            mask_period = mmdd_mask(df_dr["datetime"], start_anchor_ts, end_anchor_ts)
            df_dr_period = df_dr[mask_period].copy()
            if df_dr_period.empty:
                st.warning("選択期間内にDRデータがありません。最新時刻からの基準にフォールバックします。")
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

    # GSI年選択（共通年を使用）
    gsi_years_sel = selected_years if overlay_gsi else []
    area_year_sex_dict, all_mmdd = ({}, [])
    if overlay_gsi:
        area_year_sex_dict, all_mmdd = get_gsi_agg(selected_areas_for_gsi, gsi_years_sel)
    sex_style = {
        "F": {"dash": "dash", "alpha_band": 0.18, "label": "F", "color": "#d62728"},  # 雌
        "M": {"dash": "solid",  "alpha_band": 0.18, "label": "M", "color": "#1f77b4"},  # 雄
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
            # OBSがある水深だけに制限（補正ON時の方針）
            if use_correction and only_depths_with_obs_when_correct and not df_obs_period.empty:
                obs_depths = set(df_obs_period["depth_m"].dropna().astype(int).tolist())
                if int(d) not in obs_depths:
                    continue
            for idx_y, y in enumerate(selected_years if selected_years else sorted(df_dr["year"].unique().tolist())):
                df_dy = df_dr[(df_dr["depth_m"] == d) & (df_dr["year"] == y)].copy()
                if df_dy.empty:
                    continue
                # 1Hリサンプル（中央値）・補間
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

                # 実測値（OBS水温）
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
                           
                # 補正水温（補正ON時のみ）
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
                    

                # 積算水温（日平均・オンのとき）
                if show_sekisan:
                    # ① 期間マスク“前”の df_dy_full を作る（深度・年で絞るが、月日フィルタはかけない）
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
                        # ③ 積算の“計算”は「起算～期間終了」で行う（wrap対応）
                        mask_calc = mmdd_mask(df_daily_full["dt"], sekisan_anchor_ts, end_anchor_ts)
                        df_daily_calc = df_daily_full[mask_calc].copy()
                        if not df_daily_calc.empty:
                            # ④ “表示”は「期間開始～期間終了」でスライス（値は起算からの通算）
                            mask_show = mmdd_mask(df_daily_calc["dt"], start_anchor_ts, end_anchor_ts)
                            df_daily_show = df_daily_calc[mask_show].copy()
                            if not df_daily_show.empty:
                                x_sekisan = df_daily_show["dt"].map(lambda d0: pd.Timestamp(f"{ANCHOR_YEAR}-{d0:%m-%d}"))
                                # 予測積算（起算からの通算）
                                y_pred_accum = df_daily_calc["pred_temp"].cumsum()
                                # 表示区間に合わせた値だけ抽出（dtで join）
                                y_pred_accum_show = y_pred_accum.loc[df_daily_calc.index].reindex(df_daily_show.index).values
                                fig.add_trace(go.Scatter(
                                    x=x_sekisan, y=y_pred_accum_show, mode="lines",
                                    name=f"{d}m 積算水温（予測） {y}",
                                    line=dict(color=line_color, width=2, dash="dot"),
                                    opacity=0.70
                                ), row=1, col=1, secondary_y=False)
                                # 補正積算（補正ON時のみ）：起算から通算→表示区間へ
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

    # GSIオーバーレイ（右軸）：平均±1σ帯＋平均線（弱色帯、年なし hover）
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
    show_legend = st.checkbox("凡例を表示", value=True, key="main_show_legend")
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

    # X軸：自動刻み幅
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
# ラーバモード（表はページ最下段にまとめて表示：初期は折りたたみ）
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
    """
    Mode B（横棒：年×日別）— 左端に“年専用列”を追加（軸は非表示）
    - 月日のサブプロットタイトルは列ごとの上部（M/D 表記）
    - 本体のY軸ラベルは col=2 のみ表示（col>=3 は非表示）
    - 棒の色は3分類（<200 / 200–259 / >=260）
    - X軸の最大値は「モードでひとつ（x_max）」を使用
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    def mmdd_to_md(mdstr: str) -> str:
        m, d = mdstr.split('-')
        return f"{int(m)}/{int(d)}"

    # 月日整列＋最大日数制限
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

    # サブプロット構成（左端は年列＝タイトルなし）
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
                # f文字列内の %{..} は {{..}} でエスケープ
                hovertemplate=(f"年: {yr}<br>日: {mmdd_to_md(md)}<br>帯: %{{y}}<br>合計: %{{x:.2f}}")
            ), row=r, col=idx)

            # Y軸は col=2 のみ表示
            fig.update_yaxes(
                categoryorder="array", categoryarray=band_labels, automargin=True,
                showticklabels=(idx == 2),
                ticks=("outside" if idx == 2 else ""),
                row=r, col=idx
            )
            # X軸の共通最大値
            fig.update_xaxes(range=[0, x_max], row=r, col=idx)

        # 左ガターに年注釈（縦書き）
        R = max(1, len(years_to_show))
        y_paper_mid = 1 - (r - 0.5) / R
        # Y軸の項目値（サイズ帯ラベル）より確実に左へ：paper左端(x=0)から左マージン内へxshiftで退避
        # ※margin.l=120 前提。ラベル幅に余裕を見て -80px 固定（隙間を詰める）
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
    """
    ラーバ：Mode B（縦棒：期間／比率、横棒：年×日別／実値）
    - 縦棒は3分類色を維持しつつ年度ごとの濃淡で視認性UP（新しい年ほど濃い）
    - 横棒は年×日別列（元実装準拠）
    - 表は常に生成し、ページ最下段でまとめて表示（初期は折りたたみ）
    - 期間選択は採苗数モードと同じ“スライダー（MM-DD）”
    - グラフ直下に「期間/Area/年」キャプションを表示
    """
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

    # サイズ列抽出・数値化
    size_cols  = [c for c in df.columns if c.isdigit()]  # 例: "160", "180", ...
    size_ints  = sorted(int(c) for c in size_cols)
    others_col = next((c for c in df.columns if c.lower().startswith("others")), None)
    for c in size_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    if others_col:
        df[others_col] = pd.to_numeric(df[others_col], errors="coerce").fillna(0.0)

    # 画面内UI（ラーバモードはサイドバーを使わない）
    # ① 横棒の最大表示：5日固定
    # ② 横棒X軸：表示される日（先頭5日）の最大値に自動追従（任意調整なし）
    # ③ グラフ種類：日別 / 期間（文言短縮）
    # ④ エリア選択・表示年・対象期間もメイン画面で完結

    areas_all = sorted(df["Area"].dropna().astype(str).unique().tolist()) if "Area" in df.columns else []
    default_areas = (selected_areas or [])

    # 上部：エリア選択（メイン画面）
    sel_areas_main = st.multiselect(
        "エリア選択（複数可）",
        options=areas_all,
        default=default_areas,
        key='larv_areas_main'
    )
    # multiselect が None を返す環境差を吸収（None→空リスト）
    sel_areas_main = sel_areas_main or []

    if not sel_areas_main:
        st.info("エリアを選択してください。")
        return

    years_all = sorted(df["Year"].dropna().unique().tolist())
    latest = years_all[-1] if years_all else None

    c1, c2, c3 = st.columns([1.1, 2.0, 1.0])
    with c1:
        years_sel = st.multiselect("表示年", years_all, default=[latest] if latest else [], key='larv_years')

    # 採苗数モードと同じレンジ・既定（3/1〜7/31）
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
            mode_b = st.segmented_control('', options=['日別推移','期間内比率'], default='日別推移', key='larv_mode', label_visibility='collapsed')
        except Exception:
            mode_b = st.radio('', ['日別推移','期間内比率'], index=0, horizontal=True, key='larv_mode_radio', label_visibility='collapsed')

    # 横棒（日別）は最大5日で固定
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

    # ==== 表示対象日の決定（先頭 max_days 日） ==== 
    q_days = df[df['Area'].isin(sel_areas_main)].copy()
    if years_sel:
        q_days = q_days[q_days['Year'].isin(years_sel)]
    q_days = q_days[q_days['md_doy'].apply(lambda d: in_window(int(d), s_doy, e_doy))]

    days_all = sorted(
        q_days['MMDD'].astype(str).unique().tolist(),
        key=lambda s: pd.to_datetime(f"2000-{s}").dayofyear
    )
    days_show = days_all[:max_days] if len(days_all) > max_days else days_all

    # 横棒のX最大値（表示される日だけで最大値を評価）
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

    # —— ここからエリアごとの描画（サブヘッダーなし）。表は蓄積して最後にまとめて表示 ——
    tables_to_show: list[tuple[str, pd.DataFrame]] = []

    for i, area in enumerate(sel_areas_main):
        # サブヘッダーは表示しない（Area: xxx の見出しは不使用）

        df_area = filter_by_areas(df, [area])

        # 共通フィルタ（年・期間）
        q_area = df_area.copy()
        if years_sel:
            q_area = q_area[q_area["Year"].isin(years_sel)]
        q_area = q_area[q_area["md_doy"].apply(lambda d: in_window(d, s_doy, e_doy))]

        if q_area.empty or not size_cols:
            st.info("選択条件に該当するデータがありません。")
            if i < len(sel_areas_main or []) - 1:
                st.markdown("---")  # 区切り線は維持
            continue

        # ===== 縦棒（期間）=====
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
                years_sorted = sorted(bars_df["Year"].unique())  # 古い→新しい

                # 年度ごとの濃淡（不透明度）：新しい年ほど濃い
                def opacity_for_year(yr: int) -> float:
                    if len(years_sorted) == 1:
                        return 0.95
                    i = years_sorted.index(yr)            # 0..n-1
                    frac = (i + 1) / len(years_sorted)    # 0..1
                    return min(1.0, 0.30 + 0.75 * frac)   # 0.30〜1.00

                # HEX → RGBA
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

        # ▼ 表は“作成のみ”（後でまとめて表示）
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

        # エリア区切り線（維持）
        if i < len(sel_areas_main or []) - 1:
            st.markdown("---")

    # —— ページ最下段：表をまとめて描画（常時。ただし初期は折りたたみ） ——
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
        if i < len(sel_areas_main or []) - 1:
            st.markdown("---")





# =========================
# 経年比較モード（大型ラーバ累積 × 付着数 合算）
# - X: Drop_Date 以降、Monitoring_Date 前までの 大型ラーバ(>=250)累積
# - Y: Place をまとめた Scallop 合計（Monitoring_Date 時点の累積付着数）
# - 色: 投入年（Drop_Date の年）※カテゴリ（凡例）
# - 濃淡: 同一年・同一Areaの DropDate の早遅（早いほど薄い）
# - 点サイズ: (Monitoring_Date - Drop_Date) の日数
# - 表示: エリアは選択式（ALL / 1エリア）
# - 重要: read_csv_path は cache されるため、fp(指紋) を渡してファイル更新を反映
# =========================

def render_yearly_compare_mode():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go

    # --- 読み込み（既存と同じ） ---
    df_l = read_csv_path(LARVAE_PATH, fp=file_fingerprint(LARVAE_PATH))
    df_c = read_csv_path(COLLECTOR_NUMBER_PATH, fp=file_fingerprint(COLLECTOR_NUMBER_PATH))

    # ★追加：殻サイズ（Shell）読み込み
    COLLECTOR_SIZE_PATH = pjoin(base_dir, "collector_size.csv")
    df_s = read_csv_path(COLLECTOR_SIZE_PATH, fp=file_fingerprint(COLLECTOR_SIZE_PATH))

    if df_l is None or df_c is None:
        st.stop()
    if df_l.empty:
        st.warning("larvae.csv が空です")
        return
    if df_c.empty:
        st.warning("collector_number.csv が空です")
        return

    # ---------- larvae（日別 大型>=250） ----------
    df_l = df_l.copy()
    df_l["Date"] = pd.to_datetime(df_l.get("Date"), errors="coerce")
    df_l["Area"] = df_l.get("Area", "").astype(str).str.strip()

    size_cols = [c for c in df_l.columns if str(c).isdigit()]
    for c in size_cols:
        df_l[c] = pd.to_numeric(df_l[c], errors="coerce").fillna(0.0)

    large_cols = [c for c in size_cols if int(c) >= 250]
    df_l["X_large_day"] = df_l[large_cols].sum(axis=1) if large_cols else 0.0

    df_l = df_l.dropna(subset=["Date", "Area"]).copy()
    if df_l.empty:
        st.info("larvae.csv の有効行がありません")
        return

    df_l["date"] = df_l["Date"].dt.date
    df_l_day = (
        df_l.groupby(["Area", "date"], as_index=False)["X_large_day"].sum()
    )
    larv_by_area = {
        a: g[["date", "X_large_day"]].copy()
        for a, g in df_l_day.groupby("Area", sort=False)
    }

    # ---------- collector_number（イベント: Area×Drop×Monitoring / Place合算） ----------
    df_c = df_c.copy()
    df_c["Drop_Date"] = pd.to_datetime(df_c.get("Drop_Date"), errors="coerce")
    df_c["Monitoring_Date"] = pd.to_datetime(df_c.get("Monitoring_Date"), errors="coerce")
    df_c["Area"] = df_c.get("Area", "").astype(str).str.strip()
    df_c["Scallop"] = pd.to_numeric(df_c.get("Scallop"), errors="coerce")

    df_c = df_c.dropna(subset=["Drop_Date", "Monitoring_Date", "Area", "Scallop"]).copy()
    if df_c.empty:
        st.info("collector_number.csv の有効行がありません")
        return

    df_c["Drop_Year"] = df_c["Drop_Date"].dt.year.astype(int)
    df_c["Drop_day"] = df_c["Drop_Date"].dt.date
    df_c["Monitoring_day"] = df_c["Monitoring_Date"].dt.date

    # Place 合算（Area×Drop×Monitoring で1点）
    df_e = (
        df_c.groupby(["Area", "Drop_day", "Monitoring_day"], as_index=False)
            .agg(
                Y_total=("Scallop", "mean"),
                Drop_Year=("Drop_Year", "first")
            )
            .rename(columns={"Drop_day": "Drop_Date", "Monitoring_day": "Monitoring_Date"})
    )

    # 期間（日）も保持（hover用には残す）
    df_e["duration_days"] = (
        pd.to_datetime(df_e["Monitoring_Date"]) - pd.to_datetime(df_e["Drop_Date"])
    )
    df_e["duration_days"] = pd.to_numeric(df_e["duration_days"].dt.days, errors="coerce")
    df_e = df_e.dropna(subset=["duration_days"]).copy()
    df_e["duration_days"] = df_e["duration_days"].astype(int)

    if df_e.empty:
        st.info("有効なイベントがありません")
        return

    # ---------- X_total / overlap_days（行ごと計算） ----------
    def _calc_x(row):
        larv = larv_by_area.get(row["Area"])
        if larv is None or larv.empty:
            return 0.0, 0
        drop = row["Drop_Date"]
        mon = row["Monitoring_Date"]
        # 仕様：Monitoring当日は含めない（< mon）
        m = (larv["date"] >= drop) & (larv["date"] < mon)
        return float(larv.loc[m, "X_large_day"].sum()), int(m.sum())

    tmp = df_e.apply(_calc_x, axis=1, result_type="expand")
    tmp.columns = ["X_total", "overlap_days"]

    df_plot = pd.concat([df_e, tmp], axis=1)
    df_plot = df_plot[df_plot["overlap_days"] > 0].copy()
    if df_plot.empty:
        st.info("ラーバ採取日がイベント期間に重なるデータがありません")
        return

    # ---------- ★ Shell(mm) 平均をイベントに付与（collector_size.csv｜近傍マッチ） ----------
    # 目的：殻長データは Monitoring_Date が数日ズレることがあるため、完全一致ではなく
    #       「同一 Area × Drop_Date 内で Monitoring_Date 最近傍（tolerance付き）」で紐付ける。
    SHELL_TOL_DAYS = 5  # ←許容日数（±N日）。必要なら調整

    # collector_size.csv の想定列：Drop_Date, Monitoring_Date, Area, Place, Shell(mm)
    if df_s is None or df_s.empty:
        df_plot["shell_mean"] = np.nan
        df_plot["shell_n"] = np.nan
    else:
        df_s = df_s.copy()
        df_s["Drop_Date"] = pd.to_datetime(df_s.get("Drop_Date"), errors="coerce")
        df_s["Monitoring_Date"] = pd.to_datetime(df_s.get("Monitoring_Date"), errors="coerce")
        df_s["Area"] = df_s.get("Area", "").astype(str).str.strip()

        # 殻長列名の揺れを吸収
        shell_col = None
        for cand in ["Shell(mm)", "Shell", "Shell_mm", "shell", "shell_mm", "殻長", "殻長(mm)"]:
            if cand in df_s.columns:
                shell_col = cand
                break

        if shell_col is None:
            df_plot["shell_mean"] = np.nan
            df_plot["shell_n"] = np.nan
        else:
            df_s[shell_col] = pd.to_numeric(df_s.get(shell_col), errors="coerce")

            # 日別平均へ集約（同日複数個体/複数Place対応）
            shell_daily = (
                df_s.dropna(subset=["Area", "Drop_Date", "Monitoring_Date", shell_col])
                    .assign(
                        Drop_Date_dt=lambda d: d["Drop_Date"].dt.floor("D"),
                        Monitoring_Date_dt=lambda d: d["Monitoring_Date"].dt.floor("D"),
                    )
                    .groupby(["Area", "Drop_Date_dt", "Monitoring_Date_dt"], as_index=False)
                    .agg(shell_mean=(shell_col, "mean"), shell_n=(shell_col, "count"))
            )

            df_plot = df_plot.assign(
                Drop_Date_dt=pd.to_datetime(df_plot["Drop_Date"], errors="coerce").dt.floor("D"),
                Monitoring_Date_dt=pd.to_datetime(df_plot["Monitoring_Date"], errors="coerce").dt.floor("D"),
            )

            # 最近傍マッチ（同一 Area × Drop_Date_dt 内で Monitoring_Date_dt 最近傍）
            df_plot = df_plot.dropna(subset=["Area", "Drop_Date_dt", "Monitoring_Date_dt"]).copy()
            df_plot["Monitoring_Date_dt"] = pd.to_datetime(df_plot["Monitoring_Date_dt"], errors="coerce")
            df_plot = df_plot.sort_values(["Monitoring_Date_dt", "Area", "Drop_Date_dt"]).reset_index(drop=True).copy()
            shell_daily = shell_daily.dropna(subset=["Area", "Drop_Date_dt", "Monitoring_Date_dt"]).copy()
            shell_daily["Monitoring_Date_dt"] = pd.to_datetime(shell_daily["Monitoring_Date_dt"], errors="coerce")
            shell_daily = shell_daily.sort_values(["Monitoring_Date_dt", "Area", "Drop_Date_dt"]).reset_index(drop=True).copy()

            df_plot = pd.merge_asof(
                df_plot,
                shell_daily,
                left_on="Monitoring_Date_dt",
                right_on="Monitoring_Date_dt",
                by=["Area", "Drop_Date_dt"],
                direction="nearest",
                tolerance=pd.Timedelta(days=SHELL_TOL_DAYS),
                suffixes=("", "_s"),
            )

    # ---------- UI（エリア選択） ----------
    areas = sorted(df_plot["Area"].dropna().astype(str).unique().tolist())
    area_sel = st.selectbox("エリア", ["ALL"] + areas, index=0, key="yc_area_sel")
    if area_sel != "ALL":
        df_plot = df_plot[df_plot["Area"] == area_sel].copy()
        if df_plot.empty:
            st.info("選択エリアのデータがありません")
            return

    # 年（色）準備
    df_plot["Drop_Year"] = pd.to_numeric(df_plot["Drop_Year"], errors="coerce")
    df_plot = df_plot.dropna(subset=["Drop_Year"]).copy()
    df_plot["Drop_Year"] = df_plot["Drop_Year"].astype(int)
    df_plot["Drop_Year_cat"] = df_plot["Drop_Year"].astype(str)

    years_present = sorted(df_plot["Drop_Year"].unique().tolist())

    # 表示用の丸め
    df_plot["X_total_disp"] = df_plot["X_total"].round(2)
    df_plot["Y_total_disp"] = df_plot["Y_total"].round(0).astype(int)

    # 同じ板（Area + Drop_Date）を識別（時系列線の単位）
    df_plot["line_id"] = df_plot["Area"].astype(str) + " " + df_plot["Drop_Date"].astype(str)

    # ★ Monitoring_Date 順で時系列
    df_plot = df_plot.sort_values(["Area", "Drop_Date", "Monitoring_Date"]).copy()

    # ---------- ★ Shell 欠損を「前後の時系列平均」で補完 ----------
    # 同一 Area×Drop_Date 内で、Monitoring_Date の前後を参照して補完
    df_plot["shell_filled"] = df_plot["shell_mean"]

    for (a, d), g in df_plot.groupby(["Area", "Drop_Date"], sort=False):
        # Monitoring_Date 順を保証
        g = g.sort_values("Monitoring_Date")
        prev = g["shell_mean"].ffill()
        nxt = g["shell_mean"].bfill()

        filled = g["shell_mean"].copy()
        m = filled.isna()

        both = m & prev.notna() & nxt.notna()
        only_prev = m & prev.notna() & nxt.isna()
        only_next = m & prev.isna() & nxt.notna()

        filled.loc[both] = 0.5 * (prev.loc[both] + nxt.loc[both])
        filled.loc[only_prev] = prev.loc[only_prev]
        filled.loc[only_next] = nxt.loc[only_next]

        df_plot.loc[g.index, "shell_filled"] = filled.values

    # ---------- ★ マーカーサイズ：shell_filled を 6～40 にスケール ----------
    size_min = 6.0
    size_max = 40.0
    v = pd.to_numeric(df_plot["shell_filled"], errors="coerce")
    vmin = float(v.min()) if v.notna().any() else np.nan
    vmax = float(v.max()) if v.notna().any() else np.nan

    if pd.notna(vmin) and pd.notna(vmax) and vmax > vmin:
        df_plot["marker_size"] = size_min + (v - vmin) / (vmax - vmin) * (size_max - size_min)
    else:
        df_plot["marker_size"] = size_min

    df_plot["marker_size"] = df_plot["marker_size"].fillna(size_min)

    # ---------- 年→色マップ ----------
    palette = px.colors.qualitative.Dark24
    year_to_color = {str(y): palette[i % len(palette)] for i, y in enumerate(years_present)}

    # 凡例は年のみ
    shown_legend_year = set()
    fig = go.Figure()

    # ---------- 描画：線（破線）＋点（Monitoring_Date の順） ----------
    for line_id, g in df_plot.groupby("line_id", sort=False):
        if g.empty:
            continue

        g = g.sort_values("Monitoring_Date").copy()
        year_cat = str(g["Drop_Year_cat"].iloc[0])
        color = year_to_color.get(year_cat, "#1f77b4")

        show_legend = year_cat not in shown_legend_year
        if show_legend:
            shown_legend_year.add(year_cat)

        # 線（控えめ）
        fig.add_trace(go.Scatter(
            x=g["X_total"],
            y=g["Y_total"],
            mode="markers",
            cliponaxis=False,
            line=dict(color=color, width=1.2),
            showlegend=False,
            hoverinfo="skip"
        ))

        # 点（丸）
        fig.add_trace(go.Scatter(
            x=g["X_total"],
            y=g["Y_total"],
            mode="markers",
            name=year_cat,
            legendgroup=year_cat,
            showlegend=show_legend,
            marker=dict(
                symbol="circle",
                size=g["marker_size"],
                color=color,
                opacity=0.85,
                line=dict(color="rgba(0,0,0,0.25)", width=0.8),
            ),
            customdata=np.stack([
                g["Area"].astype(str).values,
                g["Drop_Date"].astype(str).values,
                g["Monitoring_Date"].astype(str).values,
                g["duration_days"].astype(int).values,
                g["overlap_days"].astype(int).values,
                g["X_total_disp"].astype(float).values,
                g["Y_total_disp"].astype(int).values,
                pd.to_numeric(g["shell_mean"], errors="coerce").round(2).values,     # raw shell
                pd.to_numeric(g["shell_filled"], errors="coerce").round(2).values,  # filled shell
            ], axis=1),
            hovertemplate=(
                "Area: %{customdata[0]}<br>"
                "Drop_Date: %{customdata[1]}<br>"
                "Monitoring_Date: %{customdata[2]}<br>"
                "期間（日）: %{customdata[3]}<br>"
                "ラーバ観測日数: %{customdata[4]}<br>"
                "大型ラーバ累積(>=250): %{customdata[5]}<br>"
                "付着数（平均）: %{customdata[6]}<br>"
                "Shell平均(mm): %{customdata[7]}<br>"
                "Shell補完(mm): %{customdata[8]}<br>"
            )
        ))

        # ---------- レイアウト ----------
    fig.update_layout(
        template="plotly_white",
        height=650,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )
    fig.update_xaxes(title_text="大型ラーバ累積（>=250）")
    fig.update_yaxes(title_text="付着数（平均・Place合算）")

    st.plotly_chart(fig, use_container_width=True)

# =========================
# カレンダー部品（wt_test 由来）
# =========================
HEAD_LENGTH_RATIO = 0.55
HEAD_HALF_HEIGHT_RATIO = 0.35
SHAFT_WIDTH_PX = 4.0
OUTLIER_TH = 4.0          # 観測なし時: corr - pred の閾値
OUTLIER_TH_OBS = 2.0      # 観測あり時: corr - obs の閾値
PHYS_MIN, PHYS_MAX = -1.5, 35.0

def get_arrow_svg(direction_deg, speed_mps):
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
    head_length = size * HEAD_LENGTH_RATIO
    head_half_h = size * HEAD_HALF_HEIGHT_RATIO
    line_end = size - head_length
    return f"""
<svg width="{size}" height="{size}" style="display:block;margin:0 auto;transform:rotate({css_angle}deg);">
  <line x1="4" y1="{size/2}" x2="{line_end}" y2="{size/2}"
        stroke="{color}" stroke-width="{SHAFT_WIDTH_PX}" stroke-linecap="round"/>
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
    """
    指定した 'target_depth' 1本だけで週間コメントを作る。
    corr_temp があれば優先、無ければ pred_temp を使う。
    """
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
    idx_first = [i for i in [0, 1, 2] if i < n]
    idx_last = [i for i in [4, 5, 6] if i < n]
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
    """
    利用可能な深さの配列から、浅・中・深の3層代表を返す（10m起算）。
    - 浅: 10m以上の最小値。なければ最浅。
    - 深: 最深。
    - 中: 浅と深の中間の順位（偶数は下側）。
    - 候補が2以下なら、その分だけ返す。
    """
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



# =========================
# 予測カレンダー（wt_test 同等仕様）
# - corr は data/corr の <name>_corr.csv を優先
# - 列名揺れ（CorrTemp 等）を吸収
# =========================
BASE_DIR = base_dir
PRED_DIR = "pred"
OBS_DIR = "obs"
CORR_DIR = "corr"

# 固定パラメータ
RECENT_DAYS = 7           # 直近8日（週間） ※本コードでは8日固定を計算式で表現
OUTLIER_TH = 4.0          # 観測なし時: corr - pred の閾値
OUTLIER_TH_OBS = 2.0      # 観測あり時: corr - obs の閾値
OBS_MATCH_TOL_MIN = 60    # 観測近傍マージ許容（分）
CORR_MATCH_TOL_MIN = 60   # 補正近傍マージ許容（分）
TEMP_MIN, TEMP_MAX = -2.0, 40.0
PHYS_MIN, PHYS_MAX = -1.5, 35.0
HIGH_TEMP_TH = 22.0       # コメント用
RANGE_STABLE = 0.5
DELTA_THRESH = 0.3
DISPLAY_MODE = "arrow"

# === 追加：今日（JST）基準と“未来8日”ウィンドウの明示フラグ ===
WEEK_WINDOW_FORWARD = True  # True: 今日→先7日（計8日）、False: 過去7日→今日（計8日）

def pjoin(*parts: str) -> str:
    return os.path.normpath(os.path.join(*parts))

# =========================================
# ユーティリティ
# =========================================
def _pick_series_corr_then_pred(g: pd.DataFrame) -> Optional[pd.Series]:
    """
    corr が列として存在し、かつ有効値が1つ以上あれば corr を採用。
    そうでなければ pred。どちらもダメなら None。
    """
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
    """UTCとして解釈 → JSTへ変換 → タイムゾーン情報を外す（naive）"""
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    dt = dt.dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)
    return dt
def jst_to_naive(s: pd.Series) -> pd.Series:
    """ローカル／JST相当の文字列→pandas日時→（もしtz付きなら）JSTへ変換→naive化"""
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
    """
    depth_m ごとに nearest で asof マージする。
    右側にデータが無い深さは NaN をパディング、左側の行は保持（keep-left）。
    """
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
    """
    ['corr','temp'] のようなキーワード候補から最も合致する列名を推定する。
    完全一致 → 正規化（_除去・小文字化）包含 の順で探索。
    """
    cols = list(df.columns)
    # 完全一致
    for c in cols:
        if c.lower() in [k.lower() for k in keywords]:
            return c
    # 正規化（_ 除去）
    norm = {c: c.lower().replace("_", "") for c in cols}
    for c, n in norm.items():
        ok = all(k.lower().replace("_", "") in n for k in keywords)
        if ok:
            return c
    return None
def to_rgba(color: str, alpha: float = 0.18) -> str:
    """
    '#rrggbb' / 'rgb(r,g,b)' / 'rgba(r,g,b,a)' を RGBA 文字列に正規化し、alpha を差し替える。
    不正値は緑系のデフォルトを返す。
    """
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

# ---- キャッシュ無効化用：ファイル指紋 ----
def file_fingerprint(path: str) -> str:
    """
    任意パスの存在/mtime/サイズを文字列化（キャッシュキー用）。
    存在しなければ 'missing'。
    """
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

# =========================================
# ローダ（fp をキーに追加）
# =========================================
@st.cache_data(show_spinner=False)
def load_pred(filename: str, fp: str = "") -> pd.DataFrame:
    """
    予測（pred）CSV を読み込む。
    fp はキャッシュキー用（中身では使わない）。
    """
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
    """
    補正（corr）CSV を読み込む（<name>_corr.csv）。
    fp はキャッシュキー用（中身では使わない）。
    """
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
    """
    観測（obs）CSV を読み込む。
    fp はキャッシュキー用（中身では使わない）。
    """
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
def add_corr(df_pred: pd.DataFrame, df_corr: pd.DataFrame) -> pd.DataFrame:
    """
    pred へ corr を depth_m&datetime で近傍（±CORR_MATCH_TOL_MIN 分）マージし、
    corr_temp（＋あれば corr_low / corr_high）を付加する。
    corr が空なら pred の行をそのまま返し、corr_* 列だけ NaN で補う。
    """
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
    """モバイル向けにヘッダー余白を圧縮するCSS。"""
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
    head_length = size * HEAD_LENGTH_RATIO
    head_half_h = size * HEAD_HALF_HEIGHT_RATIO
    line_end = size - head_length
    return f"""
<svg width="{size}" height="{size}" style="display:block;margin:0 auto;transform:rotate({css_angle}deg);">
  <line x1="4" y1="{size/2}" x2="{line_end}" y2="{size/2}"
        stroke="{color}" stroke-width="{SHAFT_WIDTH_PX}" stroke-linecap="round"/>
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
    """
    指定した 'target_depth' 1本だけで週間コメントを作る。
    corr_temp があれば優先、無ければ pred_temp を使う。
    """
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
    idx_first = [i for i in [0, 1, 2] if i < n]
    idx_last = [i for i in [4, 5, 6] if i < n]
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
    """
    利用可能な深さの配列から、浅・中・深の3層代表を返す（10m起算）。
    - 浅: 10m以上の最小値。なければ最浅。
    - 深: 最深。
    - 中: 浅と深の中間の順位（偶数は下側）。
    - 候補が2以下なら、その分だけ返す。
    """
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
    """水温（wt_test系のシンプル版）

    - corr/<name>_corr.csv があれば corr を主表示、pred は薄く補助表示
    - corr が無ければ pred を主表示
    - obs があれば点で重ねる
    - GSI は ON/OFF のみ（エリアは水温で選んだファイル名と一致するものを自動表示）
    - サイドバーは使わない
    """

    # ---------- エリア（predファイル）選択 ----------
    pred_folder = pjoin(BASE_DIR, PRED_DIR)
    if not os.path.exists(pred_folder):
        st.error(f"フォルダが見つかりません: {pred_folder}")
        st.stop()

    pred_files = [f for f in os.listdir(pred_folder) if f.lower().endswith(".csv")]
    if not pred_files:
        st.warning("pred に CSV がありません")
        st.stop()

    selected_file = st.selectbox("対象エリアを選択", sorted(pred_files), key="water_selected_file", label_visibility="collapsed")
    # ---------- 読み込み（pred/corr/obs） ----------
    pred_path = pjoin(BASE_DIR, PRED_DIR, selected_file)
    name, ext = os.path.splitext(selected_file)
    corr_path = pjoin(BASE_DIR, CORR_DIR, f"{name}_corr{ext}")
    obs_path  = pjoin(BASE_DIR, OBS_DIR, selected_file)

    fp_pred = file_fingerprint(pred_path)
    fp_corr = file_fingerprint(corr_path)
    fp_obs  = file_fingerprint(obs_path)

    df_pred = load_pred(selected_file, fp_pred)
    df_corr = load_corr_for(selected_file, fp_corr)
    df_obs  = load_obs_for(selected_file, fp_obs)

    if df_pred.empty:
        st.warning("予測データが読み込めませんでした")
        st.stop()

    corr_available = (df_corr is not None) and (not df_corr.empty)
    obs_available  = (df_obs  is not None) and (not df_obs.empty)

    # ---------- 期間選択（直近1か月 / 任意期間） ----------
    try:
        period_mode = st.segmented_control(
            "", options=["直近1か月", "任意期間"], default="直近1か月", key="water_period_mode"
        )
    except Exception:
        period_mode = st.radio(
            "", ["直近1か月", "任意期間"], index=0, horizontal=True,
            key="water_period_mode_radio", label_visibility="collapsed"
        )

    latest_dt = pd.to_datetime(df_pred["datetime"], errors="coerce").max()
    days = sorted(df_pred["date_day"].dropna().unique()) if "date_day" in df_pred.columns else []
    min_day = min(days) if days else latest_dt.date()
    max_day = max(days) if days else latest_dt.date()

    if period_mode == "直近1か月":
        end_day = latest_dt.date()
        start_day = (latest_dt - pd.Timedelta(days=29)).date()
    else:
        start_default = max(min_day, max_day - pd.Timedelta(days=29))
        start_day, end_day = st.slider(
            "", min_value=min_day, max_value=max_day,
            value=(start_default, max_day),
            key="water_period_slider", label_visibility="collapsed"
        )

    # ---------- 水深選択 ----------
    depths_all = sorted(pd.to_numeric(df_pred.get("depth_m"), errors="coerce").dropna().astype(int).unique().tolist())
    default_depths = depths_all[:min(3, len(depths_all))]

    # corr+obs が揃う深度があれば優先
    if corr_available:
        d_corr = set(pd.to_numeric(df_corr.get("depth_m"), errors="coerce").dropna().astype(int).unique().tolist())
        if obs_available:
            d_obs = set(pd.to_numeric(df_obs.get("depth_m"), errors="coerce").dropna().astype(int).unique().tolist())
            both = sorted(d_corr.intersection(d_obs))
            if both:
                default_depths = both[:min(3, len(both))]
        else:
            dc = sorted(d_corr)
            if dc:
                default_depths = dc[:min(3, len(dc))]

    sel_depths = st.multiselect(
        "水深（複数可）", options=depths_all, default=default_depths, key="water_depths"
    )

    # ---------- 期間フィルタ ----------
    dfp = df_pred[(df_pred["date_day"] >= start_day) & (df_pred["date_day"] <= end_day)].copy()
    dfc = df_corr[(df_corr["date_day"] >= start_day) & (df_corr["date_day"] <= end_day)].copy() if corr_available else pd.DataFrame()
    dfo = df_obs [(df_obs ["date_day"] >= start_day) & (df_obs ["date_day"] <= end_day)].copy() if obs_available else pd.DataFrame()

    # ---------- 描画 ----------
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    base_colors = px.colors.qualitative.Dark24

    # pred/corr/obs
    for i, d in enumerate(sel_depths):
        d_int = int(d)
        col = base_colors[i % len(base_colors)]

        gp = dfp[pd.to_numeric(dfp["depth_m"], errors="coerce").fillna(-9999).astype(int) == d_int]
        gc = dfc[pd.to_numeric(dfc.get("depth_m"), errors="coerce").fillna(-9999).astype(int) == d_int] if not dfc.empty else pd.DataFrame()
        go_ = dfo[pd.to_numeric(dfo.get("depth_m"), errors="coerce").fillna(-9999).astype(int) == d_int] if not dfo.empty else pd.DataFrame()

        has_corr_line = (not gc.empty) and ("corr_temp" in gc.columns) and (pd.to_numeric(gc["corr_temp"], errors="coerce").notna().any())

        # corr があれば主表示（太線）
        if has_corr_line:
            # 可能なら帯
            if ("corr_low" in gc.columns) and ("corr_high" in gc.columns):
                ylow = pd.to_numeric(gc["corr_low"], errors="coerce")
                yhigh = pd.to_numeric(gc["corr_high"], errors="coerce")
                fig.add_trace(
                    go.Scatter(x=gc["datetime"], y=ylow, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"),
                    secondary_y=False
                )
                fig.add_trace(
                    go.Scatter(x=gc["datetime"], y=yhigh, mode="lines", line=dict(width=0), fill="tonexty",
                               fillcolor=to_rgba(col, 0.12), showlegend=False, hoverinfo="skip"),
                    secondary_y=False
                )

            ycorr = pd.to_numeric(gc["corr_temp"], errors="coerce")
            fig.add_trace(
                go.Scatter(x=gc["datetime"], y=ycorr, mode="lines", name=f"{d_int}m 補正",
                           line=dict(color=col, width=3.0)),
                secondary_y=False
            )

            # pred は薄く補助
            if not gp.empty and ("pred_temp" in gp.columns):
                ypred = pd.to_numeric(gp["pred_temp"], errors="coerce")
                fig.add_trace(
                    go.Scatter(x=gp["datetime"], y=ypred, mode="lines", name=f"{d_int}m 予測",
                               line=dict(color=col, width=1.2, dash="dot"), opacity=0.35, showlegend=False),
                    secondary_y=False
                )

        # corr が無い場合は pred を主表示
        else:
            if not gp.empty and ("pred_temp" in gp.columns):
                ypred = pd.to_numeric(gp["pred_temp"], errors="coerce")
                fig.add_trace(
                    go.Scatter(x=gp["datetime"], y=ypred, mode="lines", name=f"{d_int}m 予測",
                               line=dict(color=col, width=2.0)),
                    secondary_y=False
                )

        # obs 点
        if not go_.empty and ("obs_temp" in go_.columns):
            yobs = pd.to_numeric(go_["obs_temp"], errors="coerce")
            fig.add_trace(
                go.Scatter(x=go_["datetime"], y=yobs, mode="markers", name=f"{d_int}m 実測",
                           marker=dict(size=6, color=col, line=dict(color="black", width=0.2)), opacity=0.75, showlegend=False),
                secondary_y=False
            )


    fig.update_layout( height=550,
        margin=dict(l=10, r=10, t=90, b=10),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="水温(℃)", secondary_y=False)
    fig.update_yaxes(title_text="GSI", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)
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
        cal_choice = st.segmented_control(
            "", options=["週間表示（昼頃）", "選択日（1時間毎）"],
            default="週間表示（昼頃）", key='cal_choice'
        )
    except Exception:
        cal_choice = st.radio(
            "", ["週間表示（昼頃）", "選択日（1時間毎）"],
            index=0, horizontal=True, key='cal_choice_radio',
            label_visibility='collapsed'
        )

    # ===== 週間（昼頃） =====
    if cal_choice == "週間表示（昼頃）":
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

    # ===== 選択日（1時間毎） =====
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
    import re
    import folium
    from streamlit_folium import folium_static
    from datetime import datetime, timedelta

    AREA_FILE   = pjoin(base_dir, "file_summary.csv")
    GSI_FILE    = MATURITY_PATH 
    LARVAE_FILE = LARVAE_PATH 

    # 読み込み（既存ヘルパがあれば使用）
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
        st.warning('file_summary.csv / maturity.csv / larvae.csv を配置してください。')
        return

    # 前処理
    df_area['Area'] = df_area.get('Area', '').astype(str).str.strip()
    for col in ['Laf','Lof']:
        if col in df_area.columns:
            df_area[col] = pd.to_numeric(df_area[col], errors='coerce')

    df_gsi['Date'] = pd.to_datetime(df_gsi['Date'], errors='coerce')
    df_gsi['Area'] = df_gsi.get('Area','').astype(str).str.strip()
    df_gsi['GSI']  = pd.to_numeric(df_gsi['GSI'], errors='coerce')
    df_gsi['Sex']  = df_gsi.get('Sex','Unknown').astype(str).str.strip().str.upper()
    df_gsi['Year'] = df_gsi['Date'].dt.year
    df_gsi['week'] = df_gsi['Date'].dt.isocalendar().week.astype(int)

    df_larv['Date'] = pd.to_datetime(df_larv['Date'], errors='coerce')
    df_larv['Area'] = df_larv.get('Area','').astype(str).str.strip()
    df_larv['Year'] = df_larv['Date'].dt.year
    df_larv['week'] = df_larv['Date'].dt.isocalendar().week.astype(int)

    # サイドバー（廃止：メイン画面に移動）

    # UI（メイン画面）

    c1, c2 = st.columns([2.0, 1.0])

    with c1:

        mode = st.radio('', ['GSI', 'ラーバ'], index=0, key='map_mode', horizontal=True, label_visibility='collapsed')


    years_all = sorted(set(df_gsi['Year'].dropna().unique()).union(set(df_larv['Year'].dropna().unique())))

    if not years_all:

        st.info('年度データがありません。')

        return


    with c2:

        sel_year = st.selectbox('', years_all, index=len(years_all)-1, key='map_year', label_visibility='collapsed')


    base_df = df_gsi if mode == 'GSI' else df_larv

    weeks_all = sorted(base_df[base_df['Year'] == sel_year]['week'].dropna().astype(int).unique().tolist())

    if not weeks_all:

        st.info('選択年度に週データがありません。')

        return


    def week_range(year:int, week:int):

        s = datetime.fromisocalendar(int(year), int(week), 1)

        e = s + timedelta(days=6)

        return s, e


    week_labels = []

    for w in weeks_all:

        sdt, edt = week_range(sel_year, int(w))

        week_labels.append((int(w), f'（{sdt:%m/%d}〜{edt:%m/%d}）'))


    options = [lab for _, lab in week_labels]

    n = len(options)


    if 'map_week_idx' not in st.session_state:

        st.session_state['map_week_idx'] = max(0, n - 1)


    st.session_state['map_week_idx'] = min(max(int(st.session_state['map_week_idx']), 0), max(0, n - 1))


    # 週（年・週の後に前週/翌週ボタン）

    # 仕様：前週/翌週は segmented_control（無ければ radio）に統一（ボタンには戻さない）。
    # Streamlit 制約回避：週selectboxの key は map_week_idx_widget、内部の真実は map_week_idx。

    def _clamp_idx(i: int) -> int:
        return min(max(int(i), 0), max(0, n - 1))

    st.session_state.setdefault('map_week_idx', max(0, n - 1))
    st.session_state['map_week_idx'] = _clamp_idx(st.session_state.get('map_week_idx', 0))
    st.session_state.setdefault('map_week_idx_widget', st.session_state['map_week_idx'])
    st.session_state['map_week_idx_widget'] = _clamp_idx(st.session_state.get('map_week_idx_widget', 0))
    st.session_state.setdefault('map_week_nav', None)

    def _sync_week_from_widget():
        st.session_state['map_week_idx'] = _clamp_idx(st.session_state.get('map_week_idx_widget', 0))

    def _nav_week():
        nav = st.session_state.get('map_week_nav', None)
        cur = _clamp_idx(st.session_state.get('map_week_idx', 0))
        if nav == '前週':
            cur = _clamp_idx(cur - 1)
        elif nav == '翌週':
            cur = _clamp_idx(cur + 1)
        st.session_state['map_week_idx'] = cur
        st.session_state['map_week_idx_widget'] = cur
        # 2択は連打しやすいよう None に戻してボタン風にする
        st.session_state['map_week_nav'] = None

    w_sel, w_nav = st.columns([2.2, 1.6])

    with w_sel:
        st.selectbox(
            '', options=list(range(n)),
            format_func=lambda i: options[i],
            key='map_week_idx_widget',
            on_change=_sync_week_from_widget,
            label_visibility='collapsed'
        )

    with w_nav:
        try:
            st.segmented_control('', options=['前週','翌週'], key='map_week_nav', on_change=_nav_week)
        except Exception:
            _opts = ['前週','翌週']
            _cur = st.session_state.get('map_week_nav', None)
            _idx = _opts.index(_cur) if _cur in _opts else 0
            st.radio('', _opts, index=_idx, key='map_week_nav', horizontal=True,
                     label_visibility='collapsed', on_change=_nav_week)

    sel_week = week_labels[int(st.session_state['map_week_idx'])][0] if n > 0 else None
    sel_label = week_labels[int(st.session_state['map_week_idx'])][1] if n > 0 else ''




# 対象Area（モードごと）
    if mode == 'GSI':
        sub_week_gsi = df_gsi[(df_gsi['Year']==sel_year) & (df_gsi['week']==sel_week)].copy()
        areas_with_data = sorted(sub_week_gsi['Area'].dropna().astype(str).unique().tolist())
    else:
        sub_week_larv = df_larv[(df_larv['Year']==sel_year) & (df_larv['week']==sel_week)].copy()
        size_cols = [c for c in sub_week_larv.columns if str(c).isdigit()]
        # ≥200 の合計が正のエリアのみ（コメント対象）
        areas_with_data = []
        if size_cols:
            sw = sub_week_larv.copy()
            for c in size_cols:
                sw[c] = pd.to_numeric(sw[c], errors='coerce').fillna(0.0)
            # 200–259 + ≥260
            ge200 = sw[[c for c in size_cols if int(c)>=200]].sum(axis=1).sum()
            # ただし Area 別判定
            for area, g in sw.groupby('Area'):
                total_ge200 = float(g[[c for c in size_cols if int(c)>=200]].sum().sum())
                if total_ge200 > 0:
                    areas_with_data.append(str(area))
            areas_with_data = sorted(set(areas_with_data))
        else:
            areas_with_data = sorted(sub_week_larv['Area'].dropna().astype(str).unique().tolist())

    # 地図中心
    if areas_with_data:
        df_center = df_area[df_area['Area'].isin(areas_with_data)]
        center_lat = float(df_center['Laf'].mean())
        center_lon = float(df_center['Lof'].mean())
    else:
        center_lat = float(df_area['Laf'].mean())
        center_lon = float(df_area['Lof'].mean())

    m = folium.Map(location=[center_lat, center_lon], zoom_start=8, max_bounds=True)

    # --- 表示対象（areas_with_data があればそれだけ、無ければ全エリア）から bounds を作る ---
    df_bounds = df_area[df_area['Area'].isin(areas_with_data)].copy() if areas_with_data else df_area.copy()
    df_bounds = df_bounds.dropna(subset=['Laf', 'Lof'])

    if not df_bounds.empty:
        min_lat, max_lat = float(df_bounds['Laf'].min()), float(df_bounds['Laf'].max())
        min_lon, max_lon = float(df_bounds['Lof'].min()), float(df_bounds['Lof'].max())

    # 緯度経度が同一点（1地点のみ）の場合にも対応：少しだけ余白を付ける
        pad_lat = max(0.02, (max_lat - min_lat) * 0.10)
        pad_lon = max(0.02, (max_lon - min_lon) * 0.10)

        m.fit_bounds([[min_lat - pad_lat, min_lon - pad_lon],
                      [max_lat + pad_lat, max_lon + pad_lon]])

    # 円グラフ色と凡例
    colors_gsi    = ['#d62728', '#ff7f0e', '#1f77b4']  # ≥25 / 20–24.9 / <20
    colors_larvae = ['#1f77b4', '#ff7f0e', '#d62728']  # <200 / 200–259 / ≥260
    legend_gsi = """
    <div style='background:white;padding:8px 12px;border:1px solid #ccc;border-radius:6px;font-size:13px;min-width:120px;margin-top:8px;'>
      <b>凡例 (GSI):</b><br>
      <span style='display:inline-block;width:12px;height:12px;background:#d62728;margin-right:4px;'></span> ≥25<br>
      <span style='display:inline-block;width:12px;height:12px;background:#ff7f0e;margin-right:4px;'></span> 20–24.9<br>
      <span style='display:inline-block;width:12px;height:12px;background:#1f77b4;margin-right:4px;'></span> <20
    </div>
    """
    legend_larvae = """
    <div style='background:white;padding:8px 12px;border:1px solid #ccc;border-radius:6px;font-size:13px;min-width:120px;margin-top:8px;'>
      <b>凡例 (ラーバ):</b><br>
      <span style='display:inline-block;width:12px;height:12px;background:#d62728;margin-right:4px;'></span> ≥260<br>
      <span style='display:inline-block;width:12px;height:12px;background:#ff7f0e;margin-right:4px;'></span> 200–259<br>
      <span style='display:inline-block;width:12px;height:12px;background:#1f77b4;margin-right:4px;'></span> <200
    </div>
    """

    # 円グラフSVG（既存）
    def svg_pie(values, colors, size=60, labels=None):
        total = float(sum(values))
        if total <= 0:
            return ''
        nonzeros = [(i, v) for i, v in enumerate(values) if v > 0]
        cx, cy, r = size/2, size/2, size/2 - 2
        if len(nonzeros) == 1:
            i, _ = nonzeros[0]
            c = colors[i] if colors and i < len(colors) else '#888888'
            title_text = f"{labels[i]}: {values[i]}" if labels else f"値: {values[i]}"
            svg = f"""
            <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
              <circle cx="{cx}" cy="{cy}" r="{r}" fill="{c}" stroke="#fff" stroke-width="2">
                <title>{title_text}</title>
              </circle>
            </svg>
            """.strip()
            return svg
        angles = [(v / total) * 360.0 for v in values]
        svg = f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
        start_angle = 0.0
        EPS = 1e-9
        for i, ang in enumerate(angles):
            if ang <= EPS:
                continue
            end_angle = start_angle + ang
            x1 = cx + r * np.cos(np.radians(start_angle))
            y1 = cy + r * np.sin(np.radians(start_angle))
            x2 = cx + r * np.cos(np.radians(end_angle))
            y2 = cy + r * np.sin(np.radians(end_angle))
            large_arc = 1 if ang > 180.0 else 0
            c = colors[i] if colors and i < len(colors) else '#888888'
            title_text = f"{labels[i]}: {values[i]}" if labels else f"値: {values[i]}"
            path = f"M{cx},{cy} L{x1},{y1} A{r},{r} 0 {large_arc},1 {x2},{y2} Z"
            svg += f'<path d="{path}" fill="{c}" stroke="#fff" stroke-width="2"><title>{title_text}</title></path>'
            start_angle = end_angle
        svg += "</svg>"
        return svg

    # ========== コメント（地図の「上」に表示） ==========
    # --- コメント強調（色付け） ---

    def emphasize_map_lines(lines):
        out = []
        for l in lines:
            s = str(l)

            # --- GSI の上昇/下降を色付け（部分一致に変更） ---
            if ('GSI（' in s) and ('上昇' in s or '下降' in s):
                # 下降：青字
                s = s.replace('下降', "<span style='color:#1976D2; font-weight:700;'>下降</span>")
                # 上昇：オレンジ字（必要に応じて赤へ）
                s = s.replace('上昇', "<span style='color:#FF8F00; font-weight:700;'>上昇</span>")
                # 変化なしはそのまま（必要なら薄灰などに）

            # --- ラーバ ≥200μm の行は丸ごと赤太字 ---
            if ('ラーバ ≥200μm 合計' in s) or ('・ラーバ ≥200μm :' in s):
                s = "<span style='color:#D32F2F; font-weight:700;'>" + s + "</span>"

            # --- 前週比（≥200）は 増加=赤／減少=青（ラーバ用） ---
            if ('前週比（≥200）' in s) or ('・前週比（≥200）' in s):
                s = s.replace('増加', "<span style='color:#D32F2F; font-weight:700;'>増加</span>")
                s = s.replace('減少', "<span style='color:#1976D2; font-weight:700;'>減少</span>")
            out.append(s)
        return out

    with st.expander('コメント', expanded=False):  # comment_box を使わない方が安全
        # GSIモード → GSIがあるAreaのみ / ラーバモード → ≥200合計が正のAreaのみ
        if not areas_with_data:
            st.caption("（該当するデータがありません）")
        else:
            for area in areas_with_data:
                lines = []
                if mode == 'GSI':
                    g = df_gsi[(df_gsi['Area'] == area) & (df_gsi['Year'] == sel_year) & (df_gsi['week'] == sel_week)].copy()
                    if not g.empty:
                        cur_F = float(pd.to_numeric(g.loc[g['Sex'] == 'F', 'GSI'], errors='coerce').mean())
                        cur_M = float(pd.to_numeric(g.loc[g['Sex'] == 'M', 'GSI'], errors='coerce').mean())
                        prev_week = max(int(sel_week) - 1, 1)
                        p = df_gsi[(df_gsi['Area'] == area) & (df_gsi['Year'] == sel_year) & (df_gsi['week'] == prev_week)]
                        prev_F = float(pd.to_numeric(p.loc[p['Sex'] == 'F', 'GSI'], errors='coerce').mean()) if not p.empty else np.nan
                        prev_M = float(pd.to_numeric(p.loc[p['Sex'] == 'M', 'GSI'], errors='coerce').mean()) if not p.empty else np.nan
                        yprev = df_gsi[(df_gsi['Area'] == area) & (df_gsi['Year'] == sel_year - 1) & (df_gsi['week'] == sel_week)]
                        prevY_F = float(pd.to_numeric(yprev.loc[yprev['Sex'] == 'F', 'GSI'], errors='coerce').mean()) if not yprev.empty else np.nan
                        prevY_M = float(pd.to_numeric(yprev.loc[yprev['Sex'] == 'M', 'GSI'], errors='coerce').mean()) if not yprev.empty else np.nan
                        def trend(cur, base, eps=2.0):
                            if pd.isna(cur) or pd.isna(base): return "データ不足"
                            d = cur - base
                            if d > eps: return "上昇"
                            if d < -eps: return "下降"
                            return "変化なし"
                        lines.append(f"・GSI（F）: 平均 {cur_F:.2f}（{trend(cur_F, prev_F if not pd.isna(prev_F) else prevY_F)}）")
                        lines.append(f"・GSI（M）: 平均 {cur_M:.2f}（{trend(cur_M, prev_M if not pd.isna(prev_M) else prevY_M)}）")
                else:
                    g = df_larv[(df_larv['Area'] == area) & (df_larv['Year'] == sel_year) & (df_larv['week'] == sel_week)].copy()
                    size_cols = [c for c in g.columns if str(c).isdigit()]
                    if size_cols:
                        for c in size_cols:
                            g[c] = pd.to_numeric(g[c], errors='coerce').fillna(0.0)
                        qty_200_259 = float(g[[c for c in size_cols if 200 <= int(c) <= 259]].sum().sum())
                        qty_ge260   = float(g[[c for c in size_cols if int(c) >= 260]].sum().sum())
                        qty_ge200   = qty_200_259 + qty_ge260
                        if qty_ge200 > 0:
                            prev_week = max(int(sel_week) - 1, 1)
                            p  = df_larv[(df_larv['Area'] == area) & (df_larv['Year'] == sel_year) & (df_larv['week'] == prev_week)].copy()
                            py = df_larv[(df_larv['Area'] == area) & (df_larv['Year'] == sel_year - 1) & (df_larv['week'] == sel_week)].copy()
                            p_qty_ge200  = np.nan
                            py_qty_ge200 = np.nan
                            if not p.empty:
                                for c in [c for c in p.columns if str(c).isdigit()]:
                                    p[c] = pd.to_numeric(p[c], errors='coerce').fillna(0.0)
                                p_qty_ge200 = float(p[[c for c in p.columns if c.isdigit() and int(c) >= 200]].sum().sum())
                            if not py.empty:
                                for c in [c for c in py.columns if str(c).isdigit()]:
                                    py[c] = pd.to_numeric(py[c], errors='coerce').fillna(0.0)
                                py_qty_ge200 = float(py[[c for c in py.columns if c.isdigit() and int(c) >= 200]].sum().sum())
                            diff_prev  = qty_ge200 - p_qty_ge200  if not pd.isna(p_qty_ge200)  else np.nan
                            diff_prevY = qty_ge200 - py_qty_ge200 if not pd.isna(py_qty_ge200) else np.nan
                            lines.append(f"・ラーバ ≥200μm : 合計 {int(qty_ge200)}（内訳 200–259: {int(qty_200_259)} / ≥260: {int(qty_ge260)}）")
                            if not pd.isna(diff_prev):
                                lines.append(f"・前週比（≥200）: {'増加' if diff_prev > 0 else ('減少' if diff_prev < 0 else '変化なし')}（差 {int(diff_prev)}）")

            # 10m水温（DR）も添える（常にArea.csv）（※コメントは値がある場合のみ）
                cur_t = prev_t = prevY_t = np.nan
                try:
                    df_dr = load_dr_single_file(base_dir, area)
                except Exception:
                    df_dr = pd.DataFrame()
                if not df_dr.empty:
                    dt = pd.to_datetime(df_dr["datetime"], errors='coerce')
                    df_dr["week"] = dt.dt.isocalendar().week.astype(int)
                    df_dr["year"] = dt.dt.year
                    df_10 = df_dr[(df_dr["depth_m"] == 10) & (df_dr["year"] == sel_year) & (df_dr["week"] == sel_week)].copy()
                    cur_t = float(pd.to_numeric(df_10.get("pred_temp"), errors='coerce').mean()) if not df_10.empty else np.nan
                    if not pd.isna(cur_t):
                        prev_week = max(int(sel_week) - 1, 1)
                        p10 = df_dr[(df_dr["depth_m"] == 10) & (df_dr["year"] == sel_year) & (df_dr["week"] == prev_week)].copy()
                        y10 = df_dr[(df_dr["depth_m"] == 10) & (df_dr["year"] == (sel_year - 1)) & (df_dr["week"] == sel_week)].copy()
                        prev_t  = float(pd.to_numeric(p10.get("pred_temp"), errors='coerce').mean()) if not p10.empty else np.nan
                        prevY_t = float(pd.to_numeric(y10.get("pred_temp"), errors='coerce').mean()) if not y10.empty else np.nan
                        def ttrend(cur, base, eps_t=0.3):
                            if pd.isna(cur) or pd.isna(base): return "データ不足"
                            d = cur - base
                            if d > eps_t: return "上昇"
                            if d < -eps_t: return "下降"
                            return "変化なし"
                        diffY_txt = (f"{(cur_t - prevY_t):+.1f}℃" if not pd.isna(prevY_t) else "データ不足")
                        lines.append(f"・水温（10m）: 平均 {cur_t:.1f}℃／前週比 {ttrend(cur_t, prev_t)}／前年同期差 {diffY_txt}")

                # コメントが1つ以上ある場合のみ表示（空なら完全に非表示）
                if lines:
                    em_lines = emphasize_map_lines(lines)
                    st.markdown(
                        f"<div><b>{area}</b> {sel_label}</div>"
                        + "".join([f"<p style='margin:2px 0 0 0;'>{s}</p>" for s in em_lines]),
                        unsafe_allow_html=True
                    )


    # ========== 地図の描画（マーカーは円グラフのみ） ==========
    for _, row in df_area.iterrows():
        area = str(row.get('Area')).strip()
        if area == '' or (areas_with_data and area not in areas_with_data):
            continue
        lat, lon = float(row.get('Laf')), float(row.get('Lof'))
        if mode == 'GSI':
            sub = df_gsi[(df_gsi['Area']==area) & (df_gsi['Year']==sel_year) & (df_gsi['week']==sel_week)].copy()
            n25   = int((sub['GSI']>=25).sum())
            n20_  = int(((sub['GSI']>=20) & (sub['GSI']<25)).sum())
            nlt20 = int((sub['GSI']<20).sum())
            values = [n25, n20_, nlt20]; labels = ['≥25','20–24.9','<20']
            colors = colors_gsi; size = 60; area_label = area
        else:
            sub = df_larv[(df_larv['Area']==area) & (df_larv['Year']==sel_year) & (df_larv['week']==sel_week)].copy()
            size_cols = [c for c in sub.columns if str(c).isdigit()]
            for c in size_cols:
                sub[c] = pd.to_numeric(sub[c], errors='coerce').fillna(0)
            lt200 = float(sub[[c for c in size_cols if int(c)<200]].sum().sum()) if size_cols else 0.0
            mid   = float(sub[[c for c in size_cols if 200<=int(c)<=259]].sum().sum()) if size_cols else 0.0
            ge260 = float(sub[[c for c in size_cols if int(c)>=260]].sum().sum()) if size_cols else 0.0
            total = lt200 + mid + ge260
            values = [lt200, mid, ge260]; labels = ['<200','200–259','≥260']
            colors = colors_larvae
            size   = int(min(150, max(30, np.sqrt(total)*10))) if total>0 else 60
            area_label = f"{area}（{int(total)}）"
        svg = svg_pie(values, colors, size=size, labels=labels)
        html = f"<div style='text-align:center;'><div style='font-weight:bold;'>{area_label}</div>{svg}</div>"
        folium.Marker(location=[lat, lon], icon=folium.DivIcon(html=html)).add_to(m)

    folium_static(m, height=350)
    st.markdown(legend_gsi if mode=='GSI' else legend_larvae, unsafe_allow_html=True)


    
def reset_sidebar_state_for(prefix_keep: str):
    """
    現在のモード用 prefix 以外のサイドバー関連セッションキーを掃除する。
    例: prefix_keep='map_' なら map_ 以外（sc_, larv_, water_, cal_ など）を削除。
    """
    prefixes = ('map_', 'sc_', 'larv_', 'yc_', 'water_', 'cal_')
    # list(...) でコピーしながら走査（削除によるRuntimeError回避）
    for k in list(st.session_state.keys()):
        # 既知の接頭辞に一致 かつ 「保持したい prefix」ではない
        if k.startswith(prefixes) and not k.startswith(prefix_keep):
            try:
                del st.session_state[k]
            except KeyError:
                # 競合・同時削除が起きても無害化
                pass

def main():

    try:
        inject_compact_css()
    except Exception:
        pass
    # カラムを使わずフル幅で表示
    try:
        mode = st.segmented_control(
            '',
            options=["カレンダー", "水温", "地図", "ラーバ", "経年比較"],
            key="main_mode",
            default="カレンダー",
            label_visibility="collapsed"
        )
    except Exception:
        mode = st.radio(
            '',
            options=["カレンダー", "水温", "地図", "ラーバ", "経年比較"],
            index=0,
            horizontal=True,
            key="main_mode",
            label_visibility="collapsed"
        )

    # 以降は既存の分岐のままでOK
    # すべてのArea候補（必要モードのみで使用）
    # ---- サイドバーUI（条件表示）----
    sel_areas = None
    with st.sidebar:
        pass

    # ---- 分岐（サイドバーの残留掃除を各モード直前に実施）----
    if mode == "水温":
        reset_sidebar_state_for('water_')
        render_water_mode()

    elif mode == "ラーバ":
        reset_sidebar_state_for('larv_')
        render_larvae_mode(sel_areas)

    elif mode == "経年比較":
        reset_sidebar_state_for('yc_')
        render_yearly_compare_mode()

    elif mode == "地図":
        reset_sidebar_state_for('map_')
        render_map_mode()

    else:  # "カレンダー"
        reset_sidebar_state_for('cal_')
        render_calendar_mode()

if __name__ == "__main__":
    main()
