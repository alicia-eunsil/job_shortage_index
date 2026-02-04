# ===================================
# 0) Packages
# ===================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pandas.tseries.offsets import MonthEnd, MonthBegin
import streamlit as st
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

# ===================================
# 1) CSV 데이터 불러오기  (★ 여기 포함 / ★ 버그 수정)
# ===================================
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "sample_2.csv"    # ✅ 추가
df_raw = pd.read_csv(CSV_PATH)                 # ✅ 수정 (CSV_PATH가 정의되어 있어야 함)

def set_korean_font():
    font_path = BASE_DIR / "fonts" / "NanumGothic.ttf"

    plt.rcParams["axes.unicode_minus"] = False

    # 폰트 없거나, 너무 작으면(=LFS 포인터/깨짐) 스킵
    if not font_path.exists():
        return
    if font_path.stat().st_size < 10_000:   # 10KB 미만이면 폰트일 가능성 거의 없음
        return

    try:
        fm.fontManager.addfont(str(font_path))
        font_name = fm.FontProperties(fname=str(font_path)).get_name()
        plt.rcParams["font.family"] = font_name
    except Exception:
        # 폰트 로딩 실패해도 앱은 계속 실행
        return

set_korean_font()



# 필수 컬럼 체크
required_cols = ["sigungu", "ksic1_code", "jobbig_code", "mdate", "S", "A", "L"]
missing = [c for c in required_cols if c not in df_raw.columns]
if missing:
    raise ValueError(f"CSV에 필수 컬럼이 없습니다: {missing}")

# 타입 정리
df_raw["mdate"] = pd.to_datetime(df_raw["mdate"])
df_raw["ksic1_code"] = df_raw["ksic1_code"].astype("Int64")
df_raw["jobbig_code"] = df_raw["jobbig_code"].astype("Int64")
for c in ["S", "A", "L"]:
    df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")


# ===================================
# 2) (CSV에 rate_lag가 없다면) rate_lag 계산
#    - CSV에 이미 rate_lag, S_lag, d_job이 있으면 이 블록은 스킵해도 됨
# ===================================
def ensure_rate_lag(df: pd.DataFrame):
    df = df.copy()
    group_cols = ["sigungu", "ksic1_code", "jobbig_code"]
    df = df.sort_values(group_cols + ["mdate"])

    # d_job
    if "d_job" not in df.columns:
        df["d_job"] = df["A"] - df["L"]

    # S_lag
    if "S_lag" not in df.columns:
        g = df.groupby(group_cols, sort=False)
        df["S_prev"] = g["S"].shift(1)

        # 월 차이
        df["year"] = df["mdate"].dt.year
        df["month"] = df["mdate"].dt.month
        df["year_prev"] = g["year"].shift(1)
        df["month_prev"] = g["month"].shift(1)

        diff_months = (df["year"] - df["year_prev"]) * 12 + (df["month"] - df["month_prev"])
        df["S_lag"] = np.where(diff_months == 1, df["S_prev"], np.nan)

        df = df.drop(columns=["S_prev", "year_prev", "month_prev"])

    # rate_lag
    if "rate_lag" not in df.columns:
        df["rate_lag"] = np.where(df["S_lag"] > 0, 100.0 * df["d_job"] / df["S_lag"], np.nan)

    return df

df_raw = ensure_rate_lag(df_raw)


# ===================================
# 3) (5) JSI 전처리: 결측월 채우기 + 이상치 제거 + MA36 + MA36_norm
# ===================================
def JSI_F_cleaning(df: pd.DataFrame, n_steps_ahead: int):
    df = df.copy()
    df["mdate"] = pd.to_datetime(df["mdate"])

    group_cols = ["sigungu", "ksic1_code", "jobbig_code"]
    df = df.sort_values(group_cols + ["mdate"]).reset_index(drop=True)

    # ---- 결측 월 보정(reindex) + 0 채우기
    df["original_obs"] = True
    full_list = []

    for keys, tp in df.groupby(group_cols):
        tp = tp.sort_values("mdate").dropna(subset=["mdate"])
        if tp.empty:
            continue

        start = tp["mdate"].min()
        end   = df["mdate"].max()
        full_dates = pd.date_range(start, end, freq="MS")

        tp_r = tp.set_index("mdate").reindex(full_dates)
        tp_r["mdate"] = tp_r.index

        tp_r["sigungu"], tp_r["ksic1_code"], tp_r["jobbig_code"] = keys
        tp_r["original_obs"] = tp_r["original_obs"].astype("boolean").fillna(False)

        # 라벨 컬럼이 있으면 ffill
        if "ksic1" in tp_r.columns:
            tp_r["ksic1"] = tp_r["ksic1"].ffill()
        if "job_big" in tp_r.columns:
            tp_r["job_big"] = tp_r["job_big"].ffill()

        # 날짜 파생
        tp_r["year"]  = tp_r["mdate"].dt.year.astype(int)
        tp_r["month"] = tp_r["mdate"].dt.month.astype(int)
        tp_r["ym"]    = (tp_r["year"] * 100 + tp_r["month"]).astype(int)

        # 수치 결측 0
        for c in ["rate_lag", "d_job", "S_lag", "S", "A", "L"]:
            if c not in tp_r.columns:
                tp_r[c] = 0
            tp_r[c] = tp_r[c].fillna(0)

        full_list.append(tp_r.reset_index(drop=True))

    df = pd.concat(full_list, ignore_index=True)

    # ---- Train/Test split
    unique_dates = np.sort(df["mdate"].unique())
    if len(unique_dates) <= n_steps_ahead:
        raise ValueError("mdate 유니크 개수가 n_steps_ahead보다 작거나 같습니다.")
    cutoff = unique_dates[-n_steps_ahead]  # test 시작

    df_train = df[df["mdate"] < cutoff].copy()
    df_test  = df[df["mdate"] >= cutoff].copy()

    # ---- 이상치 제거(train only): robust z (MAD)
    y_var = "rate_lag"
    z_thresh = 8.0

    df_train["robust_z"] = np.nan
    df_train["is_outlier"] = False

    for keys, tp in df_train.groupby(group_cols):
        y = tp[y_var].astype(float).to_numpy()
        y_nonan = y[~np.isnan(y)]
        if len(y_nonan) == 0:
            continue

        med = np.median(y_nonan)
        mad = np.median(np.abs(y_nonan - med))
        if mad == 0 or np.isnan(mad):
            continue

        sigma_hat = 1.4826 * mad
        z = (y - med) / sigma_hat

        df_train.loc[tp.index, "robust_z"] = z
        mask_out = np.abs(z) > z_thresh
        if mask_out.any():
            df_train.loc[tp.index[mask_out], "is_outlier"] = True

    df_train_clean = df_train[~df_train["is_outlier"]].copy()

    # ---- train/test 컬럼 맞추고 합치기
    common_cols = df_train_clean.columns.intersection(df_test.columns)
    df_train_clean = df_train_clean.reindex(columns=common_cols)
    df_test = df_test.reindex(columns=common_cols)

    df = pd.concat([df_train_clean, df_test], ignore_index=True)
    df = df.sort_values(group_cols + ["mdate"]).reset_index(drop=True)

    # ---- 36개월 이동평균
    df["MA36"] = (
        df.groupby(group_cols)[y_var]
          .transform(lambda s: s.rolling(36, min_periods=36).mean())
    )

    # ---- 표준화: SD가 0이면 0
    EPS = 1e-8
    ma_sd = df.groupby(group_cols)["MA36"].transform(lambda s: s.std(ddof=0))
    df["MA36_SD"] = ma_sd
    df["MA36_norm"] = np.where(ma_sd.isna() | (ma_sd < EPS), 0.0, df["MA36"] / ma_sd)
    df["MA36_norm"] = df["MA36_norm"].replace([np.inf, -np.inf], 0.0)

    return df


# ===================================
# 4) ETS 예측: 실제+예측 결합 + is_forecast 생성
# ===================================
def ETS_forecasting(df: pd.DataFrame, y_var: str, n_steps_ahead: int):
    import warnings
    try:
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel
        _USE_ETSMODEL = True
    except Exception:
        _USE_ETSMODEL = False
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

    group_cols = ["sigungu", "ksic1_code", "jobbig_code"]

    unique_dates = np.sort(df["mdate"].unique())
    last_date = pd.Timestamp(unique_dates[-1])
    fc_dates = pd.date_range(last_date + MonthBegin(1), periods=n_steps_ahead, freq="MS")

    df_train = df.copy()

    ets_candidates = [
        {"name": "ETS_A_N_N", "trend": None,  "damped": False},
        {"name": "ETS_A_A_N", "trend": "add", "damped": False},
        {"name": "ETS_A_A_D", "trend": "add", "damped": True},
    ]

    forecast_list = []

    for keys, g in df_train.groupby(group_cols):
        g = g.sort_values("mdate").set_index("mdate").asfreq("MS")
        y = g[y_var].dropna().astype(float)
        if len(y) < 10:
            continue

        best_aic = np.inf
        best_fc = None
        best_name = None

        for spec in ets_candidates:
            trend = spec["trend"]
            damped = spec["damped"]
            if trend is None and damped:
                continue

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if _USE_ETSMODEL:
                        res = ETSModel(
                            y, error="add", trend=trend, damped_trend=damped,
                            seasonal=None, initialization_method="estimated"
                        ).fit(disp=False)
                        aic = res.aic
                        fc = res.forecast(steps=n_steps_ahead)
                    else:
                        res = ExponentialSmoothing(
                            y, trend=trend, damped_trend=damped,
                            seasonal=None, initialization_method="estimated"
                        ).fit()
                        aic = res.aic
                        fc = res.forecast(n_steps_ahead)

                if np.isfinite(aic) and aic < best_aic:
                    best_aic = aic
                    best_fc = np.asarray(fc, dtype=float)
                    best_name = spec["name"]
            except Exception:
                continue

        if best_fc is None:
            continue

        sigungu, ksic1_code, jobbig_code = keys
        tmp = pd.DataFrame({
            "sigungu": sigungu,
            "ksic1_code": ksic1_code,
            "jobbig_code": jobbig_code,
            "mdate": fc_dates,
            y_var: best_fc,
            "selected_ets_spec": best_name,
            "aic": best_aic,
            "is_forecast": True,
        })
        forecast_list.append(tmp)

    fc_df = pd.concat(forecast_list, ignore_index=True) if forecast_list else pd.DataFrame()

    # ---- actual part
    keep_cols = ["sigungu", "ksic1_code", "jobbig_code", "mdate", y_var]
    actual = df[keep_cols].copy()
    actual["is_forecast"] = False

    # ---- combine
    if not fc_df.empty:
        df_all = pd.concat([actual, fc_df[keep_cols + ["is_forecast", "selected_ets_spec", "aic"]]], ignore_index=True)
    else:
        df_all = actual

    df_all = df_all.sort_values(["sigungu", "ksic1_code", "jobbig_code", "mdate"]).reset_index(drop=True)
    return df_all


# ===================================
# 5) 주의/위기선(rolling 36) + ✅ 장기주의/장기위기선 계산
# ===================================
def add_warning_crisis_lines(df_all: pd.DataFrame, y_var: str, long_window: int = 36):
    keys = ["sigungu", "ksic1_code", "jobbig_code"]
    df = df_all.copy()
    df["mdate"] = pd.to_datetime(df["mdate"])

    # 실제+예측 이어붙인 시계열 만들기:
    test_dates = pd.to_datetime(df.loc[df["is_forecast"] == True, "mdate"].unique())
    in_test = df["mdate"].isin(test_dates)

    use_row = (in_test & (df["is_forecast"] == True)) | (~in_test & (df["is_forecast"] == False))
    df_used = df.loc[use_row].sort_values(keys + ["mdate"]).copy()

    g = df_used.groupby(keys, sort=False)

    # (단기) rolling 36
    df_used["roll_mean_36"] = g[y_var].transform(lambda s: s.rolling(36, min_periods=36).mean())
    df_used["roll_std_36"]  = g[y_var].transform(lambda s: s.rolling(36, min_periods=36).std(ddof=1))

    df_used["Sign_warning"] = df_used["roll_mean_36"] - 1.5 * df_used["roll_std_36"]
    df_used["Sign_Crisis"]  = df_used["roll_mean_36"] - 2.0 * df_used["roll_std_36"]

    # ✅ (장기) 최근 36개월 평균을 "고정 기준선"으로
    df_used["Long_warning"] = g["Sign_warning"].transform(lambda s: s.tail(long_window).mean())
    df_used["Crisis_SS"]    = g["Sign_Crisis"].transform(lambda s: s.tail(long_window).mean())

    roll_cols = keys + ["mdate",
                        "roll_mean_36", "roll_std_36",
                        "Sign_warning", "Sign_Crisis",
                        "Long_warning", "Crisis_SS"]
    df = df.merge(df_used[roll_cols], on=keys + ["mdate"], how="left")
    return df


# ===================================
# 6) Plot (실제+예측 + 단기선 + ✅ 장기선)
# ===================================
def plot_one(df_all, *, sigungu, ksic1_code, jobbig_code,
             y_var="MA36_norm", plot_start="2021-01-01"):

    df = df_all.copy()
    df["mdate"] = pd.to_datetime(df["mdate"])

    g = df[(df["sigungu"] == sigungu) &
           (df["ksic1_code"] == ksic1_code) &
           (df["jobbig_code"] == jobbig_code)].copy()

    if g.empty:
        raise ValueError("해당 조건 데이터가 없습니다.")

    g = g.sort_values("mdate")
    g = g[g["mdate"] >= pd.Timestamp(plot_start)].copy()

    g_real = g[g["is_forecast"] == False]
    g_fc   = g[g["is_forecast"] == True]

    fig, ax = plt.subplots(figsize=(12, 7))

    # forecast shading
    if not g_fc.empty:
        ax.axvspan(g_fc["mdate"].min(), g_fc["mdate"].max() + MonthEnd(1), alpha=0.08)

    ax.plot(g_real["mdate"], g_real[y_var], marker="o", label="실제")
    if not g_fc.empty:
        ax.plot(g_fc["mdate"], g_fc[y_var], marker="p", linestyle="--", label="예측")

    # 단기(시간가변) 주의/위기선
    if "Sign_warning" in g.columns and g["Sign_warning"].notna().any():
        ax.plot(g["mdate"], g["Sign_warning"], label="주의선(rolling mean-1.5σ)")
    if "Sign_Crisis" in g.columns and g["Sign_Crisis"].notna().any():
        ax.plot(g["mdate"], g["Sign_Crisis"], label="위기선(rolling mean-2.0σ)")

    # ✅ 장기(수평) 주의/위기선
    '''
    if "Long_warning" in g.columns and g["Long_warning"].notna().any():
        lw = g["Long_warning"].dropna().iloc[0]
        ax.hlines(lw,
                  xmin=g["mdate"].min(),
                  xmax=g["mdate"].max() + MonthEnd(1),
                  linestyles=":", linewidth=1.8,
                  label="장기주의선(최근36개월 평균)")
    '''

    if "Crisis_SS" in g.columns and g["Crisis_SS"].notna().any():
        ss = g["Crisis_SS"].dropna().iloc[0]
        ax.hlines(ss,
                  xmin=g["mdate"].min(),
                  xmax=g["mdate"].max() + MonthEnd(1),
                  linestyles=":", linewidth=1.8,
                  label="장기위기선(SS, 최근36개월 평균)")

    ax.set_title(f"{sigungu} | ksic1={ksic1_code} | job={jobbig_code} | {y_var}")
    ax.set_xlabel("시간")
    ax.set_ylabel(y_var)
    ax.grid(alpha=0.2)
    ax.legend()
    plt.tight_layout()
    return fig


# ===================================
# 7) 실행
# ===================================
n_steps_ahead = 6          # 3 또는 6
y_var = "MA36_norm"

@st.cache_data(show_spinner=False)
def build_all():
    df_clean = JSI_F_cleaning(df_raw, n_steps_ahead=n_steps_ahead)
    df_all   = ETS_forecasting(df_clean, y_var=y_var, n_steps_ahead=n_steps_ahead)
    df_all   = add_warning_crisis_lines(df_all, y_var=y_var, long_window=36)
    return df_all

df_all = build_all()

st.title("경기도 시군별 JSI 그래프")
st.caption("전산업(20), 전직종(20) 고정")

KSIC_FIXED = 20
JOBBIG_FIXED = 20

sigungu_list = sorted(
    df_all.loc[
        (df_all["ksic1_code"] == KSIC_FIXED) &
        (df_all["jobbig_code"] == JOBBIG_FIXED),
        "sigungu"
    ].dropna().unique()
)

if not sigungu_list:
    st.warning("해당 조건(ksic1_code=20, jobbig_code=20)에 맞는 시군 데이터가 없습니다.")
else:
    for s in sigungu_list:
        st.subheader(str(s))
        fig = plot_one(
            df_all,
            sigungu=s,
            ksic1_code=KSIC_FIXED,
            jobbig_code=JOBBIG_FIXED,
            y_var=y_var,
            plot_start="2021-01-01"
        )
        st.pyplot(fig, clear_figure=True)











