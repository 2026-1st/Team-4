# app.py
# Streamlit 기반 US Accidents Severity 예측 웹앱

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import joblib

ARTIFACT_PATH = Path("streamlit_artifacts/severity_app_artifacts.joblib")


def weather_group(x):
    if pd.isna(x):
        return "Unknown"

    x = str(x).lower()

    if "clear" in x or "fair" in x:
        return "Clear"
    elif "cloud" in x or "overcast" in x:
        return "Cloudy"
    elif "rain" in x or "drizzle" in x or "shower" in x:
        return "Rain"
    elif "snow" in x or "sleet" in x or "ice" in x:
        return "Snow"
    elif "fog" in x or "mist" in x or "haze" in x:
        return "Fog"
    elif "storm" in x or "thunder" in x:
        return "Storm"
    else:
        return "Other"


def to_binary(value):
    if pd.isna(value):
        return np.nan

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, float, np.integer, np.floating)):
        return int(value)

    value = str(value).strip().lower()
    true_values = {"true", "t", "yes", "y", "1", "있음", "예"}
    false_values = {"false", "f", "no", "n", "0", "없음", "아니오"}

    if value in true_values:
        return 1
    if value in false_values:
        return 0

    return np.nan


@st.cache_resource
def load_artifacts():
    if not ARTIFACT_PATH.exists():
        st.error(
            "모델 파일을 찾을 수 없습니다. 먼저 노트북의 'Streamlit 웹앱용 모델 및 전처리 정보 저장' 셀을 실행하세요."
        )
        st.stop()

    artifacts = joblib.load(ARTIFACT_PATH)

    if artifacts["model_type"] == "keras":
        try:
            from tensorflow.keras.models import load_model
        except ImportError:
            st.error("Keras 모델을 불러오려면 tensorflow가 설치되어 있어야 합니다.")
            st.stop()
        artifacts["model"] = load_model(artifacts["keras_model_path"])

    return artifacts


def preprocess_user_input(user_input, artifacts):
    row = {}
    for col in artifacts["raw_input_cols"]:
        row[col] = user_input.get(col, np.nan)

    user_df = pd.DataFrame([row])

    # Start_Time에서 파생 변수 생성
    user_df["Start_Time"] = pd.to_datetime(user_df["Start_Time"], errors="coerce")
    user_df["Year"] = user_df["Start_Time"].dt.year
    user_df["Month"] = user_df["Start_Time"].dt.month
    user_df["DayOfWeek"] = user_df["Start_Time"].dt.dayofweek
    user_df["Hour"] = user_df["Start_Time"].dt.hour
    user_df = user_df.drop(columns=["Start_Time"])

    # 날씨 그룹화
    user_df["Weather_Group"] = user_df["Weather_Condition"].apply(weather_group)
    user_df = user_df.drop(columns=["Weather_Condition"])

    # 강수량 결측 여부
    user_df["Precipitation_NA"] = user_df["Precipitation(in)"].isna().astype(int)

    # 도로 환경 변수 변환
    for col in artifacts["road_cols"]:
        if col in user_df.columns:
            user_df[col] = user_df[col].apply(to_binary)

    # 학습 당시 입력 컬럼에 맞추기
    for col in artifacts["input_columns"]:
        if col not in user_df.columns:
            user_df[col] = np.nan
    user_df = user_df[artifacts["input_columns"]]

    # 고유값이 많은 범주는 학습 당시 상위 범주 외 Other 처리
    for col in artifacts["high_cardinality_cols"]:
        if col in user_df.columns:
            user_df[col] = user_df[col].fillna("Unknown")
            top_values = artifacts["top_category_dict"].get(col, [])
            user_df[col] = user_df[col].where(user_df[col].isin(top_values), "Other")

    # 결측치 처리
    num_cols = artifacts["num_cols"]
    cat_cols = artifacts["cat_cols"]

    if len(num_cols) > 0:
        user_df[num_cols] = user_df[num_cols].apply(pd.to_numeric, errors="coerce")
        user_df[num_cols] = user_df[num_cols].fillna(artifacts["num_median"])

    if len(cat_cols) > 0:
        user_df[cat_cols] = user_df[cat_cols].fillna(artifacts["cat_mode"])

    # One-Hot Encoding 후 학습 컬럼과 동일하게 정렬
    user_encoded = pd.get_dummies(user_df, columns=cat_cols)
    user_encoded = user_encoded.reindex(columns=artifacts["train_encoded_columns"], fill_value=0)

    bool_dummy_cols = user_encoded.select_dtypes(include=["bool"]).columns
    user_encoded[bool_dummy_cols] = user_encoded[bool_dummy_cols].astype("int8")

    # 표준화
    user_scaled = user_encoded.copy()
    scale_cols = artifacts["scale_cols"]
    if scale_cols:
        user_scaled[scale_cols] = artifacts["scaler"].transform(user_scaled[scale_cols])

    return user_scaled


def predict_severity(user_input, artifacts):
    model = artifacts["model"]
    user_scaled = preprocess_user_input(user_input, artifacts)

    if artifacts["model_type"] == "keras":
        proba = model.predict(user_scaled.astype("float32").values, verbose=0)[0]
        pred = int(np.argmax(proba) + 1)
        probability = {i + 1: round(float(p), 4) for i, p in enumerate(proba)}
        return pred, probability

    pred = model.predict(user_scaled)[0]
    offset = artifacts["model_label_offset"].get(artifacts["best_model_name"], 0)
    pred = int(pred + offset)

    probability = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(user_scaled)[0]
        class_labels = getattr(model, "classes_", np.array([1, 2, 3, 4]))
        if offset != 0:
            class_labels = class_labels + offset
        probability = {int(label): round(float(prob), 4) for label, prob in zip(class_labels, proba)}

    return pred, probability


# =========================
# Streamlit 화면
# =========================
st.set_page_config(page_title="US Accidents Severity 예측", page_icon="🚗", layout="centered")

st.title("🚗 교통사고 Severity 예측 웹앱")
st.write("사용자가 사고 정보를 입력하면 학습된 모델이 사고 심각도 등급을 예측합니다.")

artifacts = load_artifacts()

st.info(f"현재 사용 모델: {artifacts['best_model_name']}")

col1, col2 = st.columns(2)

with col1:
    hour = st.number_input("사고 시간", min_value=0, max_value=23, value=18, step=1)
    month = st.number_input("월", min_value=1, max_value=12, value=7, step=1)
    state = st.text_input("주", value="CA")
    city = st.text_input("도시", value="Los Angeles")
    weather = st.selectbox("날씨", ["Clear", "Cloudy", "Rain", "Snow", "Fog", "Storm", "Other"], index=2)

with col2:
    temperature = st.number_input("기온(F)", value=65.0, step=1.0)
    humidity = st.number_input("습도(%)", min_value=0.0, max_value=100.0, value=80.0, step=1.0)
    visibility = st.number_input("가시거리(mi)", min_value=0.0, value=3.0, step=0.5)
    traffic_signal = st.selectbox("신호등 여부", ["Yes", "No"], index=0)
    crossing = st.selectbox("교차로 여부", ["Yes", "No"], index=0)

# 사용자가 입력하지 않은 변수는 학습 데이터의 중앙값/최빈값으로 대체되도록 최소 입력만 구성
start_time = f"2023-{int(month):02d}-15 {int(hour):02d}:00:00"

user_input = {
    "Start_Time": start_time,
    "State": state,
    "City": city,
    "County": city,
    "Weather_Condition": weather,
    "Temperature(F)": temperature,
    "Humidity(%)": humidity,
    "Visibility(mi)": visibility,
    "Traffic_Signal": traffic_signal,
    "Crossing": crossing,

    # 사용자가 입력하지 않는 주요 숫자형 변수는 NaN으로 두면 학습 데이터 중앙값으로 채워짐
    "Start_Lat": np.nan,
    "Start_Lng": np.nan,
    "Wind_Chill(F)": np.nan,
    "Pressure(in)": np.nan,
    "Wind_Speed(mph)": np.nan,
    "Precipitation(in)": np.nan,
    "Wind_Direction": np.nan,
    "Timezone": np.nan,
    "Sunrise_Sunset": np.nan,
    "Civil_Twilight": np.nan,
    "Nautical_Twilight": np.nan,
    "Astronomical_Twilight": np.nan,

    # 도로 환경 변수 기본값
    "Amenity": "No",
    "Bump": "No",
    "Give_Way": "No",
    "Junction": "No",
    "No_Exit": "No",
    "Railway": "No",
    "Roundabout": "No",
    "Station": "No",
    "Stop": "No",
    "Traffic_Calming": "No",
    "Turning_Loop": "No",
}

if st.button("예측하기", type="primary"):
    pred, probability = predict_severity(user_input, artifacts)
    description = artifacts["severity_description"].get(pred, "정의되지 않은 Severity")

    st.success(f"결과: 예측 Severity = {pred}")
    st.write(description)

    if probability is not None:
        prob_df = pd.DataFrame({
            "Severity": list(probability.keys()),
            "Probability": list(probability.values()),
        })
        st.subheader("Severity별 예측 확률")
        st.dataframe(prob_df, use_container_width=True, hide_index=True)
