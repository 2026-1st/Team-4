# app.py
# Streamlit 기반 US Accidents Severity 예측 웹앱
# 사용자가 직접 사고 정보를 입력하면 학습된 모델이 Severity를 예측한다.

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import joblib

ARTIFACT_PATH = Path("streamlit_artifacts/severity_app_artifacts.joblib")


def weather_group(x):
    """노트북 전처리와 동일하게 Weather_Condition을 단순 Weather_Group으로 변환한다."""
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
    """Yes/No, True/False, 1/0 형태 입력을 0 또는 1로 변환한다."""
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


def yes_no_to_bool_text(value):
    return "Yes" if value == "Yes" else "No"


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
    """
    사용자가 입력한 사고 정보를 학습 데이터와 동일한 형태로 전처리한다.
    노트북에서 저장한 artifacts를 사용하므로 학습 당시 컬럼, 중앙값, 최빈값, 스케일러 기준을 그대로 따른다.
    """
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
    num_cols = list(artifacts["num_cols"])
    cat_cols = list(artifacts["cat_cols"])

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
    scale_cols = list(artifacts.get("scale_cols", []))
    if len(scale_cols) > 0:
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
st.set_page_config(page_title="US Accidents Severity 예측", page_icon="🚗", layout="wide")

st.title("🚗 교통사고 Severity 예측 웹앱")
st.write("사고 시간, 위치, 기상 조건, 도로 환경 정보를 입력하면 학습된 모델이 사고 심각도 등급을 예측합니다.")

artifacts = load_artifacts()
st.info(f"현재 사용 모델: {artifacts['best_model_name']}")

yes_no_options = ["Yes", "No"]
weather_options = ["Clear", "Cloudy", "Rain", "Snow", "Fog", "Storm", "Other"]
timezone_options = [
    "US/Pacific",
    "US/Mountain",
    "US/Central",
    "US/Eastern",
    "US/Arizona",
    "US/Alaska",
    "US/Hawaii",
]
wind_direction_options = [
    "CALM", "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW", "VAR"
]
day_night_options = ["Day", "Night"]

with st.form("severity_prediction_form"):
    st.subheader("1) 사고 기본 정보")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        accident_date = st.date_input("사고 날짜", value=pd.to_datetime("2023-07-15").date())
        hour = st.number_input("사고 시간", min_value=0, max_value=23, value=18, step=1)
    with col2:
        state = st.text_input("주(State)", value="CA")
        city = st.text_input("도시(City)", value="Los Angeles")
    with col3:
        county = st.text_input("카운티(County)", value="Los Angeles")
        timezone = st.selectbox("시간대(Timezone)", timezone_options, index=0)
    with col4:
        start_lat = st.number_input("위도(Start_Lat)", value=34.0522, format="%.6f")
        start_lng = st.number_input("경도(Start_Lng)", value=-118.2437, format="%.6f")

    st.subheader("2) 기상 정보")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        weather = st.selectbox("날씨", weather_options, index=2)
        temperature = st.number_input("기온(F)", value=65.0, step=1.0)
    with col2:
        wind_chill = st.number_input("체감온도(F)", value=65.0, step=1.0)
        humidity = st.number_input("습도(%)", min_value=0.0, max_value=100.0, value=80.0, step=1.0)
    with col3:
        pressure = st.number_input("기압(in)", min_value=0.0, value=29.9, step=0.1)
        visibility = st.number_input("가시거리(mi)", min_value=0.0, value=3.0, step=0.5)
    with col4:
        wind_speed = st.number_input("풍속(mph)", min_value=0.0, value=5.0, step=0.5)
        wind_direction = st.selectbox("풍향", wind_direction_options, index=0)

    col1, col2 = st.columns(2)
    with col1:
        precipitation_unknown = st.checkbox("강수량 정보 없음", value=False)
    with col2:
        precipitation = st.number_input(
            "강수량(in)",
            min_value=0.0,
            value=0.0,
            step=0.01,
            disabled=precipitation_unknown,
        )

    st.subheader("3) 주야간 및 도로 환경 정보")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        sunrise_sunset = st.selectbox("주야간(Sunrise/Sunset)", day_night_options, index=0)
        traffic_signal = st.selectbox("신호등 여부", yes_no_options, index=0)
        junction = st.selectbox("교차로 여부", yes_no_options, index=0)
        crossing = st.selectbox("횡단보도 여부", yes_no_options, index=0)

    with col2:
        stop = st.selectbox("정지표지판 여부", yes_no_options, index=1)
        station = st.selectbox("역/정류장 여부", yes_no_options, index=1)
        railway = st.selectbox("철도 인접 여부", yes_no_options, index=1)
        roundabout = st.selectbox("회전교차로 여부", yes_no_options, index=1)

    with col3:
        amenity = st.selectbox("편의시설 인접 여부", yes_no_options, index=1)
        bump = st.selectbox("과속방지턱 여부", yes_no_options, index=1)
        traffic_calming = st.selectbox("교통정온화시설 여부", yes_no_options, index=1)
        give_way = st.selectbox("양보표지 여부", yes_no_options, index=1)

    with col4:
        no_exit = st.selectbox("막다른 길 여부", yes_no_options, index=1)
        turning_loop = st.selectbox("Turning Loop 여부", yes_no_options, index=1)
        civil_twilight = st.selectbox("Civil Twilight", day_night_options, index=0)
        nautical_twilight = st.selectbox("Nautical Twilight", day_night_options, index=0)
        astronomical_twilight = st.selectbox("Astronomical Twilight", day_night_options, index=0)

    submitted = st.form_submit_button("예측하기", type="primary")


start_time = f"{accident_date.strftime('%Y-%m-%d')} {int(hour):02d}:00:00"
precipitation_value = np.nan if precipitation_unknown else precipitation

user_input = {
    "Start_Time": start_time,
    "Start_Lat": start_lat,
    "Start_Lng": start_lng,
    "State": state.strip().upper(),
    "County": county.strip(),
    "City": city.strip(),
    "Timezone": timezone,

    "Temperature(F)": temperature,
    "Wind_Chill(F)": wind_chill,
    "Humidity(%)": humidity,
    "Pressure(in)": pressure,
    "Visibility(mi)": visibility,
    "Wind_Direction": wind_direction,
    "Wind_Speed(mph)": wind_speed,
    "Precipitation(in)": precipitation_value,
    "Weather_Condition": weather,

    "Sunrise_Sunset": sunrise_sunset,
    "Civil_Twilight": civil_twilight,
    "Nautical_Twilight": nautical_twilight,
    "Astronomical_Twilight": astronomical_twilight,

    "Amenity": amenity,
    "Bump": bump,
    "Crossing": crossing,
    "Give_Way": give_way,
    "Junction": junction,
    "No_Exit": no_exit,
    "Railway": railway,
    "Roundabout": roundabout,
    "Station": station,
    "Stop": stop,
    "Traffic_Calming": traffic_calming,
    "Traffic_Signal": traffic_signal,
    "Turning_Loop": turning_loop,
}

if submitted:
    pred, probability = predict_severity(user_input, artifacts)
    description = artifacts["severity_description"].get(pred, "정의되지 않은 Severity")

    st.success(f"결과: 예측 Severity = {pred}")
    st.write(description)

    if probability is not None:
        prob_df = pd.DataFrame({
            "Severity": list(probability.keys()),
            "Probability": list(probability.values()),
        }).sort_values("Severity")

        st.subheader("Severity별 예측 확률")
        st.dataframe(prob_df, use_container_width=True, hide_index=True)

    with st.expander("입력값 확인"):
        st.json(user_input)
