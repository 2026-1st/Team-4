# US Accidents 교통사고 Severity 예측 프로젝트

## 1. 프로젝트 개요

본 프로젝트는 Kaggle의 **US Accidents (2016-2023)** 데이터를 활용하여 교통사고 심각도(`Severity`)를 예측하는 지도학습 기반 다중분류 프로젝트입니다.

사고 발생 시간, 위치, 기상 조건, 도로 주변 시설 정보를 바탕으로 사고 심각도를 `1`, `2`, `3`, `4`의 네 개 클래스로 분류합니다. 또한 학습된 최종 모델을 활용하여 사용자가 사고 정보를 입력하면 웹 화면에서 예측 결과를 확인할 수 있도록 **Streamlit 기반 Severity 예측 웹앱**을 구현했습니다.

웹앱에는 지도 기반 위치 선택 기능을 추가했습니다. 사용자가 미국 지도에서 사고 위치를 클릭하면 위도·경도가 자동 입력되고, 클릭 좌표와 가장 가까운 기존 사고 데이터를 기준으로 `State`, `City`, `County`, `Timezone`이 자동 설정됩니다.

---

## 2. 데이터셋 정보

| 항목 | 내용 |
| --- | --- |
| Dataset | US Accidents (2016-2023) |
| Source | Kaggle |
| URL | https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents |
| Download Date | 2026-05-24 |
| License | CC BY-NC-SA 4.0 |
| Target Variable | `Severity` |
| Task Type | Multiclass Classification |

> 원본 데이터 파일 `US_Accidents_March23.csv`는 용량이 크기 때문에 GitHub 저장소에 포함하지 않습니다.  
> 실행을 위해서는 Kaggle에서 데이터를 다운로드한 뒤 `data/US_Accidents_March23.csv` 경로에 배치해야 합니다.

---

## 3. 실행 환경

| 항목 | 권장/확인 환경 |
| --- | --- |
| OS | Windows 11 |
| Python | Python 3.10 이상 권장 |
| IDE | VS Code / Jupyter Notebook |
| Notebook | `Team4_US_Accidents_Final_Project_revised.ipynb` |
| Web App | Streamlit |
| Main ML Library | scikit-learn |
| Additional Libraries | pandas, numpy, matplotlib, joblib, xgboost, lightgbm, tensorflow |
| Map Libraries | folium, streamlit-folium |

Streamlit 지도 기능을 사용하려면 `folium`, `streamlit-folium`이 설치되어 있어야 합니다.

---

## 4. 리포지토리 구조

```text
Team-4/
├─ README.md
├─ .gitignore
├─ requirements.txt
├─ requirements_streamlit.txt
├─ app.py
├─ Team4_US_Accidents_Final_Project_revised.ipynb
├─ streamlit_artifacts/
│  ├─ severity_app_artifacts.joblib
│  └─ location_reference.csv
├─ outputs/
│  ├─ model_validation_results_all_candidates.csv
│  ├─ model_test_result.csv
│  ├─ final_model_classification_report.csv
│  ├─ final_model_feature_importance.csv
│  ├─ rf_depth_results.csv
│  ├─ rf_leaf_results.csv
│  ├─ nn_validation_result.csv
│  ├─ imbalance_experiment_results.csv
│  └─ experiment_summary.csv
├─ figures/
│  ├─ fig01_severity_distribution.png
│  ├─ fig02_original_sample_ratio.png
│  ├─ fig03_missing_ratio.png
│  ├─ fig04_accidents_by_hour_line.png
│  ├─ fig05_accidents_by_month.png
│  ├─ fig06_accidents_by_weather.png
│  ├─ fig07_weather_severity_ratio.png
│  ├─ fig08_signal_severity_ratio.png
│  ├─ fig09_junction_severity_ratio.png
│  ├─ fig10_visibility_by_severity.png
│  ├─ fig11_top10_states.png
│  ├─ fig12_road_feature_ratio.png
│  ├─ fig13_us_accidents_severity_map.png
│  ├─ fig14_rf_depth_comparison.png
│  ├─ fig15_rf_leaf_comparison.png
│  ├─ fig15_nn_training_curve.png
│  ├─ fig17_top10_macro_f1_comparison.png
│  ├─ fig18_final_confusion_matrix.png
│  ├─ fig19_final_feature_importance.png
│  └─ imbalance_experiment_comparison.png
└─ data/
   └─ US_Accidents_March23.csv
```

### 주요 파일 설명

| 파일/폴더 | 설명 |
| --- | --- |
| `Team4_US_Accidents_Final_Project_revised.ipynb` | 데이터 로드, EDA, 전처리, 모델 학습, 평가, 결과 저장을 수행하는 메인 노트북 |
| `app.py` | Streamlit 기반 Severity 예측 웹앱 |
| `requirements.txt` | 노트북 실행에 필요한 주요 패키지 |
| `requirements_streamlit.txt` | Streamlit 웹앱 실행에 필요한 패키지 |
| `streamlit_artifacts/severity_app_artifacts.joblib` | 최종 모델과 전처리 정보를 저장한 웹앱용 artifact |
| `streamlit_artifacts/location_reference.csv` | 지도 클릭 시 위치 정보를 자동 설정하기 위한 위치 참조 데이터 |
| `outputs/` | 모델 성능 결과 CSV 저장 폴더 |
| `figures/` | EDA 및 모델 평가 시각화 이미지 저장 폴더 |
| `data/` | 원본 데이터 저장 폴더. GitHub에는 업로드하지 않음 |

---

## 5. 사용 변수

### 5.1 시간 변수

원본 `Start_Time`에서 다음 파생 변수를 생성했습니다.

- `Month`
- `DayOfWeek`
- `Hour`

`Year`는 최종 입력 변수에서 제외했습니다. 사고 발생 연도는 연도별 데이터 수집 분포나 기록 방식 차이를 반영할 가능성이 있으므로, 최종 모델에서는 사고 상황을 더 직접적으로 나타내는 월, 요일, 시간대만 사용했습니다.

### 5.2 위치 변수

- `Start_Lat`
- `Start_Lng`
- `City`
- `County`
- `State`
- `Timezone`

### 5.3 기상 변수

- `Temperature(F)`
- `Wind_Chill(F)`
- `Humidity(%)`
- `Pressure(in)`
- `Visibility(mi)`
- `Wind_Direction`
- `Wind_Speed(mph)`
- `Precipitation(in)`
- `Precipitation_NA`
- `Weather_Group`

### 5.4 도로 및 주변 시설 변수

- `Amenity`
- `Bump`
- `Crossing`
- `Give_Way`
- `Junction`
- `No_Exit`
- `Railway`
- `Roundabout`
- `Station`
- `Stop`
- `Traffic_Calming`
- `Traffic_Signal`
- `Turning_Loop`

### 5.5 주야간 관련 변수

- `Sunrise_Sunset`
- `Civil_Twilight`
- `Nautical_Twilight`
- `Astronomical_Twilight`

---

## 6. 전처리 과정

노트북에서는 다음과 같은 전처리를 수행했습니다.

1. 필요한 변수만 선택하여 데이터 로드
2. `Severity` 클래스 분포 확인
3. 전체 데이터에서 100,000개 샘플을 `Severity` 기준 층화 샘플링
4. `Start_Time`을 날짜형으로 변환 후 `Month`, `DayOfWeek`, `Hour` 생성
5. `Weather_Condition`을 단순화하여 `Weather_Group` 생성
6. `City`, `County` 결측치는 `Unknown`으로 대체
7. 학습 데이터 기준 `City`, `County` 상위 30개 값만 유지하고 나머지는 `Other`로 처리
8. `Precipitation_NA` 변수를 추가하여 강수량 결측 여부 반영
9. Boolean 도로 환경 변수를 0/1 정수형으로 변환
10. 수치형 변수는 학습 데이터의 중앙값으로 결측치 대체
11. 범주형 변수는 학습 데이터의 최빈값으로 결측치 대체
12. 범주형 변수는 One-Hot Encoding 적용
13. 연속형 수치 변수는 StandardScaler로 표준화

Streamlit 웹앱에서도 노트북 학습 시 저장한 전처리 정보를 `severity_app_artifacts.joblib`에서 불러와 동일한 방식으로 입력값을 변환합니다.

---

## 7. EDA 및 시각화

주요 EDA에서는 `Severity` 클래스 분포, 시간대별 사고 발생 건수, 월별 사고 발생 건수, 기상 조건, 도로 환경 변수, 주별 사고 건수, 위도·경도 기반 사고 위치 분포를 확인했습니다.

위치 분포 시각화는 전체 샘플 데이터의 사고 위치를 위도·경도 산점도로 표현한 **단일 지도형 시각화**입니다. 클래스별 5,000개 균형 지도 시각화는 최종 코드에서 사용하지 않습니다.

---

## 8. 데이터 분할

모델 학습 및 평가를 위해 데이터를 다음과 같이 분할했습니다.

| 구분 | 비율 |
| --- | --- |
| Train Set | 약 70% |
| Validation Set | 약 15% |
| Test Set | 약 15% |

모든 분할 과정에서 `Severity` 클래스 비율이 유지되도록 `stratify` 옵션을 사용했습니다.

---

## 9. 사용 모델 및 실험

본 프로젝트에서는 다음 모델들을 비교했습니다.

| 모델 | 설명 |
| --- | --- |
| Dummy Classifier | 최빈 클래스만 예측하는 기준 모델 |
| Logistic Regression | 선형 기반 분류 모델 |
| Decision Tree | 트리 기반 단일 분류 모델 |
| Random Forest | 여러 결정트리를 결합한 앙상블 모델 |
| Random Forest Balanced | 클래스 불균형을 고려한 Random Forest |
| HistGradientBoosting | 히스토그램 기반 Gradient Boosting 모델 |
| XGBoost Weighted | 클래스 가중치를 반영한 XGBoost 모델 |
| LightGBM Balanced | 클래스 불균형을 고려한 LightGBM 모델 |
| Simple Dense Neural Network | 간단한 완전연결 신경망 모델 |

추가로 클래스 불균형 문제를 완화하기 위해 다음 실험을 수행했습니다.

- Undersampling
- Manual class weight

`RandomOverSampler`와 `SMOTE` 실험 코드는 포함되어 있지만, `imbalanced-learn` 설치 여부에 따라 선택적으로 실행됩니다. 현재 최종 결과 해석은 실제 결과표에 포함된 Undersampling과 Manual class weight 실험을 중심으로 정리했습니다.

---

## 10. 평가 지표

본 데이터는 `Severity=2` 클래스가 큰 비중을 차지하는 불균형 데이터입니다. 따라서 단순 정확도만 사용하지 않고 다음 평가 지표를 함께 사용했습니다.

- Accuracy
- Balanced Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Weighted F1-score

최종 모델은 Validation set의 `macro_f1`을 주요 기준으로 선택했습니다. `macro_f1`은 전체 정답률을 의미하는 값이 아니라, 각 Severity 클래스별 F1-score를 동일한 비중으로 평균한 값입니다. 따라서 클래스 불균형 상황에서 소수 클래스 성능까지 함께 평가하는 데 적합합니다.

발표용 최종 모델 비교 그래프는 여러 지표를 모두 표시하지 않고, **Macro F1 기준 상위 10개 모델**만 가로 막대그래프로 비교하여 한눈에 확인할 수 있도록 구성했습니다.

---

## 11. 주요 Validation 결과

최종 전처리 및 실험 결과, Validation set의 `macro_f1` 기준으로 가장 높은 성능을 보인 모델은 다음과 같습니다.

| Model | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: | ---: |
| RF Balanced min_samples_leaf=5 | 0.7481 | 0.4308 | 0.4095 | 0.7584 |

불균형 처리 실험에서는 일부 모델의 `balanced_accuracy`가 상승했지만, 최종 통합 비교의 `macro_f1` 기준으로는 `RF Balanced min_samples_leaf=5`가 가장 우수했습니다. 따라서 본 프로젝트에서는 해당 모델을 최종 모델로 선정했습니다.

---

## 12. 최종 모델 해석

최종 모델은 Validation set에서 가장 높은 `macro_f1`을 보인 `RF Balanced min_samples_leaf=5`입니다. Test set 평가 결과 전체 accuracy는 일정 수준을 보였지만, Confusion Matrix를 확인하면 `Severity 2`에 비해 `Severity 1`과 `Severity 4`의 예측 성능은 낮게 나타났습니다.

이는 데이터가 `Severity 2` 중심으로 불균형하게 구성되어 있고, 사고 심각도를 직접적으로 설명할 수 있는 교통량, 속도, 차종, 사고 유형 등의 변수가 부족하기 때문으로 해석할 수 있습니다. 따라서 본 모델은 전체적인 사고 심각도 예측 가능성을 확인하는 데 의미가 있지만, 소수 클래스 예측에는 한계가 있습니다.

---

## 13. Streamlit 웹앱 기능

본 프로젝트에는 학습된 모델을 활용하여 사용자가 직접 입력한 사고 정보로 Severity를 예측하는 Streamlit 웹앱이 포함되어 있습니다.

### 13.1 지도 기반 위치 자동 입력

웹앱에서는 사용자가 미국 지도에서 사고 위치를 클릭할 수 있습니다.

```text
지도에서 사고 위치 클릭
→ 위도(Start_Lat), 경도(Start_Lng) 자동 입력
→ 클릭 좌표와 가장 가까운 기존 사고 데이터를 검색
→ State, City, County, Timezone 자동 설정
```

위치 정보 자동 설정에는 `streamlit_artifacts/location_reference.csv`가 사용됩니다. 이 파일은 원본 사고 데이터에서 다음 컬럼만 추출하여 생성합니다.

```text
Start_Lat, Start_Lng, State, City, County, Timezone
```

정확한 행정구역 API를 호출하는 방식이 아니라, 데이터셋 안에서 클릭 위치와 가장 가까운 사고 지점의 정보를 참조하는 방식입니다. 따라서 별도의 외부 API 키 없이 실행할 수 있습니다.

### 13.2 입력 화면 구성

웹앱에서 사용자는 다음 정보를 입력하거나 지도 클릭을 통해 자동 설정합니다.

```text
사고 기본 정보
- 사고 날짜
- 사고 시간
- 지도 기반 위치 선택
- 주(State)
- 도시(City)
- 카운티(County)
- 시간대(Timezone)
- 위도(Start_Lat)
- 경도(Start_Lng)

기상 정보
- 날씨
- 기온
- 체감온도
- 습도
- 기압
- 가시거리
- 풍속
- 풍향
- 강수량
- 강수량 정보 없음 여부

주야간 및 도로 환경 정보
- 주야간
- Civil / Nautical / Astronomical Twilight
- 신호등 여부
- 교차로 여부
- 횡단보도 여부
- 정지표지판 여부
- 역/정류장 여부
- 철도 인접 여부
- 회전교차로 여부
- 편의시설 인접 여부
- 과속방지턱 여부
- 교통정온화시설 여부
- 양보표지 여부
- 막다른 길 여부
- Turning Loop 여부
```

### 13.3 예측 흐름

```text
사용자 입력 및 지도 위치 선택
→ app.py에서 입력값 수집
→ streamlit_artifacts/severity_app_artifacts.joblib 로드
→ 학습 당시와 동일한 전처리 적용
→ 최종 선택 모델로 Severity 예측
→ 웹 화면에 예측 Severity 및 클래스별 확률 출력
```

---

## 14. 실행 방법

### 14.1 저장소 클론

```bash
git clone https://github.com/2026-1st/Team-4.git
cd Team-4
```

### 14.2 가상환경 생성 및 활성화

Windows PowerShell 기준:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux 기준:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 14.3 데이터 다운로드

Kaggle에서 `US_Accidents_March23.csv` 파일을 다운로드한 뒤 아래 경로에 배치합니다.

```text
data/US_Accidents_March23.csv
```

원본 CSV는 용량이 크기 때문에 GitHub에 포함하지 않습니다.

### 14.4 패키지 설치

노트북 실행 및 모델 학습용 패키지를 설치합니다.

```bash
pip install -r requirements.txt
```

Streamlit 웹앱 실행용 패키지를 설치합니다.

```bash
pip install -r requirements_streamlit.txt
```

지도 기능을 위해 `requirements_streamlit.txt`에는 다음 패키지가 포함되어야 합니다.

```text
folium
streamlit-folium
```

### 14.5 노트북 실행

Jupyter Notebook 또는 VS Code에서 다음 파일을 처음부터 끝까지 실행합니다.

```text
Team4_US_Accidents_Final_Project_revised.ipynb
```

노트북 실행이 완료되면 Streamlit 웹앱 실행에 필요한 파일들이 생성됩니다.

```text
streamlit_artifacts/severity_app_artifacts.joblib
streamlit_artifacts/location_reference.csv
```

### 14.6 Streamlit 웹앱 실행

```bash
streamlit run app.py
```

실행 후 브라우저에서 다음 주소가 열립니다.

```text
http://localhost:8501
```

웹앱에서 미국 지도 위 사고 위치를 클릭한 뒤, 기상 및 도로 환경 정보를 입력하고 **예측하기** 버튼을 누르면 예측 Severity가 출력됩니다.

---

## 15. GitHub 업로드 관련 주의사항

원본 데이터 파일과 가상환경 폴더는 GitHub에 업로드하지 않습니다.

```gitignore
.venv/
venv/
env/
__pycache__/
*.pyc
US_Accidents_March23.csv
US_Accidents*.csv
data/US_Accidents_March23.csv
data/US_Accidents*.csv
.streamlit/
```

업로드 대상 예시는 다음과 같습니다.

```text
README.md
.gitignore
requirements.txt
requirements_streamlit.txt
app.py
Team4_US_Accidents_Final_Project_revised.ipynb
outputs/
figures/
```

`streamlit_artifacts/severity_app_artifacts.joblib`와 `streamlit_artifacts/location_reference.csv`는 웹앱 실행 편의성을 위해 포함할 수 있습니다. 다만 파일 크기가 크거나 팀 제출 기준상 제외해야 하는 경우, 노트북을 실행하여 다시 생성할 수 있습니다.

---

## 16. 한계점 및 향후 개선 방향

본 프로젝트는 사고 발생 시점에서 활용 가능한 변수만 사용하기 위해 `End_Time`, `Distance(mi)`, `End_Lat`, `End_Lng`, `Description` 등 정보 누수 가능성이 있는 변수를 제외했습니다. 이 선택은 실제 예측 상황을 고려한 보수적인 모델링이라는 장점이 있지만, 모델 성능에는 한계로 작용할 수 있습니다.

주요 한계점은 다음과 같습니다.

- `Severity=2` 클래스가 대부분을 차지하는 클래스 불균형 문제
- `Severity 1`, `Severity 4` 등 소수 클래스 예측 성능 부족
- 교통량, 평균 속도, 차종, 사고 유형, 도로 등급, 차로 수 등 핵심 교통 변수 부재
- `Severity 2`와 `Severity 3`처럼 인접 클래스 간 경계가 명확하지 않은 문제
- 원-핫 인코딩된 범주형 변수가 많아 일부 샘플링 기법 적용 시 해석상 주의 필요

향후 개선 방향은 다음과 같습니다.

- 교통량, 속도, 도로 등급, 제한속도 등 외부 교통·도로 데이터 결합
- 4분류 대신 `Low / High` 또는 `Low / Medium / High` 형태의 2분류·3분류 문제 재정의
- `SMOTENC`, `BalancedRandomForest`, `EasyEnsemble` 등 불균형 데이터에 적합한 기법 추가 적용
- `macro_f1` 기준의 하이퍼파라미터 튜닝 강화
- 출퇴근 시간 여부, 주말 여부, 야간 여부, 사고 다발 지역 여부 등 교통공학적 파생 변수 추가

---

## 17. 최종 결론

본 프로젝트에서는 US Accidents 데이터를 활용하여 교통사고 심각도(`Severity`)를 4개 클래스로 분류했습니다. 데이터는 `Severity=2` 클래스에 집중된 불균형 구조를 보였기 때문에 accuracy뿐 아니라 balanced accuracy, macro F1-score, weighted F1-score를 함께 사용했습니다.

최종적으로 `RF Balanced min_samples_leaf=5` 모델이 Validation set의 macro F1-score 기준으로 가장 높은 성능을 보여 최종 모델로 선정되었습니다. 다만 Confusion Matrix 분석 결과, 모델은 다수 클래스인 `Severity 2`에 비해 소수 클래스인 `Severity 1`과 `Severity 4`의 예측 성능이 낮았습니다. 따라서 본 모델은 전체적인 사고 심각도 예측 가능성을 확인하는 데 의미가 있지만, 소수 클래스 예측 성능 개선이 필요한 모델로 해석할 수 있습니다.

또한 Streamlit 웹앱을 구현하여 사용자가 사고 기본 정보, 위치 정보, 기상 정보, 도로 환경 정보를 입력하면 예측 Severity를 확인할 수 있도록 구성했습니다. 지도 기반 위치 자동 입력 기능을 통해 사용자가 미국 지도에서 사고 위치를 선택하면 위도, 경도, 주, 도시, 카운티, 시간대가 자동 설정되도록 하여 입력 편의성을 개선했습니다.

향후에는 외부 교통·도로 데이터 결합, 클래스 재정의, 불균형 처리 기법 고도화, `macro_f1` 기준 하이퍼파라미터 튜닝을 통해 소수 클래스 예측 성능을 개선할 필요가 있습니다.

---
## 18. 팀 정보

- Team: Team 4
- Project: US Accidents 교통사고 Severity 예측 프로젝트
- Task: Multiclass Classification
- Target: `Severity`
- Web App: Streamlit 기반 Severity 예측 웹앱

### 역할 분담

| 팀원 | 담당 영역 | 세부 내용 |
| --- | --- | --- |
| 예상형 | 데이터 전처리 · 모델링 설계 | 데이터 로딩 및 층화 샘플링, 전처리 파이프라인 구성(정보 누수 방지), 모델 학습·평가 함수 설계|
| 윤석민 | 모델 실험 · 성능 분석 | balanced·부스팅 모델 비교, Random Forest 하이퍼파라미터 및 클래스 불균형 처리 실험, 최종 모델 선정, 결과 분석, 발표 진행 |
| 강우석 | 데이터 조사 · EDA 정리 | 데이터셋 출처·라이선스 조사, 입력 변수 의미 정리, EDA 그래프 해석 및 문서화 |
| 정시영 | 문서 작성 · 발표 자료 | README 및 보고서 작성, 한계점·개선 방향 자료조사, 발표 슬라이드 정리 |
