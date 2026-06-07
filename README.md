# US Accidents 교통사고 Severity 예측 프로젝트

## 1. 프로젝트 개요 

본 프로젝트는 Kaggle의 **US Accidents (2016-2023)** 데이터를 활용하여 교통사고 심각도(`Severity`)를 예측하는 지도학습 기반 다중분류 프로젝트입니다.


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

## 3. 프로젝트 구조

```text
Team-4/
├─ README.md
├─ .gitignore
├─ requirements.txt
├─ requirements_streamlit.txt
├─ app.py
├─ Team4_US_Accidents_Final_Project_revised.ipynb
├─ streamlit_artifacts/
│  └─ severity_app_artifacts.joblib
├─ outputs/
│  ├─ model_validation_results.csv
│  ├─ model_validation_results_with_nn.csv
│  ├─ model_test_result.csv
│  ├─ rf_depth_results.csv
│  ├─ rf_leaf_results.csv
│  ├─ feature_importance.csv
│  ├─ experiment_summary.csv
│  └─ imbalance_experiment_results.csv
├─ figures/
│  ├─ fig01_severity_distribution.png
│  ├─ fig02_original_sample_ratio.png
│  ├─ fig04_accidents_by_hour.png
│  ├─ fig06_accidents_by_weather.png
│  ├─ fig08_signal_severity_ratio.png
│  ├─ fig10_visibility_by_severity.png
│  ├─ fig12_road_feature_ratio.png
│  ├─ fig13_validation_score_comparison.png
│  ├─ fig14_confusion_matrix.png
│  ├─ fig15_feature_importance.png
│  ├─ fig16_rf_depth_comparison.png
│  ├─ fig16_1_rf_leaf_comparison.png
│  ├─ fig17_nn_loss.png
│  └─ imbalance_experiment_comparison.png
└─ data/
   └─ US_Accidents_March23.csv
```

---

## 4. 사용 변수

### 4.1 시간 변수

원본 `Start_Time`에서 다음 파생 변수를 생성했습니다.

- `Month`
- `DayOfWeek`
- `Hour`

### 4.2 위치 변수

- `Start_Lat`
- `Start_Lng`
- `City`
- `County`
- `State`
- `Timezone`

### 4.3 기상 변수

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

### 4.4 도로 및 주변 시설 변수

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

### 4.5 주야간 관련 변수

- `Sunrise_Sunset`
- `Civil_Twilight`
- `Nautical_Twilight`
- `Astronomical_Twilight`

---

## 5. 전처리 과정

노트북에서는 다음과 같은 전처리를 수행했습니다.

1. 필요한 변수만 선택하여 데이터 로드
2. `Severity` 클래스 분포 확인
3. 전체 데이터에서 100,000개 샘플을 `Severity` 기준 층화 샘플링
4. `Start_Time`을 날짜형으로 변환 후 `Month`, `DayOfWeek`, `Hour` 생성
5. `Weather_Condition`을 단순화하여 `Weather_Group` 생성
6. `City`, `County`는 학습 데이터 기준 상위 30개 값만 유지하고 나머지는 `Other`로 처리
7. `Precipitation_NA` 변수를 추가하여 강수량 결측 여부 반영
8. Boolean 도로 환경 변수를 0/1 정수형으로 변환
9. 수치형 변수는 학습 데이터의 중앙값으로 결측치 대체
10. 범주형 변수는 학습 데이터의 최빈값으로 결측치 대체
11. 범주형 변수는 One-Hot Encoding 적용
12. 수치형 변수는 StandardScaler로 표준화

Streamlit 웹앱에서도 노트북 학습 시 저장한 전처리 정보를 `severity_app_artifacts.joblib`에서 불러와 동일한 방식으로 입력값을 변환합니다.

---

## 6. 데이터 분할

모델 학습 및 평가를 위해 데이터를 다음과 같이 분할했습니다.

| 구분 | 비율 |
| --- | --- |
| Train Set | 약 70% |
| Validation Set | 약 15% |
| Test Set | 약 15% |

모든 분할 과정에서 `Severity` 클래스 비율이 유지되도록 `stratify` 옵션을 사용했습니다.

---

## 7. 사용 모델 및 실험

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

추가로 클래스 불균형 처리를 위해 다음 실험을 수행했습니다.

- Undersampling
- Manual class weight
- RandomOverSampler
- SMOTE

---

## 8. 평가 지표

본 데이터는 `Severity=2` 클래스가 큰 비중을 차지하는 불균형 데이터입니다. 따라서 단순 정확도만 사용하지 않고 다음 평가 지표를 함께 사용했습니다.

- Accuracy
- Balanced Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Weighted F1-score

최종 모델은 Validation set의 `macro_f1`을 주요 기준으로 선택했습니다.

---

## 9. 주요 Validation 결과

최종 전처리 및 실험 결과, Validation set의 `macro_f1` 기준으로 가장 높은 성능을 보인 모델은 다음과 같습니다.

| Model | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: | ---: |
| RF Balanced min_samples_leaf=5 | 0.7481 | 0.4308 | 0.4095 | 0.7584 |

불균형 처리 실험에서는 일부 모델의 `balanced_accuracy`가 상승했지만, `macro_f1` 기준으로는 `RF Balanced min_samples_leaf=5`가 가장 우수했습니다. 따라서 본 프로젝트에서는 해당 모델을 최종 모델로 선정했습니다.

---

## 10. Streamlit 웹앱 기능

본 프로젝트에는 학습된 모델을 활용하여 사용자가 직접 입력한 사고 정보로 Severity를 예측하는 Streamlit 웹앱이 포함되어 있습니다.

### 10.1 입력 화면 구성

웹앱에서 사용자는 다음 정보를 입력합니다.

```text
사고 기본 정보
- 사고 날짜
- 사고 시간
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

### 10.2 예측 흐름

```text
사용자 입력
→ app.py에서 입력값 수집
→ streamlit_artifacts/severity_app_artifacts.joblib 로드
→ 학습 당시와 동일한 전처리 적용
→ 최종 선택 모델로 Severity 예측
→ 웹 화면에 결과 출력
```

---

## 11. 실행 방법

### 11.1 저장소 클론

```bash
git clone https://github.com/2026-1st/Team-4.git
cd Team-4
```

### 11.2 데이터 다운로드

Kaggle에서 `US_Accidents_March23.csv` 파일을 다운로드한 뒤 아래 경로에 배치합니다.

```text
data/US_Accidents_March23.csv
```

### 11.3 패키지 설치

```bash
pip install -r requirements.txt
pip install -r requirements_streamlit.txt
```

### 11.4 노트북 실행

Jupyter Notebook 또는 VS Code에서 다음 파일을 처음부터 끝까지 실행합니다.

```text
Team4_US_Accidents_Final_Project_revised.ipynb
```

노트북 실행이 완료되면 Streamlit 웹앱용 모델 파일이 생성됩니다.

```text
streamlit_artifacts/severity_app_artifacts.joblib
```

### 11.5 Streamlit 웹앱 실행

```bash
streamlit run app.py
```

실행 후 브라우저에서 다음 주소가 열립니다.

```text
http://localhost:8501
```

---

## 12. GitHub 업로드 관련 주의사항

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

`streamlit_artifacts/severity_app_artifacts.joblib`는 모델 파일 크기와 팀 제출 방식에 따라 포함 여부를 결정합니다. 포함하지 않는 경우, 사용자는 노트북 마지막 저장 셀을 실행하여 파일을 생성해야 합니다.

---

## 13. 최종 결론

본 프로젝트에서는 US Accidents 데이터를 활용하여 교통사고 심각도(`Severity`)를 4개 클래스로 분류했습니다. 데이터는 `Severity=2` 클래스에 집중된 불균형 구조를 보였기 때문에 accuracy뿐 아니라 balanced accuracy, macro F1-score, weighted F1-score를 함께 사용했습니다.

최종적으로 `RF Balanced min_samples_leaf=5` 모델이 Validation set의 macro F1-score 기준으로 가장 높은 성능을 보여 최종 모델로 선정되었습니다. 또한 Streamlit 웹앱을 구현하여 사용자가 사고 기본 정보, 위치 정보, 기상 정보, 도로 환경 정보를 직접 입력하면 예측 Severity를 확인할 수 있도록 구성했습니다.

향후에는 추가적인 불균형 처리 기법, XGBoost/LightGBM 튜닝, 소수 클래스 예측 개선 방법을 적용하여 macro F1-score와 Severity 4 예측 성능을 개선할 필요가 있습니다.

---

## 14. 팀 정보

- Team: Team 4
- Project: US Accidents 교통사고 Severity 예측 프로젝트
- Task: Multiclass Classification
- Target: `Severity`
- Web App: Streamlit 기반 Severity 예측 웹앱
