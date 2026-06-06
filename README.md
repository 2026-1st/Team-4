# US Accidents 교통사고 Severity 예측 프로젝트

## 1. 프로젝트 개요

본 프로젝트는 Kaggle의 **US Accidents (2016-2023)** 데이터를 활용하여 교통사고 심각도(`Severity`)를 예측하는 지도학습 기반 다중분류 프로젝트입니다.

사고 발생 시간, 위치, 기상 조건, 도로 주변 시설 정보를 바탕으로 사고 심각도를 `1`, `2`, `3`, `4`의 네 개 클래스로 분류합니다. 또한 학습된 모델을 활용하여 사용자가 직접 사고 정보를 입력하면 웹 화면에서 예측 결과를 확인할 수 있도록 **Streamlit 기반 예측 웹앱**을 추가했습니다.

본 프로젝트는 기계학습 과제 요구사항에 맞추어 다음 내용을 포함합니다.

- 공개 데이터셋 활용
- 데이터 출처, 다운로드 날짜, 라이선스 명시
- 10개 이상의 입력 변수 사용
- 1,000개 이상의 인스턴스 사용
- 전처리, 탐색적 데이터 분석, 모델 학습, 평가, 결과 저장
- 여러 기계학습 모델 비교
- 하이퍼파라미터 실험
- 딥러닝 모델 추가 실험
- 학습된 모델을 활용한 Streamlit 웹 예측 기능 구현

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
│  └─ experiment_summary.csv
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
│  └─ fig17_nn_loss.png
└─ data/
   └─ US_Accidents_March23.csv
```

`data/US_Accidents_March23.csv`는 `.gitignore`를 통해 GitHub 업로드 대상에서 제외합니다.  
`streamlit_artifacts/severity_app_artifacts.joblib`는 노트북 실행 후 생성되는 Streamlit 예측용 모델 파일입니다.

---

## 4. 사용 변수

본 프로젝트에서는 사고 심각도 예측을 위해 다음과 같은 변수들을 사용했습니다.

### 4.1 시간 변수

원본 `Start_Time`에서 다음 파생 변수를 생성했습니다.

- `Year`
- `Month`
- `DayOfWeek`
- `Hour`

Streamlit 웹앱에서는 사용자가 입력한 `사고 시간`, `월` 값을 바탕으로 임시 `Start_Time`을 구성하고, 동일한 방식으로 시간 파생 변수를 생성합니다.

### 4.2 위치 변수

- `Start_Lat`
- `Start_Lng`
- `City`
- `County`
- `State`

Streamlit 웹앱에서는 사용자 입력 간소화를 위해 `State`, `City`를 직접 입력받습니다. 위도, 경도 등 사용자가 입력하지 않는 값은 학습 데이터의 중앙값 또는 최빈값으로 대체됩니다.

### 4.3 기상 변수

- `Temperature(F)`
- `Wind_Chill(F)`
- `Humidity(%)`
- `Pressure(in)`
- `Visibility(mi)`
- `Wind_Direction`
- `Wind_Speed(mph)`
- `Precipitation(in)`
- `Weather_Group`

Streamlit 웹앱에서는 `날씨`, `기온`, `습도`, `가시거리`를 주요 입력값으로 사용합니다.

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

Streamlit 웹앱에서는 사용자가 쉽게 입력할 수 있도록 `신호등 여부`, `교차로 여부`를 선택하도록 구성했습니다.

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
4. `Start_Time`을 날짜형으로 변환 후 시간 변수 생성
5. `Weather_Condition`을 단순화하여 `Weather_Group` 생성
6. `City`, `County`는 상위 30개 값만 유지하고 나머지는 `Other`로 처리
7. `Precipitation_NA` 변수를 추가하여 강수량 결측 여부 반영
8. Boolean 변수를 0/1 정수형으로 변환
9. 수치형 변수는 중앙값으로 결측치 대체
10. 범주형 변수는 최빈값으로 결측치 대체
11. 범주형 변수는 One-Hot Encoding 적용
12. 수치형 변수는 StandardScaler로 표준화

Streamlit 웹앱에서도 노트북 학습 시 사용한 전처리 정보를 `severity_app_artifacts.joblib`에 저장한 뒤 동일한 방식으로 입력값을 변환합니다.

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

## 7. 사용 모델

본 프로젝트에서는 다음 모델들을 비교했습니다.

| 모델 | 설명 |
| --- | --- |
| Dummy Classifier | 최빈 클래스만 예측하는 기준 모델 |
| Logistic Regression | 선형 기반 분류 모델 |
| Decision Tree | 트리 기반 단일 분류 모델 |
| Random Forest | 여러 결정트리를 결합한 앙상블 모델 |
| Random Forest Balanced | 클래스 불균형을 고려한 Random Forest |
| HistGradientBoosting | 히스토그램 기반 Gradient Boosting 모델 |
| Simple Dense Neural Network | 간단한 완전연결 신경망 모델 |

최종 모델은 Validation set에서 `macro_f1`이 가장 높은 모델을 기준으로 선정했습니다.

---

## 8. 평가 지표

본 데이터는 `Severity=2` 클래스가 큰 비중을 차지하는 불균형 데이터입니다. 따라서 단순 정확도만 사용하지 않고 다음 평가 지표를 함께 사용했습니다.

- Accuracy
- Balanced Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Weighted F1-score

최종 모델 선정에는 `macro_f1`을 주요 기준으로 사용했습니다. 이는 소수 클래스의 예측 성능도 함께 고려하기 위함입니다.

---

## 9. 하이퍼파라미터 실험

Random Forest Balanced 모델을 중심으로 다음 하이퍼파라미터 실험을 수행했습니다.

### 9.1 `max_depth` 실험

```python
max_depth = [5, 10, 15, 20, None]
```

### 9.2 `min_samples_leaf` 실험

```python
min_samples_leaf = [1, 3, 5, 10]
```

실험 결과는 다음 파일에 저장됩니다.

```text
outputs/rf_depth_results.csv
outputs/rf_leaf_results.csv
```

---

## 10. 딥러닝 추가 실험

기계학습 모델과 비교하기 위해 간단한 Dense Neural Network를 추가로 실험했습니다.

모델 구조는 다음과 같습니다.

```text
Input Layer
Dense(64, activation="relu")
Dropout(0.3)
Dense(4, activation="softmax")
```

딥러닝 모델의 Validation 결과는 다음 파일에 저장됩니다.

```text
outputs/nn_validation_result.csv
outputs/model_validation_results_with_nn.csv
```

---

## 11. Streamlit 웹앱 기능

본 프로젝트에는 학습된 모델을 활용하여 사용자가 직접 입력한 사고 정보로 Severity를 예측하는 Streamlit 웹앱이 포함되어 있습니다.

### 11.1 입력 화면 구성

웹앱에서 사용자는 다음 값을 입력합니다.

```text
사고 시간: 18
월: 7
주: CA
도시: Los Angeles
날씨: Rain
기온(F): 65
습도(%): 80
가시거리(mi): 3
신호등 여부: Yes
교차로 여부: Yes
```

이후 **예측하기** 버튼을 누르면 다음과 같은 결과가 출력됩니다.

```text
결과: 예측 Severity = 3
```

모델이 확률 예측을 지원하는 경우, Severity별 예측 확률도 함께 출력됩니다.

### 11.2 Streamlit 예측 흐름

```text
사용자 입력
→ app.py에서 입력값 수집
→ 노트북에서 저장한 전처리 정보 로드
→ 학습 당시와 동일한 전처리 적용
→ 최종 선택 모델로 Severity 예측
→ 웹 화면에 결과 출력
```

### 11.3 Streamlit 관련 파일

| 파일 | 설명 |
| --- | --- |
| `app.py` | Streamlit 웹앱 실행 파일 |
| `requirements_streamlit.txt` | Streamlit 실행에 필요한 패키지 목록 |
| `streamlit_artifacts/severity_app_artifacts.joblib` | 학습된 모델과 전처리 정보를 저장한 파일 |

---

## 12. 주요 결과 파일

노트북 실행 후 다음 결과 파일들이 생성됩니다.

### 12.1 CSV 결과 파일

```text
outputs/model_validation_results.csv
outputs/model_validation_results_with_nn.csv
outputs/model_test_result.csv
outputs/rf_depth_results.csv
outputs/rf_leaf_results.csv
outputs/feature_importance.csv
outputs/experiment_summary.csv
```

### 12.2 시각화 파일

```text
figures/fig01_severity_distribution.png
figures/fig02_original_sample_ratio.png
figures/fig04_accidents_by_hour.png
figures/fig06_accidents_by_weather.png
figures/fig08_signal_severity_ratio.png
figures/fig10_visibility_by_severity.png
figures/fig12_road_feature_ratio.png
figures/fig13_validation_score_comparison.png
figures/fig14_confusion_matrix.png
figures/fig15_feature_importance.png
figures/fig16_rf_depth_comparison.png
figures/fig16_1_rf_leaf_comparison.png
figures/fig17_nn_loss.png
```

### 12.3 Streamlit 모델 파일

```text
streamlit_artifacts/severity_app_artifacts.joblib
```

이 파일은 노트북 마지막 부분의 Streamlit 웹앱용 모델 저장 셀을 실행하면 생성됩니다.

---

## 13. 실행 방법

### 13.1 저장소 클론

```bash
git clone https://github.com/2026-1st/Team-4.git
cd Team-4
```

### 13.2 데이터 다운로드

Kaggle에서 `US_Accidents_March23.csv` 파일을 다운로드한 뒤 아래 경로에 배치합니다.

```text
data/US_Accidents_March23.csv
```

데이터 파일은 용량 문제로 GitHub에 포함하지 않습니다.

### 13.3 패키지 설치

기본 노트북 실행용 패키지를 설치합니다.

```bash
pip install -r requirements.txt
```

만약 `requirements.txt`가 없거나 개별 설치가 필요한 경우 다음 패키지를 설치합니다.

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

Streamlit 웹앱 실행을 위해서는 다음 패키지도 설치합니다.

```bash
pip install -r requirements_streamlit.txt
```

### 13.4 노트북 실행

Jupyter Notebook 또는 VS Code에서 다음 파일을 실행합니다.

```text
Team4_US_Accidents_Final_Project_revised.ipynb
```

노트북의 데이터 경로는 기본적으로 다음과 같이 설정되어 있습니다.

```python
file_path = "data/US_Accidents_March23.csv"
```

데이터를 다른 위치에 저장한 경우, 해당 경로에 맞게 `file_path`만 수정하면 됩니다.

### 13.5 Streamlit 예측용 모델 파일 생성

노트북을 끝까지 실행한 뒤 마지막에 추가된 **Streamlit 웹앱용 모델 및 전처리 정보 저장** 셀을 실행합니다. 실행이 완료되면 다음 파일이 생성됩니다.

```text
streamlit_artifacts/severity_app_artifacts.joblib
```

### 13.6 Streamlit 웹앱 실행

VS Code 터미널 또는 명령 프롬프트에서 프로젝트 폴더로 이동한 뒤 다음 명령어를 실행합니다.

```bash
streamlit run app.py
```

실행 후 브라우저에서 다음 주소가 자동으로 열립니다.

```text
http://localhost:8501
```

웹 화면에서 사고 정보를 입력하고 **예측하기** 버튼을 누르면 예측된 Severity가 출력됩니다.

---

## 14. GitHub 업로드 관련 주의사항

원본 데이터 파일은 GitHub에 업로드하지 않습니다. `.gitignore`에는 다음 항목이 포함되어야 합니다.

```gitignore
US_Accidents_March23.csv
US_Accidents*.csv
data/US_Accidents_March23.csv
data/US_Accidents*.csv
```

반대로, 결과 파일과 Streamlit 앱 실행에 필요한 다음 파일은 GitHub에 포함할 수 있습니다.

```text
README.md
app.py
requirements.txt
requirements_streamlit.txt
Team4_US_Accidents_Final_Project_revised.ipynb
outputs/
figures/
```

`streamlit_artifacts/severity_app_artifacts.joblib`는 모델 파일 크기와 팀 제출 방식에 따라 포함 여부를 결정하면 됩니다. 포함하지 않는 경우, 사용자는 노트북 마지막 저장 셀을 실행하여 직접 생성해야 합니다.

수정한 파일을 GitHub에 올릴 때는 다음 명령어를 사용할 수 있습니다.

```bash
git status
git add README.md app.py requirements_streamlit.txt Team4_US_Accidents_Final_Project_revised.ipynb
git commit -m "Add Streamlit severity prediction app"
git push origin <branch-name>
```

`streamlit_artifacts`까지 올릴 경우에는 다음 명령어를 추가로 사용합니다.

```bash
git add streamlit_artifacts/severity_app_artifacts.joblib
```

---

## 15. 최종 결론

본 프로젝트에서는 US Accidents 데이터를 활용하여 교통사고 심각도(`Severity`)를 4개 클래스로 분류했습니다. 데이터는 `Severity=2` 클래스에 집중된 불균형 구조를 보였기 때문에 단순 정확도뿐만 아니라 `balanced accuracy`, `macro F1-score`, `weighted F1-score`를 함께 사용했습니다.

여러 모델을 비교한 결과, Validation set에서 `macro F1-score`가 가장 높은 모델을 최종 모델로 선정했습니다. 또한 Test set 평가를 통해 최종 모델이 학습 데이터에만 과도하게 맞춰진 것은 아닌지 확인했습니다.

추가적으로 Streamlit 웹앱을 구현하여 모델 활용성을 높였습니다. 사용자는 사고 시간, 월, 주, 도시, 날씨, 기온, 습도, 가시거리, 신호등 여부, 교차로 여부를 입력하고 예측 버튼을 누르면 학습된 모델이 사고 심각도 등급을 예측합니다.

다만 소수 클래스의 예측 성능은 여전히 제한적일 수 있으므로, 향후에는 SMOTE, class weight 세부 조정, XGBoost/LightGBM 등의 추가 모델을 적용하여 클래스 불균형 문제를 개선할 필요가 있습니다. 또한 실제 서비스 수준으로 확장하기 위해서는 입력 변수 자동 수집, 지도 기반 위치 입력, 교통량 및 속도 데이터 연계 등을 추가할 수 있습니다.

---

## 16. 팀 정보

- Team: Team 4
- Project: US Accidents 교통사고 Severity 예측 프로젝트
- Task: Multiclass Classification
- Target: `Severity`
- Web App: Streamlit 기반 Severity 예측 웹앱
