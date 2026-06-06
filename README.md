# US Accidents 교통사고 심각도 예측 프로젝트

## 1. 프로젝트 개요 

본 프로젝트는 Kaggle의 **US Accidents (2016-2023)** 데이터를 활용하여 교통사고 심각도(`Severity`)를 예측하는 지도학습 기반 다중분류 프로젝트입니다.

사고 발생 시간, 위치, 기상 조건, 도로 주변 시설 정보를 바탕으로 사고 심각도를 `1`, `2`, `3`, `4`의 네 개 클래스로 분류하는 것을 목표로 합니다.

본 프로젝트는 기계학습 과제 요구사항에 맞추어 다음 내용을 포함합니다

* 공개 데이터셋 활용
* 데이터 출처, 다운로드 날짜, 라이선스 명시
* 10개 이상의 입력 변수 사용
* 1,000개 이상의 인스턴스 사용
* 전처리, 탐색적 데이터 분석, 모델 학습, 평가, 결과 저장
* 여러 기계학습 모델 비교
* 하이퍼파라미터 실험
* 딥러닝 모델 추가 실험

---

## 2. 데이터셋 정보

| 항목              | 내용                                                         |
| --------------- | ---------------------------------------------------------- |
| Dataset         | US Accidents (2016-2023)                                   |
| Source          | Kaggle                                                     |
| URL             | https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents |
| Download Date   | 2026-05-24                                                 |
| License         | CC BY-NC-SA 4.0                                            |
| Target Variable | `Severity`                                                 |
| Task Type       | Multiclass Classification                                  |

> 원본 데이터 파일 `US_Accidents_March23.csv`는 용량이 크기 때문에 GitHub 저장소에 포함하지 않습니다.
> 실행을 위해서는 Kaggle에서 데이터를 다운로드한 뒤 `data/US_Accidents_March23.csv` 경로에 배치해야 합니다.

---

## 3. 프로젝트 구조

```text
Team-4/
├─ README.md
├─ .gitignore
├─ requirements.txt
├─ Team4_US_Accidents_Final_Project_revised.ipynb
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
│  ├─ fig13_validation_score_comparison.png
│  ├─ fig14_confusion_matrix.png
│  ├─ fig15_feature_importance.png
│  ├─ fig16_rf_depth_comparison.png
│  ├─ fig16_1_rf_leaf_comparison.png
│  └─ fig17_nn_loss.png
└─ data/
   └─ US_Accidents_March23.csv
```

단, `data/US_Accidents_March23.csv`는 `.gitignore`를 통해 GitHub 업로드 대상에서 제외합니다.

---

## 4. 사용 변수

본 프로젝트에서는 사고 심각도 예측을 위해 다음과 같은 변수들을 사용했습니다.

### 4.1 시간 변수

원본 `Start_Time`에서 다음 파생 변수를 생성했습니다.

* `Year`
* `Month`
* `DayOfWeek`
* `Hour`

### 4.2 위치 변수

* `Start_Lat`
* `Start_Lng`
* `City`
* `County`
* `State`

### 4.3 기상 변수

* `Temperature(F)`
* `Wind_Chill(F)`
* `Humidity(%)`
* `Pressure(in)`
* `Visibility(mi)`
* `Wind_Direction`
* `Wind_Speed(mph)`
* `Precipitation(in)`
* `Weather_Group`

### 4.4 도로 및 주변 시설 변수

* `Amenity`
* `Bump`
* `Crossing`
* `Give_Way`
* `Junction`
* `No_Exit`
* `Railway`
* `Roundabout`
* `Station`
* `Stop`
* `Traffic_Calming`
* `Traffic_Signal`
* `Turning_Loop`

### 4.5 주야간 관련 변수

* `Sunrise_Sunset`
* `Civil_Twilight`
* `Nautical_Twilight`
* `Astronomical_Twilight`

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

---

## 6. 데이터 분할

모델 학습 및 평가를 위해 데이터를 다음과 같이 분할했습니다.

| 구분             | 비율    |
| -------------- | ----- |
| Train Set      | 약 70% |
| Validation Set | 약 15% |
| Test Set       | 약 15% |

모든 분할 과정에서 `Severity` 클래스 비율이 유지되도록 `stratify` 옵션을 사용했습니다.

---

## 7. 사용 모델

본 프로젝트에서는 다음 모델들을 비교했습니다.

| 모델                          | 설명                            |
| --------------------------- | ----------------------------- |
| Dummy Classifier            | 최빈 클래스만 예측하는 기준 모델            |
| Logistic Regression         | 선형 기반 분류 모델                   |
| Decision Tree               | 트리 기반 단일 분류 모델                |
| Random Forest               | 여러 결정트리를 결합한 앙상블 모델           |
| Random Forest Balanced      | 클래스 불균형을 고려한 Random Forest    |
| HistGradientBoosting        | 히스토그램 기반 Gradient Boosting 모델 |
| Simple Dense Neural Network | 간단한 완전연결 신경망 모델               |

---

## 8. 평가 지표

본 데이터는 `Severity=2` 클래스가 큰 비중을 차지하는 불균형 데이터입니다.
따라서 단순 정확도만 사용하지 않고 다음 평가 지표를 함께 사용했습니다.

* Accuracy
* Balanced Accuracy
* Macro Precision
* Macro Recall
* Macro F1-score
* Weighted F1-score

최종 모델은 Validation set의 `macro_f1`을 주요 기준으로 선택했습니다.

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

## 11. 주요 결과 파일

노트북 실행 후 다음 결과 파일들이 생성됩니다.

### 11.1 CSV 결과 파일

```text
outputs/model_validation_results.csv
outputs/model_validation_results_with_nn.csv
outputs/model_test_result.csv
outputs/rf_depth_results.csv
outputs/rf_leaf_results.csv
outputs/feature_importance.csv
outputs/experiment_summary.csv
```

### 11.2 시각화 파일

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

---

## 12. 실행 방법

### 12.1 저장소 클론

```bash
git clone https://github.com/2026-1st/Team-4.git
cd Team-4
```

### 12.2 데이터 다운로드

Kaggle에서 `US_Accidents_March23.csv` 파일을 다운로드한 뒤 아래 경로에 배치합니다.

```text
data/US_Accidents_March23.csv
```

데이터 파일은 용량 문제로 GitHub에 포함하지 않습니다.

### 12.3 패키지 설치

```bash
pip install -r requirements.txt
```

만약 `requirements.txt`가 없거나 개별 설치가 필요한 경우 다음 패키지를 설치합니다.

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

### 12.4 노트북 실행

Jupyter Notebook 또는 VS Code에서 다음 파일을 실행합니다.

```text
Team4_US_Accidents_Final_Project_revised.ipynb
```

노트북의 데이터 경로는 기본적으로 다음과 같이 설정되어 있습니다.

```python
file_path = "data/US_Accidents_March23.csv"
```

데이터를 다른 위치에 저장한 경우, 해당 경로에 맞게 `file_path`만 수정하면 됩니다.

---

## 13. GitHub 업로드 관련 주의사항

원본 데이터 파일은 GitHub에 업로드하지 않습니다.
`.gitignore`에는 다음 항목이 포함되어야 합니다.

```gitignore
US_Accidents_March23.csv
US_Accidents*.csv
data/US_Accidents_March23.csv
data/US_Accidents*.csv
```

반대로, 결과 파일을 공유하기 위해 다음 폴더는 GitHub에 포함할 수 있습니다.

```text
outputs/
figures/
```

---

## 14. 최종 결론

본 프로젝트에서는 US Accidents 데이터를 활용하여 교통사고 심각도(`Severity`)를 4개 클래스로 분류했습니다.
데이터는 `Severity=2` 클래스에 집중된 불균형 구조를 보였기 때문에 단순 정확도뿐만 아니라 `balanced accuracy`, `macro F1-score`, `weighted F1-score`를 함께 사용했습니다.

여러 모델을 비교한 결과, Validation set에서 `macro F1-score`가 가장 높은 모델을 최종 모델로 선정했습니다.
또한 Test set 평가를 통해 최종 모델이 학습 데이터에만 과도하게 맞춰진 것은 아닌지 확인했습니다.

변수 중요도 분석 결과 사고 발생 연도, 위치 정보, 월, 기압, 체감온도, 기온, 시간대 등의 변수가 사고 심각도 예측에 상대적으로 큰 영향을 미치는 것으로 나타났습니다.
다만 소수 클래스의 예측 성능은 여전히 제한적이므로, 향후에는 SMOTE, class weight 세부 조정, XGBoost/LightGBM 등의 추가 모델을 적용하여 클래스 불균형 문제를 개선할 필요가 있습니다.

---

## 15. 팀 정보

* Team: Team 4
* Project: US Accidents 교통사고 심각도 예측 프로젝트
* Task: Multiclass Classification
* Target: `Severity`
