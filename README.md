# US Accidents 사고 심각도 분류 프로젝트

## 1. 프로젝트 개요

본 프로젝트는 Kaggle의 **US Accidents (2016–2023)** 데이터를 활용하여 교통사고의 심각도(`Severity`)를 예측하는 지도학습 기반 다중분류 프로젝트이다.

`Severity`는 1, 2, 3, 4의 네 개 클래스로 구성되어 있으며, 본 프로젝트에서는 사고 발생 시점에서 활용 가능하다고 판단되는 시간, 위치, 기상, 도로환경 변수를 사용하여 사고 심각도를 예측하였다.

본 프로젝트의 주요 목표는 다음과 같다.

- 교통 분야와 관련된 지도학습 문제 정의
- 공개 데이터 기반 전처리 및 feature engineering 수행
- 여러 머신러닝 모델을 동일 조건에서 비교
- 클래스 불균형 상황에서 Accuracy 외의 평가 지표 함께 사용
- 최종 모델의 성능과 한계 분석

---

## 2. 데이터셋

| 항목 | 내용 |
|---|---|
| 데이터셋 | US Accidents (2016–2023) |
| 출처 | Kaggle |
| URL | https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents |
| 파일명 | `US_Accidents_March23.csv` |
| 예측 목표 | `Severity` 1~4 다중분류 |
| 원본 데이터 규모 | 약 7,728,394행 |
| 사용 표본 | Severity 기준 층화 샘플링 100,000행 |
| 다운로드 날짜 | 2026-06-__ |
| 라이선스 | Kaggle 데이터셋 페이지에서 확인 후 입력 |

원본 CSV 파일은 용량이 크기 때문에 GitHub 저장소에 포함하지 않는다.  
Kaggle에서 직접 다운로드한 뒤 Google Drive 또는 로컬 `data/` 폴더에 배치하여 실행한다.

---

## 3. 저장소 구조

```text
.
├── README.md
├── .gitignore
├── requirements.txt
├── notebooks/
│   └── Team4_US_Accidents_Final_Project.ipynb
├── outputs/
│   ├── model_validation_results.csv
│   ├── model_test_result.csv
│   ├── rf_depth_results.csv
│   ├── experiment_summary.csv
│   └── feature_importance.csv
├── figures/
│   ├── fig01_severity_distribution.png
│   ├── fig02_original_sample_ratio.png
│   ├── ...
│   └── fig17_nn_loss.png
└── Report/
    └── Submission/
        └── Team1_report.pdf
```

`data/` 폴더와 원본 CSV 파일은 GitHub에 업로드하지 않는다.

---

## 4. 사용 변수

### 4.1 사용한 변수 그룹

| 변수 그룹 | 주요 변수 |
|---|---|
| 시간 변수 | `Start_Time` |
| 위치 변수 | `Start_Lat`, `Start_Lng`, `State`, `County`, `City`, `Timezone` |
| 기상 변수 | `Temperature(F)`, `Wind_Chill(F)`, `Humidity(%)`, `Pressure(in)`, `Visibility(mi)`, `Wind_Speed(mph)`, `Precipitation(in)`, `Weather_Condition` |
| 주야간 변수 | `Sunrise_Sunset`, `Civil_Twilight`, `Nautical_Twilight`, `Astronomical_Twilight` |
| 도로환경 변수 | `Amenity`, `Bump`, `Crossing`, `Give_Way`, `Junction`, `No_Exit`, `Railway`, `Roundabout`, `Station`, `Stop`, `Traffic_Calming`, `Traffic_Signal`, `Turning_Loop` |

### 4.2 제외한 변수

| 제외 변수 | 제외 이유 |
|---|---|
| `End_Time` | 사고 종료 이후 확정되는 정보 |
| `Distance(mi)` | 사고 영향 구간을 직접 반영할 가능성이 있음 |
| `End_Lat`, `End_Lng` | 사고 종료 지점 정보 |
| `Description` | 자연어 텍스트 변수로 별도 NLP 처리가 필요함 |
| `ID`, `Street`, `Zipcode` | 식별자 성격 또는 고유값이 많은 변수 |

---

## 5. 전처리 과정

본 프로젝트의 주요 전처리 과정은 다음과 같다.

1. `Start_Time`을 날짜형으로 변환
2. `Year`, `Month`, `DayOfWeek`, `Hour` 생성
3. `Weather_Condition`을 `Weather_Group`으로 단순화
4. `City`, `County`는 상위 30개 값만 유지하고 나머지는 `Other`로 통합
5. `Precipitation_NA` 변수 생성
6. Boolean 도로환경 변수는 0/1로 변환
7. 수치형 결측치는 훈련 세트 기준 중앙값으로 대체
8. 범주형 결측치는 훈련 세트 기준 최빈값으로 대체
9. 범주형 변수는 `pd.get_dummies()`로 원-핫 인코딩
10. 수치형 변수는 `StandardScaler`로 표준화

---

## 6. 모델링

모든 모델은 동일한 train / validation / test 분할을 사용하였다.

| 구분 | 비율 |
|---|---|
| Train | 약 70% |
| Validation | 약 15% |
| Test | 약 15% |

비교한 모델은 다음과 같다.

| 모델 | 설명 |
|---|---|
| Dummy Classifier | 가장 많은 클래스만 예측하는 기준 모델 |
| Logistic Regression | 선형 기반 분류 모델 |
| Decision Tree | 조건 분할 기반 트리 모델 |
| Random Forest | 여러 결정 트리를 결합한 앙상블 모델 |
| Random Forest Balanced | 클래스 불균형을 고려한 랜덤 포레스트 |
| HistGradientBoosting | 히스토그램 기반 그라디언트 부스팅 모델 |
| Simple Dense Neural Network | 선택 실험으로 수행한 간단한 딥러닝 모델 |

---

## 7. 평가 지표

본 데이터는 `Severity=2` 클래스가 대부분을 차지하는 불균형 데이터이므로 Accuracy만으로 모델 성능을 판단하지 않았다.

사용한 평가 지표는 다음과 같다.

| 지표 | 설명 |
|---|---|
| Accuracy | 전체 데이터 중 맞게 예측한 비율 |
| Balanced Accuracy | 클래스별 Recall의 평균 |
| Macro Precision | 클래스별 Precision의 단순 평균 |
| Macro Recall | 클래스별 Recall의 단순 평균 |
| Macro F1 | 클래스별 F1-score의 단순 평균 |
| Weighted F1 | 클래스 비율을 반영한 F1-score |

최종 모델은 Validation set의 **Macro F1**을 기준으로 선택하였다.

---

## 8. 주요 결과

### 8.1 Validation 성능

| Model | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| Random Forest Balanced | 0.7865 | 0.4294 | 0.4457 | 0.7848 |
| HistGradientBoosting | 0.8235 | 0.3688 | 0.4044 | 0.7891 |
| Decision Tree | 0.8075 | 0.3302 | 0.3560 | 0.7646 |
| Random Forest | 0.8122 | 0.2883 | 0.2917 | 0.7560 |
| Logistic Regression | 0.7990 | 0.2631 | 0.2499 | 0.7257 |
| Dummy | 0.7966 | 0.2500 | 0.2217 | 0.7065 |

### 8.2 최종 선택 모델

Validation set의 Macro F1이 가장 높은 **Random Forest Balanced**를 최종 모델로 선택하였다.

### 8.3 Test 성능

| Model | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| Random Forest Balanced | 0.7859 | 0.4261 | 0.4373 | 0.7846 |

---

## 9. 하이퍼파라미터 실험

Random Forest의 `max_depth`를 변경하며 성능을 비교하였다.

| max_depth | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| 5 | 0.3230 | 0.5364 | 0.2439 | 0.3962 |
| 10 | 0.5140 | 0.5821 | 0.3313 | 0.5957 |
| 15 | 0.6700 | 0.5561 | 0.4034 | 0.7134 |
| 20 | 0.7486 | 0.4877 | 0.4410 | 0.7636 |
| None | 0.7865 | 0.4294 | 0.4457 | 0.7848 |

`max_depth=None`에서 Macro F1이 가장 높게 나타났으나, Balanced Accuracy는 `max_depth=10`에서 가장 높았다.  
따라서 깊은 트리가 모든 클래스의 Recall 균형을 항상 개선하는 것은 아니라는 점을 확인하였다.

---

## 10. 실행 방법

### 10.1 Google Colab 실행

1. Kaggle에서 `US_Accidents_March23.csv` 파일을 다운로드한다.
2. Google Drive에 CSV 파일을 업로드한다.
3. Colab에서 노트북을 연다.
4. 아래 경로를 본인 Google Drive 경로에 맞게 수정한다.

```python
file_path = "/content/drive/MyDrive/US_Accidents_March23.csv"
```

5. 위에서부터 모든 셀을 차례대로 실행한다.
6. 실행 후 결과표는 `outputs/`, 그래프는 `figures/` 폴더에 저장된다.

### 10.2 로컬 실행

로컬에서는 데이터 파일을 `data/` 폴더에 저장하고 아래처럼 경로를 수정한다.

```python
file_path = "./data/US_Accidents_March23.csv"
```

---

## 11. 필요 라이브러리

주요 라이브러리는 다음과 같다.

```text
numpy
pandas
matplotlib
scikit-learn
tensorflow
jupyter
```

설치 명령어:

```bash
pip install -r requirements.txt
```

---

## 12. 재현성 설정

노트북에서는 다음 값을 사용하였다.

```python
RANDOM_STATE = 42
SAMPLE_SIZE = 100_000
```

데이터 분할과 샘플링 과정에서는 `stratify`를 적용하여 Severity 클래스 비율을 유지하였다.

---

## 13. 한계 및 개선 방향

본 프로젝트의 한계는 다음과 같다.

1. `Severity`는 인명피해 정도가 아니라 교통 흐름에 미친 영향 정도에 가까우므로 해석에 주의가 필요하다.
2. 데이터의 클래스 불균형이 크기 때문에 Severity 1과 Severity 4 예측 성능이 낮게 나타났다.
3. 교통량, 제한속도, 도로 등급, 도로 기하구조 등 추가 교통 관련 변수가 포함되지 않았다.
4. 위치 변수는 특정 지역의 데이터 패턴을 반영할 수 있어 일반화에 한계가 있다.

향후 개선 방향은 다음과 같다.

- 소수 클래스에 대한 추가적인 불균형 처리 기법 적용
- 언더샘플링, 오버샘플링, SMOTE 등 샘플링 기법 비교
- 도로 등급, 제한속도, 교통량 등 외부 변수 결합
- 지역별 모델 분리 또는 지역 특성을 반영한 추가 분석
- Severity 4 탐지를 개선하기 위한 별도 이진분류 모델 실험
