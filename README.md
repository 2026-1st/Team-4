# US Accidents 교통사고 심각도 예측 프로젝트

## 1. 프로젝트 개요

본 프로젝트는 Kaggle의 **US Accidents (2016–2023)** 데이터를 활용하여 사고 발생 시간, 위치, 기상 조건, 도로 주변 시설물 정보를 바탕으로 교통사고 심각도(`Severity`)를 예측하는 지도학습 기반 다중분류 프로젝트입니다.

사고 심각도는 `1, 2, 3, 4`의 네 개 클래스로 구성되어 있으며, 본 프로젝트에서는 이를 예측하기 위해 여러 기계학습 모델을 동일한 데이터 분할과 평가 지표로 비교합니다.

## 2. 문제 정의

- 문제 유형: 지도학습 기반 다중분류
- 예측 대상: `Severity`
- 클래스: `1`, `2`, `3`, `4`
- 입력 변수: 시간, 위치, 기상, 도로환경, 도로 주변 시설물 관련 변수
- 목표: 사고 당시 조건을 기반으로 사고 심각도를 예측하고, 심각도에 영향을 주는 주요 요인을 분석

## 3. 데이터셋

본 프로젝트는 Kaggle 공개 데이터셋인 **US Accidents (2016–2023)**를 사용합니다.

| 항목 | 내용 |
|---|---|
| 데이터셋 | US Accidents (2016–2023) |
| 출처 | Kaggle |
| 원본 규모 | 약 7,728,394건 |
| 수집 기간 | 2016년 2월 ~ 2023년 3월 |
| 예측 목표 | 교통사고 심각도 `Severity` |
| 라이선스 | CC BY-NC-SA 4.0 |
| 다운로드 날짜 | 2026-05-08 |

데이터 출처:  
https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents

> 원본 CSV 파일은 용량이 크기 때문에 GitHub 저장소에는 포함하지 않습니다.  
> 실행 시 Kaggle에서 `US_Accidents_March23.csv` 파일을 다운로드한 뒤 `data/` 폴더에 넣어야 합니다.

## 4. 프로젝트 구조

```text
Team4_US_Accidents_Project/
│
├── README.md
│
├── data/
│   └── US_Accidents_March23.csv        # 원본 데이터, GitHub 업로드 제외 권장
│
├── notebooks/
│   └── Team4_US_Accidents_Multiclass_Project_split_explained.ipynb
│
├── src/
│   └── team4_us_accidents_multiclass_project_split_explained.py
│
├── figures/
│   └── eda_and_model_results.png
│
└── reports/
    └── final_report.pdf
```

## 5. 사용한 주요 변수

본 프로젝트에서는 원본 데이터의 여러 변수 중 사고 심각도 예측에 활용 가능한 변수를 선별하여 사용했습니다.

### 시간 관련 변수

- `Start_Time`
- `End_Time`
- `Year`
- `Month`
- `DayOfWeek`
- `Hour`
- `Is_Weekend`
- `Season`
- `Time_Period`
- `Duration_Min`

### 위치 관련 변수

- `Start_Lat`
- `Start_Lng`
- `State`
- `City`
- `County`

### 기상 관련 변수

- `Temperature(F)`
- `Humidity(%)`
- `Pressure(in)`
- `Visibility(mi)`
- `Wind_Speed(mph)`
- `Weather_Condition`

### 도로환경 및 시설물 관련 변수

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

## 6. 분석 과정

### 6.1 데이터 전처리

- 필요한 컬럼만 선택하여 메모리 사용량 감소
- 날짜형 변수 변환
- 사고 지속시간 `Duration_Min` 생성
- 월, 요일, 시간대, 계절 변수 생성
- 결측치 처리
- 범주형 변수 인코딩
- 수치형 변수 스케일링
- `Severity` 기준 층화 샘플링 적용

### 6.2 데이터 분할

모든 모델이 동일한 조건에서 비교될 수 있도록 같은 random seed와 같은 데이터 분할을 사용했습니다.

- Train: 60%
- Validation: 20%
- Test: 20%

또한 클래스 비율이 유지되도록 `stratify` 옵션을 사용했습니다.

### 6.3 모델링

다음 세 가지 기계학습 모델을 비교했습니다.

1. Logistic Regression
2. Random Forest
3. HistGradientBoostingClassifier

선택 과제로 TensorFlow 기반 MLP 모델 코드도 추가했습니다.

## 7. 평가 지표

본 프로젝트는 다중분류 문제이므로 단순 정확도뿐만 아니라 클래스 불균형을 고려할 수 있는 지표를 함께 사용했습니다.

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Weighted F1-score
- Confusion Matrix

특히 `Macro F1-score`는 각 클래스를 동일한 비중으로 평가하므로, 클래스 불균형이 존재하는 사고 심각도 예측 문제에서 중요한 지표로 활용했습니다.

## 8. 시각화

보고서 요구사항을 충족하기 위해 EDA 및 모델 분석 과정에서 8개 이상의 그래프를 생성했습니다.

주요 시각화 항목은 다음과 같습니다.

- 사고 심각도 분포
- 월별 사고 건수
- 요일별 사고 건수
- 시간대별 사고 건수
- 주별 사고 건수
- 사고 지속시간 분포
- 기상 조건별 사고 분포
- 수치형 변수 상관관계 히트맵
- 모델별 성능 비교 그래프
- 혼동행렬
- 하이퍼파라미터 변화에 따른 성능 변화
- 변수 중요도 그래프

## 9. 실행 방법

### 9.1 패키지 설치

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

선택 과제인 MLP 모델까지 실행하려면 TensorFlow도 설치합니다.

```bash
pip install tensorflow
```

### 9.2 데이터 준비

Kaggle에서 `US_Accidents_March23.csv` 파일을 다운로드한 뒤 아래 경로에 저장합니다.

```text
data/US_Accidents_March23.csv
```

### 9.3 노트북 실행

Colab 또는 Jupyter Notebook에서 아래 파일을 실행합니다.

```text
notebooks/Team4_US_Accidents_Multiclass_Project_split_explained.ipynb
```

### 9.4 파이썬 스크립트 실행

VS Code 또는 터미널에서 실행할 경우 다음 명령어를 사용합니다.

```bash
python src/team4_us_accidents_multiclass_project_split_explained.py
```

## 10. 실행 시 주의사항

원본 데이터는 약 772만 건으로 매우 크기 때문에 로컬 환경이나 Colab 무료 버전에서는 실행 시간이 오래 걸릴 수 있습니다.

코드의 `SAMPLE_SIZE` 값을 조정하면 실행 속도를 줄일 수 있습니다.

```python
SAMPLE_SIZE = 100_000
```

실행이 느릴 경우 다음과 같이 줄일 수 있습니다.

```python
SAMPLE_SIZE = 50_000
```

또는

```python
SAMPLE_SIZE = 30_000
```

## 11. 프로젝트 기대 효과

본 프로젝트를 통해 사고 심각도와 관련된 시간, 기상, 위치, 도로환경 요인을 데이터 기반으로 파악할 수 있습니다.  
이는 향후 위험구간 관리, 도로 시설 개선, 교통안전 정책 수립, 운전자 경고 시스템 설계 등에 기초자료로 활용될 수 있습니다.

## 12. 프로젝트 한계

- 미국 교통사고 데이터를 사용하므로 국내 교통환경에 바로 일반화하기 어렵습니다.
- `Severity`는 실제 인명피해 정도가 아니라 교통 흐름에 미치는 영향 정도를 나타내므로 해석에 주의가 필요합니다.
- 클래스 불균형이 존재하여 단순 정확도만으로 모델을 평가하기 어렵습니다.
- 원본 데이터 크기가 매우 커서 샘플링 과정이 필요합니다.

## 13. 팀원 역할

| 이름 | 역할 |
|---|---|
| 정시영 | 데이터 수집, 데이터 명세 정리, 결측치·이상치 처리 |
| 예상형 | EDA 시각화, 변수 분포 및 클래스 불균형 분석 |
| 강우석 | 모델 구현, 성능 비교, 하이퍼파라미터 조정 |
| 윤석민 | 결과 해석, Proposal·발표자료·README 작성 |

## 14. 참고문헌

- Moosavi, S. (2023). *US Accidents (2016–2023)*. Kaggle.  
  https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents

- Moosavi, S., Samavatian, M. H., Parthasarathy, S., & Ramnath, R. (2019).  
  *A Countrywide Traffic Accident Dataset*. arXiv:1906.05409.
