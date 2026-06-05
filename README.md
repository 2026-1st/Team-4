# US Accidents 다중분류 프로젝트 — 변경 이력 및 한계점 대처

본 문서는 초기 버전(v2)에서 최종 버전(v6)까지의 변경 사항과, 식별된
한계점에 대한 대처 내용을 정리한다.

---

## 1. 버전별 변경 요약

| 버전 | 핵심 변경 |
|------|-----------|
| v2 | 초기 버전 (Colab 기반). 기본 EDA, 전처리, 6개 모델 비교, RF 하이퍼파라미터 분석 |
| v3 | 한계점 보강을 위한 분석 추가 (누수 수정, XGBoost 공정 비교, SMOTE, Year 실험) |
| v4 | 실행 환경을 Colab → VS Code(로컬)로 이전 |
| v5 | Year ablation 전처리기 공유 버그 수정 |
| v6 | Year dtype(int32) 미탐지 문제 수정 → ablation 정상 작동 |

---

## 2. v2 대비 주요 변경 사항

### 2-1. 분석 내용 추가 (v3에서 반영)

**(1) 전처리 정보 누수 제거**
v2는 City/County의 상위 30개 범주를 전체 데이터 기준으로 계산하여
train/test 분할 이전에 적용했다. 이 경우 valid/test의 분포 정보가
train 전처리에 일부 새어드는 미세한 누수가 발생한다.
v6에서는 상위 범주 목록을 **train 데이터에서만 학습**한 뒤 valid/test에
동일하게 적용하도록 분할 이후로 이동했다. (섹션 21-1)

**(2) 모델 비교 공정성 확보**
v2는 Random Forest와 Logistic Regression Balanced에만 클래스 불균형
보정(class_weight='balanced')을 적용했고, XGBoost에는 보정이 없었다.
v6에서는 XGBoost에 `compute_sample_weight('balanced')`로 만든
sample_weight를 적용하여 동일 조건으로 비교한다.
(모델명: XGBoost → XGBoost Balanced)

**(3) SMOTE 오버샘플링 비교 모델 추가**
클래스 불균형에 직접 대응하기 위해, train 데이터에만 SMOTE를 적용한
모델(HistGB + SMOTE)을 추가했다. 누수 방지를 위해 imbalanced-learn의
Pipeline을 사용하여 오버샘플링이 학습 시에만 동작하도록 했다. (섹션 33-1)

**(4) Year 변수 영향 실험(ablation) 추가**
Year 변수를 포함한 모델과 제외한 모델의 성능을 비교하여, 모델이
연도별 정보에 의존하는 정도를 정량적으로 측정한다. (섹션 33-2)

### 2-2. 실행 환경 변경 (v4에서 반영)

- `from google.colab import drive` 및 `drive.mount()` 제거
- Google Drive 경로 → 로컬 다운로드 폴더 자동 탐색으로 변경
- 패키지 설치를 `sys.executable -m pip` 방식으로 변경 (로컬 다중 Python
  환경에서 설치-import 불일치 방지)

### 2-3. 버그 수정 (v5, v6에서 반영)

- v5: Year ablation에서 두 모델이 동일한 전처리기 객체를 공유하여
  결과가 완전히 같게 나오던 문제를, 각 파이프라인이 새 전처리기를
  사용하도록 수정.
- v6: Year가 int32 타입이어서 `select_dtypes(['int64','float64'])`에
  잡히지 않던 문제를 `include=['number']`로 수정. 이후 ablation 정상 작동.

---

## 3. 한계점 및 대처

각 한계에 대해 "대처 방식"과 "결과"를 함께 정리한다. 대부분의 한계는
완전히 제거되지 않으며, 대신 **시도와 측정을 통해 정직하게 확인**하였다.

### 3-1. 클래스 불균형
- **현상**: Severity 2가 약 79.7%, Severity 1은 0.87%, Severity 4는 2.6%.
- **대처**: class_weight='balanced'(RF, LR), XGBoost sample_weight,
  SMOTE 오버샘플링을 적용하여 비교.
- **결과**: 소수 클래스 예측은 여전히 낮다 (Test 기준 Severity 1 f1 0.13,
  Severity 4 f1 0.17). SMOTE의 Macro F1(0.343)도 Random Forest(0.385)를
  넘지 못했다. 표본 자체가 부족하여 기법만으로는 근본 개선이 어렵다는
  점을 데이터로 확인하였다.
- **상태**: 완화 시도 → 근본 해결 불가 → 한계로 확정.

### 3-2. Severity 정의의 한계
- **현상**: Severity는 실제 부상·사망 여부가 아니라 사고의 교통 영향
  수준에 가깝다.
- **대처**: 데이터셋 정의 자체의 문제로 코드로 해결 불가.
- **상태**: 해석상의 한계로 명시. 본 모델은 의학적 심각도가 아니라
  영향 수준을 예측하는 것으로 해석해야 한다.

### 3-3. 정보 누수 방지를 위한 변수 제외
- **현상**: Distance(mi), End_Time, Duration_Min 등을 제외하면 성능이
  낮아질 수 있음.
- **대처**: 이는 한계라기보다 의도적 설계 선택이다. 해당 변수들은 사고
  발생 시점에 알 수 없으므로, 성능을 일부 포기하더라도 실사용 가능한
  모델을 구성하였다. 추가로 City/County 전처리상의 누수도 제거하였다.
- **상태**: 올바른 설계 선택으로 유지 + 전처리 누수 추가 수정.

### 3-4. 지역 편향 가능성
- **현상**: 미국 전체 데이터지만 특정 주·도시의 기록이 많을 수 있음.
- **대처**: Feature Importance로 위치 변수의 영향력을 측정.
- **결과**: Start_Lng, Start_Lat가 중요도 1·2위로 확인되어, 모델이
  위치에 크게 의존함을 정량적으로 보였다. 편향을 제거한 것이 아니라
  편향 우려가 실제로 근거 있음을 확인하였다.
- **상태**: 해결 불가 → 증거 확보.

### 3-5. 시간 변화 문제
- **현상**: 연도별 데이터 수집 방식, 교통량, 정책 변화 등이 모델에
  영향을 줄 수 있음.
- **대처**: Year 변수 ablation 실험 수행.
- **결과**: Year 제거 시 Macro F1이 0.444 → 0.385, Balanced Accuracy가
  0.426 → 0.374로 하락. Year가 예측에 기여하나, 이는 모델이 연도별
  수집 편향을 학습하고 있을 가능성을 시사한다. 미래/타 기간으로의
  일반화 관점에서는 한계로 해석해야 한다.
- **상태**: 해결 불가 → 정량화.

---

## 4. 핵심 결과 (참고)

- **최종 모델**: Random Forest (Validation Macro F1 기준 선택)
- **Test 성능**: Accuracy 0.758, Macro F1 0.405, Balanced Accuracy 0.407
- **주목할 점**: Dummy Baseline의 Accuracy(0.797)가 Random Forest(0.758)
  보다 높지만, Macro F1은 RF(0.405)가 Dummy(0.222)의 약 2배다. 이는
  클래스 불균형 상황에서 Accuracy가 아니라 Macro F1으로 모델을 평가해야
  하는 이유를 직접 보여준다.
