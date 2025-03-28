import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as pit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

# 데이터 로드
bike_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
sampleSubmission = pd.read_csv('sampleSubmission.csv')

# datetime 형태로 변경
bike_df['datetime'] = pd.to_datetime(bike_df['datetime'])
test_df['datetime'] = pd.to_datetime(test_df['datetime'])

# 특성 생성
bike_df['month'] = pd.to_datetime(bike_df['datetime']).dt.month
test_df['month'] = pd.to_datetime(test_df['datetime']).dt.month
bike_df['hour'] = pd.to_datetime(bike_df['datetime']).dt.hour
test_df['hour'] = pd.to_datetime(test_df['datetime']).dt.hour

# 모델 훈련 (단순화된 모델로 진행)

# 특성과 타겟 변수 정의 (일부 피쳐만 사용하여 성능 낮추기)
X = bike_df[['temp', 'hour']]  # 최소한의 특성만 사용
y = bike_df['count']

# 훈련/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 스케일링 (선형 회귀에서는 스케일링이 중요할 수 있음)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 모델 초기화
model = LinearRegression()

# 교차 검증을 통한 RMSE 계산
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=10, scoring='neg_mean_squared_error')

# 교차 검증에서의 RMSE 계산 (평균값을 구할 때 부호 변경)
rmse_cv = np.sqrt(-cv_scores.mean())

# 결과 출력
print(f"Cross-Validation RMSE: {rmse_cv:.4f}")

# 모델 훈련
model.fit(X_train_scaled, y_train)

# 예측
y_pred = model.predict(X_test_scaled)

# 테스트 데이터에 대한 RMSE 계산
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))

# 결과 출력
print(f"Test Set RMSE: {rmse_test:.4f}")

# 테스트 데이터에서 예측
X_test = test_df[['temp', 'hour']]  # 테스트 데이터에서도 동일한 특성 사용
X_test_scaled = scaler.transform(X_test)  # 스케일링 적용
test_df['count'] = model.predict(X_test_scaled)

# 제출 파일 생성
submission = test_df[['datetime', 'count']]
submission.loc[:, 'datetime'] = submission['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

# 예측 결과를 제출 파일로 저장
submission.to_csv('submission2.csv', index=False)
