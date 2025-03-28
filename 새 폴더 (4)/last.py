import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# 데이터 로드
bike_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# datetime 형식 변환
bike_df['datetime'] = pd.to_datetime(bike_df['datetime'])
test_df['datetime'] = pd.to_datetime(test_df['datetime'])

# feature engineering
bike_df['hour'] = bike_df['datetime'].dt.hour
bike_df['month'] = bike_df['datetime'].dt.month
bike_df['weekday'] = bike_df['datetime'].dt.weekday

test_df['hour'] = test_df['datetime'].dt.hour
test_df['month'] = test_df['datetime'].dt.month
test_df['weekday'] = test_df['datetime'].dt.weekday

# NaN값 처리
test_df.fillna(test_df.mean(), inplace=True)

# 모델 학습 및 예측
features = ['hour', 'month', 'weekday', 'temp', 'humidity', 'windspeed']
X_train = bike_df[features]
y_train = bike_df['count']
X_test = test_df[features]

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
test_df['count'] = rf_model.predict(X_test)

# 음수값 클리핑
test_df['count'] = test_df['count'].apply(lambda x: max(0, x))

# 제출 파일 생성
submission = test_df[['datetime', 'count']]
submission['datetime'] = submission['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
submission.to_csv('submission25.csv', index=False)