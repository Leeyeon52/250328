import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import randint, uniform
import lightgbm as lgb
from sklearn.preprocessing import OneHotEncoder

# 데이터 로드
bike_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
sampleSubmission = pd.read_csv('sampleSubmission.csv')

# datetime 형태로 변경
bike_df['datetime'] = pd.to_datetime(bike_df['datetime'])
test_df['datetime'] = pd.to_datetime(test_df['datetime'])

# Feature Engineering 추가
bike_df['hour'] = bike_df['datetime'].dt.hour
bike_df['month'] = bike_df['datetime'].dt.month
bike_df['weekday'] = bike_df['datetime'].dt.weekday
bike_df['humidity_squared'] = bike_df['humidity'] ** 2
bike_df['windspeed_squared'] = bike_df['windspeed'] ** 2
bike_df['is_rush_hour'] = bike_df['hour'].apply(lambda x: 1 if x in [7, 8, 9, 17, 18, 19] else 0)

# One-Hot Encoding 적용
ohe = OneHotEncoder(sparse_output=False, drop='first')
weather_encoded = ohe.fit_transform(bike_df[['weather']])
weather_encoded_df = pd.DataFrame(weather_encoded, columns=ohe.get_feature_names_out(['weather']))
bike_df = pd.concat([bike_df, weather_encoded_df], axis=1)

# 동일한 변환을 test 데이터에도 적용
test_df['hour'] = test_df['datetime'].dt.hour
test_df['month'] = test_df['datetime'].dt.month
test_df['weekday'] = test_df['datetime'].dt.weekday
test_df['humidity_squared'] = test_df['humidity'] ** 2
test_df['windspeed_squared'] = test_df['windspeed'] ** 2
test_df['is_rush_hour'] = test_df['hour'].apply(lambda x: 1 if x in [7, 8, 9, 17, 18, 19] else 0)
weather_encoded_test = ohe.transform(test_df[['weather']])
weather_encoded_test_df = pd.DataFrame(weather_encoded_test, columns=ohe.get_feature_names_out(['weather']))
test_df = pd.concat([test_df, weather_encoded_test_df], axis=1)

# 사용할 변수 정의
features = ['hour', 'month', 'weekday', 'temp', 'humidity', 'windspeed', 'humidity_squared', 'windspeed_squared', 'is_rush_hour'] + list(weather_encoded_df.columns)
X = bike_df[features]
y = np.log1p(bike_df['count'])  # log1p 변환 적용

# 훈련/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 정의 및 학습
lgb_model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=10, random_state=42)
lgb_model.fit(X_train, y_train)

# 예측 및 RMSE 평가
y_pred = np.expm1(lgb_model.predict(X_test))  # log1p 변환 해제
rmse = np.sqrt(mean_squared_error(np.expm1(y_test), y_pred))
print(f'LightGBM RMSE: {rmse:.4f}')

# 최종 테스트 데이터 예측 및 저장
test_df['count'] = np.expm1(lgb_model.predict(test_df[features]))
test_df['count'] = np.clip(test_df['count'], 0, None)  # 음수 방지
submission = test_df[['datetime', 'count']]
submission.to_csv('submission1.csv', index=False)
