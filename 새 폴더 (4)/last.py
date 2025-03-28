import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as pit
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from scipy.stats import randint, uniform

# 데이터 로드
bike_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
sampleSubmission = pd.read_csv('sampleSubmission.csv')

# 데이터 처리 및 변환 코드 (이전 코드와 동일)

# 모델 훈련 (LightGBM 사용)
X = bike_df[['season', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'hour']]
y = bike_df['count']

# 훈련/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LGBM 모델 초기화
lgb_model = lgb.LGBMRegressor(random_state=42)

# 하이퍼파라미터 튜닝을 위한 랜덤 서치
lgb_param_dist = {
    'num_leaves': randint(20, 150),
    'learning_rate': uniform(0.01, 0.2),
    'n_estimators': randint(50, 200),
    'max_depth': randint(3, 20),
    'min_child_samples': randint(1, 20)
}

lgb_search = RandomizedSearchCV(lgb_model, lgb_param_dist, n_iter=100, cv=3, random_state=42, n_jobs=-1)
lgb_search.fit(X_train, y_train)

# 최적 하이퍼파라미터 출력
print("Best parameters found by RandomizedSearchCV:", lgb_search.best_params_)

# 최적 모델로 예측
y_pred_lgb = lgb_search.best_estimator_.predict(X_test)

# RMSE 계산
rmse_lgb = np.sqrt(mean_squared_error(y_test, y_pred_lgb))

# 결과 출력
print(f"LightGBM RMSE (after RandomizedSearchCV): {rmse_lgb:.4f}")

# 테스트 데이터에 대한 예측
X_test_data = test_df[['season', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'hour']]
test_df['count'] = lgb_search.best_estimator_.predict(X_test_data)

# 제출 파일 생성
submission = test_df[['datetime', 'count']]
submission.loc[:,'datetime'] = submission['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')  # datetime 포맷 맞추기

# 예측 결과를 제출 파일로 저장
submission.to_csv('submission_lgbm.csv4', index=False)
