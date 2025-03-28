import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as pit
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import randint, uniform

# 데이터 로드
bike_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
sampleSubmission = pd.read_csv('sampleSubmission.csv')

print(bike_df.shape)
print(test_df.shape)

print(bike_df.shape)
bike_df.head()

print(test_df.shape)
test_df.head()

print(sampleSubmission.shape)
sampleSubmission.head()

bike_df.info()

# datetime 형태로 변경
bike_df['datetime'] = pd.to_datetime(bike_df['datetime'])
test_df['datetime'] = pd.to_datetime(test_df['datetime'])
bike_df.info()

bike_df.isnull().sum()

bike_df.describe()

df = bike_df 
print(df.shape)

print(df.columns)

season_demand = df.groupby('season')['count'].agg(['sum','mean'])

season_demand.columns = ["total_rentals", "average_rentals"]

season_demand

# Set graph style
sns.set_style("whitegrid")

# Set figure size
fig, axes = pit.subplots(2, 2, figsize=(14, 10))

# 1. Bike demand by season
sns.barplot(x="season", y="count", data=bike_df, ax=axes[0, 0], hue="season", palette="coolwarm")
axes[0, 0].set_title("Bike Demand by Season")
axes[0, 0].set_xlabel("Season (1: Spring, 2: Summer, 3: Fall, 4: Winter)")
axes[0, 0].set_ylabel("Rental Count")

# 2. Bike demand by temperature (scatter plot)
sns.scatterplot(x="temp", y="count", data=bike_df, ax=axes[0, 1], alpha=0.5, color="blue")
axes[0, 1].set_title("Bike Demand by Temperature")
axes[0, 1].set_xlabel("Temperature (°C)")
axes[0, 1].set_ylabel("Rental Count")

# 3. Bike demand by perceived temperature (scatter plot)
sns.scatterplot(x="atemp", y="count", data=bike_df, ax=axes[1, 0], alpha=0.5, color="red")
axes[1, 0].set_title("Bike Demand by Perceived Temperature")
axes[1, 0].set_xlabel("Perceived Temperature (°C)")
axes[1, 0].set_ylabel("Rental Count")

# 4. Bike demand by wind speed (scatter plot)
sns.scatterplot(x="windspeed", y="count", data=bike_df, ax=axes[1, 1], alpha=0.5, color="green")
axes[1, 1].set_title("Bike Demand by Wind Speed")
axes[1, 1].set_xlabel("Wind Speed")
axes[1, 1].set_ylabel("Rental Count")

# Adjust layout
pit.tight_layout()
pit.show()

def con_year_month(datetime):
    return "{0}-{1}".format(datetime.year, datetime.month)

bike_df["year_month"] = bike_df["datetime"].apply(con_year_month)
test_df["year_month"] = test_df["datetime"].apply(con_year_month)

print(bike_df.shape)
bike_df[["datetime", "year_month"]].head()

bike_df['month'] = pd.to_datetime(bike_df['datetime']).dt.month

pd.crosstab(bike_df['season'],bike_df['month'])

np.random.seed(42)
bike_df["predicted_count"] = bike_df["count"] * np.random.uniform(0.8, 1.2, size=len(bike_df))

# Figure 
fig, axes = pit.subplots(2, 2, figsize=(14, 10))

# Bar Plot
sns.barplot(x="season", y="count", hue="season", data=bike_df, ax=axes[0, 0], palette="coolwarm")
axes[0, 0].set_title("Bike Demand by Season")
axes[0, 0].set_xlabel("Season (1: Spring, 2: Summer, 3: Fall, 4: Winter)")
axes[0, 0].set_ylabel("Rental Count")
axes[0, 0].legend().set_visible(False)  

# Histogram
sns.histplot(bike_df["count"], bins=30, kde=True, color="purple", ax=axes[0, 1])
axes[0, 1].set_title("Distribution of Bike Rentals")
axes[0, 1].set_xlabel("Rental Count")
axes[0, 1].set_ylabel("Frequency")

# Predicted vs Actual Plot
sns.scatterplot(x=bike_df["count"], y=bike_df["predicted_count"], alpha=0.5, color="blue", label="Predicted vs Actual", ax=axes[1, 0])

# 완벽한 예측 (y=x 선)
axes[1, 0].plot([bike_df["count"].min(), bike_df["count"].max()], 
                [bike_df["count"].min(), bike_df["count"].max()], 
                color="red", linestyle="--", label="Perfect Prediction (y = x)")

axes[1, 0].set_title("Predicted vs Actual Plot")
axes[1, 0].set_xlabel("Actual Rental Count")
axes[1, 0].set_ylabel("Predicted Rental Count")
axes[1, 0].legend()
axes[1, 0].grid(True)

# Correlation Heatmap
corr_matrix = bike_df[['season', 'weather', 'temp', 'count', 'atemp', 'windspeed']].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=axes[1, 1])
axes[1, 1].set_title("Feature Correlation Heatmap")

# 레이아웃 조정
pit.tight_layout()
pit.show()

bike_df['hour'] = pd.to_datetime(bike_df['datetime']).dt.hour

# 시간에 따른 자전거 수요 분석 (시각화)
bike_df['hour'] = pd.to_datetime(bike_df['datetime']).dt.hour
fig, axes = pit.subplots(2, 1, figsize=(10, 8))
sns.pointplot(x='hour', y='count', hue='weather', data=bike_df, ax=axes[0], palette="Set2")
sns.pointplot(x='hour', y='count', hue='season', data=bike_df, ax=axes[1], palette="Set2")

# 이상치 처리 (IQR 방법을 사용하여 이상치를 제거)
Q1 = bike_df[['season', 'weather', 'atemp', 'windspeed', 'count','temp']].quantile(0.25)
Q3 = bike_df[['season', 'weather', 'atemp', 'windspeed', 'count','temp']].quantile(0.75)
IQR = Q3 - Q1

# IQR 범위 밖의 값들을 이상치로 간주하여 제거
bike_df = bike_df[~((bike_df[['season', 'weather', 'atemp', 'windspeed', 'count','temp']] < (Q1 - 1.5 * IQR)) | 
                     (bike_df[['season', 'weather', 'atemp', 'windspeed', 'count','temp']] > (Q3 + 1.5 * IQR))).any(axis=1)]

fig, axes = pit.subplots(6, 1, figsize = (12, 10))

sns.boxplot(data = bike_df, x="season", ax=axes[0])
sns.boxplot(data = bike_df, x="weather", ax=axes[1])
sns.boxplot(data = bike_df, x="atemp", ax=axes[2])
sns.boxplot(data = bike_df, x="windspeed", ax=axes[3])
sns.boxplot(data = bike_df, x="temp", ax=axes[4])
sns.boxplot(data = bike_df, x="count", ax=axes[5])

# 이상치 제거 후 박스플롯
fig, axes = pit.subplots(6, 1, figsize=(12, 8))
sns.boxplot(data=bike_df, x="season", ax=axes[0])
sns.boxplot(data=bike_df, x="weather", ax=axes[1])
sns.boxplot(data=bike_df, x="atemp", ax=axes[2])
sns.boxplot(data=bike_df, x="windspeed", ax=axes[3])
sns.boxplot(data=bike_df, x="temp", ax=axes[4])
sns.boxplot(data=bike_df, x="count", ax=axes[5])

# 모델 훈련 (선형 회귀, 릿지, 라쏘, 랜덤 포레스트)

# 특성과 타겟 변수 정의
X = bike_df[['season', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'hour']]
y = bike_df['count']

# 훈련/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 초기화
linear_model = LinearRegression()
ridge_model = Ridge()
lasso_model = Lasso()
rf_model = RandomForestRegressor(random_state=42)

# RandomizedSearchCV를 통한 하이퍼파라미터 튜닝

# Random Forest 하이퍼파라미터 설정
rf_param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20),
    'max_features': ['sqrt', 'log2', None]
}

rf_search = RandomizedSearchCV(rf_model, rf_param_dist, n_iter=100, cv=3, random_state=42, n_jobs=-1)
rf_search.fit(X_train, y_train)

# 최적 하이퍼파라미터 출력
print("Best parameters found by RandomizedSearchCV:", rf_search.best_params_)

# 최적 모델로 예측
y_pred_rf = rf_search.best_estimator_.predict(X_test)

# RMSE 계산
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

# 결과 출력
print(f"Random Forest RMSE (after RandomizedSearchCV): {rmse_rf:.4f}")

#eature engineering (datetime에서 필요한 정보 추출)
bike_df.loc[:,'datetime'] = pd.to_datetime(bike_df.loc[:,'datetime'])
test_df['datetime'] = pd.to_datetime(test_df['datetime'])

bike_df.loc[:,'hour'] = bike_df.loc[:,'datetime'].dt.hour
bike_df.loc[:,'month'] = bike_df.loc[:,'datetime'].dt.month
bike_df.loc[:, 'weekday'] = bike_df['datetime'].dt.weekday

test_df['hour'] = test_df['datetime'].dt.hour
test_df['month'] = test_df['datetime'].dt.month
test_df['weekday'] = test_df['datetime'].dt.weekday

# 사용할 변수 선택 (예시로 몇 가지 컬럼만 사용)
features = ['hour', 'month', 'weekday', 'temp', 'humidity', 'windspeed']
X_train = bike_df[features]
y_train = bike_df['count']

# 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 예측
X_test = test_df[features]
test_df['count'] = model.predict(X_test)

# 제출 파일 생성
submission = test_df[['datetime', 'count']]
submission.loc[:,'datetime'] = submission['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')  # datetime 포맷 맞추기

# 예측 결과를 제출 파일로 저장
submission.to_csv('submission3.csv', index=False)