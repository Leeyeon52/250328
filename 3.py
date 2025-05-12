import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 파일 경로 설정
train_path = "train.csv"
test_path = "test.csv"
sample_submission_path = "sample_submission.csv"
output_path = "submission.csv"

# 파일 존재 여부 확인
def check_file_exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ 파일을 찾을 수 없습니다: {path}. 파일이 존재하는지 확인하세요.")

for path in [train_path, test_path, sample_submission_path]:
    check_file_exists(path)

# 데이터 로드
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
df_sample = pd.read_csv(sample_submission_path)

# URL 컬럼명 변경 (자동 감지)
def rename_url_column(df):
    for col in df.columns:
        if 'url' in col.lower():
            df.rename(columns={col: 'URL'}, inplace=True)
            return df
    raise KeyError("❌ 'URL' 컬럼을 찾을 수 없습니다. 파일을 확인하세요.")

df_train = rename_url_column(df_train)
df_test = rename_url_column(df_test)

# 결측값 및 중복 제거
df_train = df_train.drop_duplicates().dropna()
df_test = df_test.drop_duplicates().dropna()

# 특징(X)와 타겟(y) 분리
if 'malicious' not in df_train.columns:
    raise KeyError("❌ 'malicious' 컬럼이 train.csv에 없습니다. 파일을 확인하세요.")
X_texts = df_train['URL'].astype(str)
y = df_train['malicious']

# TF-IDF 벡터화
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3), stop_words='english')  # Changed to n-grams (1,3)
X_tfidf = vectorizer.fit_transform(X_texts)
test_tfidf = vectorizer.transform(df_test['URL'].astype(str))

# 데이터 분할
X_train, X_val, y_train, y_val = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# 랜덤 포레스트 모델 학습 및 하이퍼파라미터 튜닝
rf_model = RandomForestClassifier(random_state=42)

# GridSearchCV로 하이퍼파라미터 튜닝
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 50, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best parameters found: ", grid_search.best_params_)

