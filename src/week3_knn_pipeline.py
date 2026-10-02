
# ==========================================
# TUAN 3: PIPELINE MO HINH CHINH VA THI NGHIEM
# ==========================================

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os

# Thiet lap chung
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def load_and_split():
    df = pd.read_csv('data/penguins.csv')
    df = df.dropna(subset=['bill_length_mm']).reset_index(drop=True)
    
    # Cho Thi nghiem 1 & 2, chung ta chi dung 4 dac trung so do (theo yeu cau cot loi)
    features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    X = df[features]
    y = df['species']
    
    # Chia y het nhu Tuan 2 de dam bao tinh nhat quan (70/15/15)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def run_week_3():
    print("--- DANG TAI VA CHIA DU LIEU ---")
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split()
    
    print("\n=======================================================")
    print("THI NGHIEM 1: SO SANH KHONG CHUAN HOA vs CO CHUAN HOA")
    print("=======================================================")
    # 1.1 Pipeline KHONG chuan hoa
    pipe_unscaled = Pipeline([
        ('imputer', SimpleImputer(strategy='median')), # Dien gia tri thieu bang trung vi
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ])
    pipe_unscaled.fit(X_train, y_train)
    acc_unscaled = accuracy_score(y_val, pipe_unscaled.predict(X_val))
    
    # 1.2 Pipeline CO chuan hoa (StandardScaler)
    pipe_scaled = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()), # Chuan hoa dua cac dac trung ve cung thang do
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ])
    pipe_scaled.fit(X_train, y_train)
    acc_scaled = accuracy_score(y_val, pipe_scaled.predict(X_val))
    
    print(f"Do chinh xac tren tap Validation (KHONG chuan hoa): {acc_unscaled:.2%}")
    print(f"Do chinh xac tren tap Validation (CO chuan hoa)   : {acc_scaled:.2%}")
    print("=> KET LUAN: k-NN la thuat toan dua tren khoang cach. Neu khong chuan hoa,")
    print("   dac trung co don vi lon (body_mass_g) se ap dao cac dac trung nho (mm).")

    print("\n=======================================================")
    print("THI NGHIEM 2: TIM THAM SO K TOT NHAT BANG CROSS-VALIDATION")
    print("=======================================================")
    # Dinh nghia cac tham so can thu nghiem (K tu 1 den 15, Euclid vs Manhattan, uniform vs distance)
    param_grid = {
        'knn__n_neighbors': [ 5, 7, 9, 11, 13, 15],
        'knn__metric': ['euclidean', 'manhattan'],
        'knn__weights': ['uniform', 'distance']
    }
    
    # Su dung GridSearchCV tren tap Train (chia lam 5 fold)
    # Pipeline co scaler duoc dung lam estimator goc
    grid_search = GridSearchCV(
        estimator=pipe_scaled,
        param_grid=param_grid,
        cv=5, 
        scoring='accuracy',
        n_jobs=-1 # Dung toan bo CPU de chay nhanh
    )
    
    print("Dang chay Cross-Validation tren tap Train... (Khong dung vao tap Test!)")
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print("\nKet qua tot nhat tu Cross-Validation:")
    print(f"- So luong lang gieng (K): {grid_search.best_params_['knn__n_neighbors']}")
    print(f"- Khoang cach (metric): {grid_search.best_params_['knn__metric']}")
    print(f"- Trong so (weights): {grid_search.best_params_['knn__weights']}")
    print(f"- Cross-Validation Accuracy: {grid_search.best_score_:.2%}")
    
    # Kiem tra lai mo hinh tot nhat tren tap Validation
    val_pred = best_model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    print(f"\n=> Do chinh xac cua mo hinh tot nhat tren tap Validation: {val_acc:.2%}")
    
    print("\n--- LUU MO HINH (PIPELINE) ---")
    if not os.path.exists('models'):
        os.makedirs('models')
    
    joblib.dump(best_model, 'models/knn_best_pipeline.pkl')
    print("Da luu toan bo Pipeline (bao gom Imputer, Scaler, KNN) vao 'models/knn_best_pipeline.pkl'")
    print("Hoan thanh Tuan 3!")

if __name__ == "__main__":
    run_week_3()