
# ==========================================
# TUAN 4: DANH GIA TAP TEST VA PHAN TICH LOI
# ==========================================

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# Thiet lap chung de giong cac tuan truoc
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def run_week_4():
    # --- 1. DOC VA CHIA DU LIEU (CHUNG MOT SEED) ---
    df = pd.read_csv('data/penguins.csv')
    df = df.dropna(subset=['bill_length_mm']).reset_index(drop=True)
    
    X = df.drop(columns=['species'])
    y = df['species']
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp)
    
    print("=======================================================")
    print("THI NGHIEM 3: SO SANH DAC TRUNG SO vs DAC TRUNG HOAN CHINH")
    print("=======================================================")
    numeric_features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    categorical_features = ['island', 'sex']
    
    # Pipeline 1: Chi dung so do hinh thai (giong tuan 3)
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    preprocessor_num = ColumnTransformer([('num', num_pipe, numeric_features)])
    model_num_only = Pipeline([('prep', preprocessor_num), ('knn', KNeighborsClassifier(n_neighbors=5))])
    
    model_num_only.fit(X_train, y_train)
    acc_num = accuracy_score(y_val, model_num_only.predict(X_val))
    
    # Pipeline 2: Dung ca so do va bien phan loai (dao, gioi tinh)
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore')) # Ma hoa bien phan loai thanh One-Hot
    ])
    preprocessor_all = ColumnTransformer([
        ('num', num_pipe, numeric_features),
        ('cat', cat_pipe, categorical_features)
    ])
    model_all = Pipeline([('prep', preprocessor_all), ('knn', KNeighborsClassifier(n_neighbors=5))])
    
    model_all.fit(X_train, y_train)
    acc_all = accuracy_score(y_val, model_all.predict(X_val))
    
    print(f"Accuracy (Chi dung 4 so do):      {acc_num:.2%}")
    print(f"Accuracy (Them Dao & Gioi tinh):  {acc_all:.2%}")
    print("=> KET LUAN: Viec them bien phan loai co the khong cai thien nhieu hoac tham chi lam giam do chinh xac do chieu khong gian K-NN bi tang len. Ta se dung mo hinh chi chua so do (da luu o Tuan 3) de test cuoi cung.")

    print("\n=======================================================")
    print("DANH GIA CHINH THUC TREN TAP TEST (CHI CHAY 1 LAN)")
    print("=======================================================")
    # Tai mo hinh da toi uu tu Tuan 3
    best_model = joblib.load('models/knn_best_pipeline.pkl')
    
    # Vi mo hinh Tuan 3 chi nhan 4 bien so, ta loc X_test
    X_test_num = X_test[numeric_features]
    
    # Du doan va tinh metric
    y_pred = best_model.predict(X_test_num)
    test_acc = accuracy_score(y_test, y_pred)
    test_macro_f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"Do chinh xac (Accuracy) tren tap Test : {test_acc:.2%}")
    print(f"Chi so Macro-F1 tren tap Test         : {test_macro_f1:.2%}")
    
    # Luu ma tran nham lan
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Ma tran nham lan tren tap Test (Final)')
    plt.savefig('reports/figures/test_confusion_matrix.png')
    print("Da luu Ma tran nham lan vao 'reports/figures/test_confusion_matrix.png'")

    print("\n=======================================================")
    print("THI NGHIEM 4: PHAN TICH O NHAM LAN BANG LANG GIENG GAN NHAT")
    print("=======================================================")
    # Tim cac truong hop du doan sai
    errors_mask = (y_test.values != y_pred)
    X_test_errors = X_test_num[errors_mask]
    y_test_errors = y_test[errors_mask]
    y_pred_errors = y_pred[errors_mask]

    if len(X_test_errors) > 0:
        # Lay truong hop loi dau tien ra phan tich
        idx = X_test_errors.index[0]
        actual_label = y_test_errors.loc[idx]
        pred_label = y_pred_errors[0]
        
        print(f"Phat hien chim canh cut thuoc Dong so {idx} bi du doan sai!")
        print(f" - Thuc te la loai  : {actual_label}")
        print(f" - Mo hinh du doan  : {pred_label}")
        print(f" - Thong so cua no  : {X_test_num.loc[idx].to_dict()}")
        
        # Phan tich lang gieng
        # Buoc 1: Bien doi diem du lieu nay qua Imputer/Scaler cua mo hinh
        X_test_single_trans = best_model[:-1].transform(X_test_num.loc[[idx]])
        
        # Buoc 2: Dung thuat toan KNN de tim hang xom trong tap Train
        knn_step = best_model.named_steps['knn']
        distances, indices = knn_step.kneighbors(X_test_single_trans)
        
        X_train_num = X_train[numeric_features]
        print("\nDanh sach cac lang gieng gan nhat (Trong khong gian da chuan hoa):")
        for i, train_idx in enumerate(indices[0]):
            real_idx = X_train_num.index[train_idx]
            dist = distances[0][i]
            neighbor_label = y_train.loc[real_idx]
            print(f" Lang gieng {i+1}: {neighbor_label} (Khoang cach: {dist:.4f}) | So do: {X_train_num.loc[real_idx].to_dict()}")
            
        print(f"\n=> GIAI THICH: Mo hinh du doan la {pred_label} boi vi da so cac lang gieng gan nhat trong khong gian dac trung lai thuoc ve loai {pred_label}, do ca the nay co cac so do (nhu chieu dai mo, can nang) giao thoa/giong voi loai {pred_label} hon la dong loai cua minh.")
    else:
        print("Tuyet voi! Mo hinh du doan dung 100% tren tap Test, khong co loi nao de phan tich!")

if __name__ == "__main__":
    run_week_4()