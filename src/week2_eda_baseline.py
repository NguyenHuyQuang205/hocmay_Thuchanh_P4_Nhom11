
# ==========================================
# TUAN 2: KHAM PHA DU LIEU (EDA) VA BASELINE
# ==========================================

# Buoc 1: Import thu vien can thiet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Buoc 2: Thiet lap random_seed (Co dinh de tai lap ket qua)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def run_week_2():
    print("--- BUOC 3 & 4: DOC DU LIEU ---")
    # Dung file da tai ve o Tuan 1
    df = pd.read_csv('data/penguins.csv')
    print("Doc du lieu thanh cong!")
    
    print("\n--- BUOC 5: KIEM TRA KICH THUOC ---")
    print(f"Kich thuoc du lieu: {df.shape}")
    print(f"So dong: {df.shape[0]}, So cot: {df.shape[1]}")
    
    print("\n--- BUOC 6: KIEM TRA THONG TIN VA MISSING VALUES ---")
    print("Thong tin chung:")
    print(df.info())
    print("\nSo luong gia tri thieu (NaN) moi cot:")
    print(df.isna().sum())
    print("Tong missing: ", df.isna().sum().sum())
    
    print("\nSo dong trung lap: ", df.duplicated().sum())
    
    print("\n--- BUOC 6.1: LAM SACH CO BAN ---")
    # Dac thu cua Palmer Penguins co 2 dong bi thieu hoan toan so do.
    # Ta se loai bo 2 dong nay vi khong co can cu de dien (Imputer).
    # Nhung dong thieu gioi tinh (sex) se duoc xu ly o Tuan 3 bang Imputer tren tap Train.
    df = df.dropna(subset=['bill_length_mm']).reset_index(drop=True)
    print(f"Kich thuoc sau khi xoa 2 dong trong hoan toan: {df.shape}")
    
    print("\n--- BUOC 7: KHAM PHA DU LIEU (EDA) tren toan tap ---")
    # Kiem tra phan bo nhan (Target)
    print("Phan bo cac loai chim canh cut (Target):")
    print(df['species'].value_counts())
    print(df['species'].value_counts(normalize=True) * 100)
    
    # Ve bieu do (De tai yeu cau ve bieu do phuc vu quyet dinh)
    # Luu y: Do dang chay script, bieu do se pop-up len. Tat cua so bieu do de code chay tiep.
    print("\nDang ve bieu do phan bo... (Vui long tat cua so bieu do de tiep tuc)")
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x='bill_length_mm', y='bill_depth_mm', hue='species')
    plt.title('Phan bo Chieu dai vs Do day mo chim')
    plt.savefig('reports/figures/eda_bill_scatter.png') # Luu bieu do vao thu muc reports
    # plt.show() # Mo comment neu ban muon xem truc tiep tren man hinh
    print("Da luu bieu do tai reports/figures/eda_bill_scatter.png")

    print("\n--- BUOC 8: TAO X, y VA CHIA TRAIN/VALID/TEST ---")
    # Tao X, y
    X = df.drop(columns=['species'])
    y = df['species']
    
    # Chia theo ty le 70% Train, 15% Valid, 15% Test
    # Buoc 8.1: Tach 70% Train, 30% con lai cho Val+Test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    # Buoc 8.2: Tach 30% thanh 15% Val va 15% Test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
    )
    
    print(f"Kich thuoc tap Train: {X_train.shape}")
    print(f"Kich thuoc tap Validation: {X_val.shape}")
    print(f"Kich thuoc tap Test: {X_test.shape}")
    
    print("\nKiem tra ty le class tap Train:")
    print(y_train.value_counts(normalize=True))

    print("\n--- BUOC 9: XAY DUNG BASELINE (DummyClassifier) ---")
    # De tai bat buoc dung DummyClassifier thay vi luat thu cong
    baseline_model = DummyClassifier(strategy='most_frequent')
    
    # Huan luyen baseline tren tap Train
    baseline_model.fit(X_train, y_train)
    
    # Du doan tren tap Test
    baseline_pred = baseline_model.predict(X_test)
    
    # Do Accuracy
    baseline_acc = accuracy_score(y_test, baseline_pred)
    print(f"\n=> Baseline Accuracy tren tap Test: {baseline_acc:.2%}")
    print("\nBao cao phan loai Baseline:")
    print(classification_report(y_test, baseline_pred, zero_division=0))
    
    # Luu ma tran nham lan cua baseline
    cm = confusion_matrix(y_test, baseline_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=baseline_model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Ma tran nham lan - Baseline')
    plt.savefig('reports/figures/baseline_confusion_matrix.png')
    print("Da luu ma tran nham lan Baseline tai reports/figures/baseline_confusion_matrix.png")
    
    print("\nHoan thanh Tuan 2! Base line da duoc thiet lap. Day la moc de k-NN vuot qua trong Tuan 3.")

if __name__ == "__main__":
    run_week_2()