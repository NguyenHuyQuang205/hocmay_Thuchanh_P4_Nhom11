
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os

app = Flask(__name__)

# 1. NAP MO HINH DA HUAN LUYEN (OFFLINE SERVING)
MODEL_PATH = 'models/knn_best_pipeline.pkl'
DATA_PATH = 'data/penguins.csv'

# Kiem tra xem file mo hinh co ton tai chua
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Khong tim thay mo hinh tai {MODEL_PATH}. Vui long chay Tuan 3 truoc!")

model = joblib.load(MODEL_PATH)

# 2. NAP LAI TAP TRAIN DE TRUY XUAT THONG TIN LANG GIENG
# (Phai dung dung seed 42 de dam bao khop chi so index voi cay KNN ben trong mo hinh)
df = pd.read_csv(DATA_PATH).dropna(subset=['bill_length_mm']).reset_index(drop=True)
X = df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
y = df['species']
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

@app.route('/')
def index():
    """Tra ve giao dien trang chu (Frontend)"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """API nhan du lieu JSON, kiem tra va tra ve ket qua du doan"""
    try:
        data = request.get_json()
        
        # 3. KIEM TRA GIA TRI DAU VAO (VALIDATION)
        required_keys = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
        input_data = {}
        
        for key in required_keys:
            if key not in data or data[key] == '':
                return jsonify({'error': f'Thieu thong tin: {key}'}), 400
            
            try:
                val = float(data[key])
                input_data[key] = [val]
            except ValueError:
                return jsonify({'error': f'Sai dinh dang so: {key}'}), 400

        # Kiem tra mien gia tri (Domain bounds based on Palmer Penguins stats)
        if not (30 <= input_data['bill_length_mm'][0] <= 65):
            return jsonify({'error': 'Chieu dai mo chim vuot mien gia tri thuc te (30 - 65 mm)'}), 400
        if not (10 <= input_data['bill_depth_mm'][0] <= 25):
            return jsonify({'error': 'Do day mo chim vuot mien gia tri thuc te (10 - 25 mm)'}), 400
        if not (160 <= input_data['flipper_length_mm'][0] <= 240):
            return jsonify({'error': 'Chieu dai canh vuot mien gia tri thuc te (160 - 240 mm)'}), 400
        if not (2500 <= input_data['body_mass_g'][0] <= 6500):
            return jsonify({'error': 'Can nang vuot mien gia tri thuc te (2500 - 6500 g)'}), 400

        # 4. THUC HIEN DU DOAN
        input_df = pd.DataFrame(input_data)
        
        # Du doan loai & xac suat
        pred = model.predict(input_df)[0]
        probas = model.predict_proba(input_df)[0]
        confidence = max(probas) * 100
        
        # Lay thong tin xac suat cua tung loai de tra ve
        classes = model.classes_
        class_probabilities = {cls: round(prob * 100, 2) for cls, prob in zip(classes, probas)}

        # 5. TIM 5 LANG GIENG GAN NHAT
        input_transformed = model[:-1].transform(input_df)
        knn_step = model.named_steps['knn']
        distances, indices = knn_step.kneighbors(input_transformed, n_neighbors=5)
        
        neighbors_list = []
        for i, train_idx in enumerate(indices[0]):
            real_idx = X_train.index[train_idx]
            neighbor_data = {
                'species': str(y_train.loc[real_idx]),
                'distance': round(float(distances[0][i]), 4),
                'bill_length_mm': float(X_train.loc[real_idx, 'bill_length_mm']),
                'bill_depth_mm': float(X_train.loc[real_idx, 'bill_depth_mm']),
                'flipper_length_mm': float(X_train.loc[real_idx, 'flipper_length_mm']),
                'body_mass_g': float(X_train.loc[real_idx, 'body_mass_g'])
            }
            neighbors_list.append(neighbor_data)

        # 6. TRA VE KET QUA JSON (THEO CHUAN API)
        response = {
            'prediction': pred,
            'confidence': round(confidence, 2),
            'probabilities': class_probabilities,
            'neighbors': neighbors_list
        }
        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': f'Loi may chu: {str(e)}'}), 500

if __name__ == '__main__':
    # Chay server Flask o cong 5000
    app.run(debug=True, host='0.0.0.0', port=5000)