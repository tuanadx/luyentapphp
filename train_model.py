import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# 1. Load dữ liệu
try:
    df = pd.read_csv("cityu10c_train_dataset.csv")
    print("Đã tải dữ liệu thành công!")
except FileNotFoundError:
    print("Không tìm thấy file dữ liệu 'cityu10c_train_dataset.csv'")
    exit()

# 2. Tiền xử lý dữ liệu
X = df.drop(['ID', 'ApplicationDate', 'LoanApproved'], axis=1, errors='ignore')
y = df['LoanApproved']

# Xử lý giá trị thiếu
X.fillna(X.median(numeric_only=True), inplace=True)

# Mã hóa các biến phân loại
categorical_cols = X.select_dtypes(include=['object']).columns
label_encoders = {}  # Dictionary chứa encoder cho từng cột

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le  # Lưu encoder của từng cột

# 3. Chia tập huấn luyện/kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Huấn luyện mô hình RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 6. Đánh giá mô hình
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# 7. Lưu mô hình và các thành phần xử lý
try:
    # Lưu mô hình ở chế độ binary write 'wb'
    with open('best_rf_model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    # Lưu scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    # Lưu encoders - lưu dictionary chứa encoder cho từng cột
    with open('encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
        
    print("✅ Đã lưu mô hình vào 'best_rf_model.pkl', scaler vào 'scaler.pkl' và encoders vào 'encoders.pkl'!")
except Exception as e:
    print(f"❌ Lỗi khi lưu mô hình: {str(e)}")
