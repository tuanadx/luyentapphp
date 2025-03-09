import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

st.set_page_config(
    page_title="Banking Loan Approval Predictor",
    page_icon="💰",
    layout="wide"
)

# Tiêu đề ứng dụng
st.title("📊 Dự đoán phê duyệt khoản vay")
st.write("Vui lòng nhập thông tin của bạn để dự đoán xem khoản vay có được phê duyệt hay không.")

# Tạo bảng ánh xạ tên đặc trưng sang tiếng Việt
features_vietnamese = {
    # Thông tin cá nhân
    "Age": "Tuổi",
    "Gender": "Giới tính",
    "MaritalStatus": "Tình trạng hôn nhân",
    "NumberOfDependents": "Số người phụ thuộc",
    "Education": "Trình độ học vấn",
    "EmploymentStatus": "Tình trạng việc làm",
    "YearsOfEmployment": "Số năm làm việc",
    "HasMortgage": "Có thế chấp",
    "HasDependents": "Có người phụ thuộc",
    
    # Thông tin tài chính
    "Income": "Thu nhập",
    "LoanAmount": "Số tiền vay",
    "LoanTerm": "Thời hạn vay",
    "LoanPurpose": "Mục đích vay",
    "InterestRate": "Lãi suất",
    "MonthlyLoanPayment": "Số tiền trả hàng tháng",
    "DebtToIncomeRatio": "Tỷ lệ nợ/thu nhập",
    "CreditScore": "Điểm tín dụng",
    "NumberOfCreditAccounts": "Số tài khoản tín dụng",
    "NumberOfDelinquentAccounts": "Số tài khoản quá hạn",
    "CreditUtilization": "Mức sử dụng tín dụng",
    "LoanToValueRatio": "Tỷ lệ vay/giá trị",
    
    # Lịch sử tín dụng
    "HasPriorDefault": "Đã từng vỡ nợ",
    "LoanGrade": "Xếp hạng khoản vay",
    "HasBankruptcy": "Đã từng phá sản",
    "MonthsWithZeroBalanceOverLast12Months": "Số tháng số dư bằng 0 trong 12 tháng qua",
    "MonthsWithLowSpendingOverLast12Months": "Số tháng chi tiêu thấp trong 12 tháng qua",
    "MonthsWithHighSpendingOverLast12Months": "Số tháng chi tiêu cao trong 12 tháng qua",
    "AmountInvestedMonthly": "Số tiền đầu tư hàng tháng",
    "MonthlyBalance": "Số dư hàng tháng",
    
    # Thông tin bổ sung
    "ApplicationDate": "Ngày nộp đơn",
    "ApplicationTime": "Thời gian nộp đơn",
    "ApplicationType": "Loại đơn",
    "HousingStatus": "Tình trạng nhà ở",
    "ZipCode": "Mã bưu chính"
}

# Khởi tạo các biến trước
model = None
scaler = None
label_encoders = {}

# Đọc dữ liệu huấn luyện để lấy thông tin về các đặc trưng
try:
    df_train = pd.read_csv("cityu10c_train_dataset.csv")
    # Lấy tất cả các đặc trưng (trừ ID, ApplicationDate, LoanApproved)
    features = df_train.drop(['ID', 'ApplicationDate', 'LoanApproved'], axis=1, errors='ignore').columns.tolist()
    # Phân loại đặc trưng số và phân loại
    numeric_features = df_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_features = [feat for feat in numeric_features if feat not in ['ID', 'LoanApproved']]
    categorical_features = df_train.select_dtypes(include=['object']).columns.tolist()
    categorical_features = [feat for feat in categorical_features if feat not in ['ApplicationDate']]
    
    st.sidebar.info(f"Đã tìm thấy {len(features)} đặc trưng từ dữ liệu huấn luyện")
except Exception as e:
    st.sidebar.error(f"Không thể đọc dữ liệu huấn luyện: {str(e)}")
    features = []
    numeric_features = []
    categorical_features = []

# Kiểm tra các file cần thiết
required_files = ["best_rf_model.pkl", "scaler.pkl", "encoders.pkl"]
missing_files = [f for f in required_files if not os.path.exists(f)]

if missing_files:
    st.error(f"Thiếu các file: {', '.join(missing_files)}. Vui lòng huấn luyện mô hình trước.")
    st.stop()

# Load model và các thành phần liên quan
try:
    with open("best_rf_model.pkl", "rb") as f:
        model = pickle.load(f)
    
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    with open("encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)
    
except Exception as e:
    st.error(f"Lỗi khi tải mô hình: {str(e)}")
    st.stop()

# Tạo form nhập liệu
st.subheader("Nhập thông tin khoản vay")

# Tạo container cho input
input_data = {}

# Hiển thị các đặc trưng theo nội dung thực tế
if len(features) > 0:
    # Chia thành 2 cột để giao diện đẹp hơn
    col1, col2 = st.columns(2)
    
    # Hiển thị các đặc trưng số
    for i, feature in enumerate(numeric_features):
        col = col1 if i % 2 == 0 else col2
        with col:
            # Lấy giá trị min, max và median từ dữ liệu huấn luyện
            min_val = float(df_train[feature].min())
            max_val = float(df_train[feature].max())
            default_val = float(df_train[feature].median())
            
            # Sử dụng tên tiếng Việt nếu có
            feature_label = features_vietnamese.get(feature, feature)
            
            # Tạo input số
            input_data[feature] = st.number_input(
                f"{feature_label}", 
                min_value=min_val, 
                max_value=max_val, 
                value=default_val,
                step=(max_val - min_val) / 100,
                help=f"Đặc trưng gốc: {feature}"
            )
    
    # Hiển thị các đặc trưng phân loại
    for i, feature in enumerate(categorical_features):
        col = col1 if i % 2 == 0 else col2
        with col:
            # Lấy các giá trị duy nhất
            unique_values = df_train[feature].dropna().unique().tolist()
            
            # Sử dụng tên tiếng Việt nếu có
            feature_label = features_vietnamese.get(feature, feature)
            
            # Tạo selectbox
            input_data[feature] = st.selectbox(
                f"{feature_label}", 
                unique_values,
                help=f"Đặc trưng gốc: {feature}"
            )
else:
    st.warning("Không tìm thấy thông tin về các đặc trưng từ dữ liệu huấn luyện!")
    # Hiển thị form nhập liệu đơn giản nếu không có thông tin về đặc trưng
    col1, col2 = st.columns(2)
    with col1:
        input_data["Amount"] = st.number_input("Số tiền vay", value=10000.0, step=1000.0)
        input_data["Age"] = st.number_input("Tuổi", value=30, step=1)
    with col2:
        input_data["Income"] = st.number_input("Thu nhập", value=5000.0, step=500.0)
        input_data["Credit_Score"] = st.number_input("Điểm tín dụng", value=700, step=10)

# Nút dự đoán
if st.button("🔍 Dự đoán", use_container_width=True) and model is not None and scaler is not None:
    try:
        # Chuẩn bị dữ liệu đầu vào bằng cách tạo DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Xử lý các đặc trưng phân loại nếu cần
        for col in input_df.columns:
            if col in categorical_features:
                # Kiểm tra nếu có encoder cho cột này
                if col in label_encoders:
                    # Kiểm tra giá trị đầu vào có trong tập huấn luyện không
                    input_value = input_df[col].iloc[0]
                    encoder = label_encoders[col]
                    
                    # Kiểm tra xem giá trị có trong classes_ của encoder không
                    if input_value not in encoder.classes_:
                        st.warning(f"Giá trị '{input_value}' cho đặc trưng '{col}' không có trong dữ liệu huấn luyện. Sử dụng giá trị phổ biến nhất.")
                        # Sử dụng giá trị phổ biến nhất
                        most_common = encoder.classes_[0]
                        input_df[col] = most_common
                    
                    # Chuyển đổi giá trị
                    input_df[col] = encoder.transform([input_df[col].iloc[0]])[0]
        
        # Đảm bảo thứ tự các đặc trưng đúng
        if len(features) > 0:
            # Kiểm tra xem có đầy đủ các đặc trưng cần thiết không
            missing_cols = [col for col in features if col not in input_df.columns]
            if missing_cols:
                st.error(f"Thiếu các đặc trưng: {', '.join(missing_cols)}")
                st.stop()
            
            # Sắp xếp lại các cột theo thứ tự đúng
            input_df = input_df[features]
        
        # Chuẩn hóa dữ liệu
        input_scaled = scaler.transform(input_df)
        
        # Dự đoán
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        # Hiển thị kết quả
        st.subheader("Kết quả dự đoán")
        
        if prediction == 1:
            st.success("✅ Khoản vay có khả năng được phê duyệt!")
        else:
            st.error("❌ Khoản vay có khả năng bị từ chối!")
        
        # Hiển thị xác suất
        st.subheader("Xác suất")
        col1, col2 = st.columns(2)
        col1.metric("Xác suất từ chối", f"{prediction_proba[0]:.2%}")
        col2.metric("Xác suất phê duyệt", f"{prediction_proba[1]:.2%}")
        
    except Exception as e:
        st.error(f"Lỗi khi dự đoán: {str(e)}")
        st.exception(e)
