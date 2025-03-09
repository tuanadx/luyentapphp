# luyentapphp
# Banking Loan Approval Predictor

Ứng dụng dự đoán phê duyệt khoản vay sử dụng mô hình RandomForest, được triển khai với Streamlit.

## Tính năng

- **Khám phá dữ liệu**: Phân tích và khám phá dữ liệu khoản vay
- **Huấn luyện mô hình**: Tùy chỉnh và huấn luyện mô hình RandomForest với nhiều tùy chọn
- **Dự đoán**: Nhập thông tin khoản vay và nhận kết quả dự đoán

## Cài đặt

1. Clone repository này
2. Cài đặt các package cần thiết:

```bash
pip install -r requirements.txt
```

3. Đảm bảo file dữ liệu `cityu10c_train_dataset.csv` nằm trong thư mục chính

## Chạy ứng dụng locally

```bash
streamlit run streamlit_app.py
```

## Triển khai lên Streamlit Cloud

1. Đưa code lên GitHub repository
2. Đăng nhập vào [Streamlit Cloud](https://streamlit.io/cloud)
3. Tạo New App và trỏ đến repository của bạn
4. Chỉ định file chính: `streamlit_app.py`
5. Đợi ứng dụng được triển khai

## Hướng dẫn sử dụng

### Tab Khám phá dữ liệu
- Xem thông tin tổng quan về dữ liệu
- Phân tích phân phối của từng đặc trưng
- Xem mối quan hệ giữa đặc trưng và kết quả phê duyệt

### Tab Huấn luyện mô hình
- Chọn đặc trưng để huấn luyện mô hình
- Tùy chỉnh tham số mô hình RandomForest
- Thực hiện tối ưu hóa siêu tham số (nếu cần)
- Đánh giá mô hình với các chỉ số như Accuracy, F1 Score
- Xem tầm quan trọng của các đặc trưng

### Tab Dự đoán
- Nhập thông tin khoản vay
- Nhận kết quả dự đoán và xác suất duyệt/từ chối

## Yêu cầu file dữ liệu

Để ứng dụng hoạt động, bạn cần file dữ liệu `cityu10c_train_dataset.csv` với các trường thông tin sau:
- ID: Mã số khoản vay
- ApplicationDate: Ngày đăng ký
- LoanApproved: Biến mục tiêu (1 = Duyệt, 0 = Từ chối)
- Các đặc trưng khác của khoản vay

## Lưu ý

- Mô hình được lưu tại file `loan_model.pkl` sau khi huấn luyện
- Dữ liệu thử nghiệm có thể được dự đoán qua tab Dự đoán 