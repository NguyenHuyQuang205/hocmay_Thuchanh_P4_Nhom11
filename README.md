# Project 04: Nhận dạng loài chim cánh cụt bằng k-NN

## 1. Giới thiệu
Trang giáo dục sinh thái gợi ý loài chim cánh cụt dựa trên các số đo hình thái, sử dụng thuật toán k-NN.

## 2. Cài đặt môi trường
\`\`\`bash
pip install -r requirements.txt
\`\`\`

# Hồ sơ dữ liệu: Palmer Penguins

## 1. Thông tin chung
- **Nguồn chính thức**: [https://github.com/allisonhorst/palmerpenguins](https://github.com/allisonhorst/palmerpenguins)
- **Ngày tải**: 01/10/2026
- **Tệp sử dụng**: `penguins.csv` (Dữ liệu gốc đã qua làm sạch cơ bản).
- **Giấy phép**: CC0 (Miễn trừ bản quyền) - Ghi công Palmer Station LTER và palmerpenguins.

## 2. Từ điển dữ liệu (Data Dictionary)
Dữ liệu bao gồm 344 cá thể thuộc 3 loài chim cánh cụt (Adelie, Chinstrap, Gentoo) sống trên 3 đảo thuộc quần đảo Palmer.

| Biến (Cột) | Kiểu dữ liệu | Đơn vị | Vai trò | Mô tả |
| :--- | :--- | :--- | :--- | :--- |
| `species` | Phân loại (String)| Không | **Target** | Loài chim cánh cụt (Đầu ra dự đoán). |
| `island` | Phân loại (String)| Không | Feature | Đảo nơi cá thể được quan sát (Tùy chọn). |
| `bill_length_mm` | Số thực (Float) | mm | Feature | Chiều dài mỏ chim. |
| `bill_depth_mm` | Số thực (Float) | mm | Feature | Độ dày mỏ chim. |
| `flipper_length_mm` | Số nguyên/Thực| mm | Feature | Chiều dài cánh chèo. |
| `body_mass_g` | Số nguyên/Thực| gram | Feature | Khối lượng cơ thể. |
| `sex` | Phân loại (String)| Không | Feature | Giới tính của chim (Tùy chọn). |
| `year` | Số nguyên (Int) | Năm | ID/Loại| Năm ghi nhận dữ liệu. |