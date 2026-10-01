
import pandas as pd
import os

def download_data():
    # URL trực tiếp đến file raw trên GitHub chính thức
    url = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv"
    
    # Đường dẫn lưu file vào thư mục data của dự án
    output_path = os.path.join("data", "penguins.csv")
    
    # Đọc dữ liệu từ URL và lưu thành file CSV
    print("Đang tải dữ liệu Palmer Penguins...")
    df = pd.read_csv(url)
    df.to_csv(output_path, index=False)
    print(f"Tải thành công! Dữ liệu đã được lưu tại: {output_path}")

if __name__ == "__main__":
    download_data()