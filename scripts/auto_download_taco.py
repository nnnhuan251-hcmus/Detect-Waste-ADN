import os
import shutil
import kagglehub
import subprocess
import sys

def main():
    print("Bắt đầu tải dataset từ Kaggle...")
    # Tải dataset
    try:
        path = kagglehub.dataset_download("wilmacv251/taco-dataset-waste")
        print("Đã tải xong tại:", path)
    except Exception as e:
        print("Lỗi tải data:", e)
        sys.exit(1)

    target_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "TACO"))
    
    print(f"Chép dữ liệu từ {path} sang {target_dir}...")
    # Tạo thư mục nếu chưa có
    os.makedirs(target_dir, exist_ok=True)
    
    # Copy dữ liệu
    for item in os.listdir(path):
        s = os.path.join(path, item)
        d = os.path.join(target_dir, item)
        if os.path.isdir(s):
            if not os.path.exists(d):
                shutil.copytree(s, d)
        else:
            if not os.path.exists(d):
                shutil.copy2(s, d)
                
    print("Chép xong! Đang chạy prepare_taco.py...")
    
    # Chạy lệnh chuẩn bị dữ liệu
    prepare_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "prepare_taco.py"))
    
    try:
        subprocess.run([sys.executable, prepare_script], check=True, cwd=os.path.join(os.path.dirname(__file__), ".."))
        print("HOÀN TẤT TẤT CẢ!")
    except subprocess.CalledProcessError as e:
        print("Lỗi khi chạy prepare_taco.py:", e)

if __name__ == "__main__":
    main()
