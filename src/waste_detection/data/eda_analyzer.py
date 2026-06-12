import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import cv2
from pycocotools.coco import COCO

class CocoEDAAnalyzer:
    """
    Công cụ Phân tích Dữ liệu Khám phá (EDA) Hướng đối tượng cho bộ dữ liệu chuẩn COCO.
    Hỗ trợ xuất các biểu đồ phân tích chuyên sâu cho bài toán Object Detection.
    """
    
    def __init__(self, annotation_file, image_dir, dataset_name="Dataset", label_mapper=None):
        """
        Khởi tạo Analyzer.
        
        Args:
            annotation_file (str): Đường dẫn tới file annotations.json
            image_dir (str): Đường dẫn tới thư mục chứa ảnh
            dataset_name (str): Tên dataset (dùng để lưu file)
            label_mapper (callable, optional): Hàm ánh xạ nhãn. Nhận vào (cat_id, cat_name), trả về (new_cat_name).
        """
        self.annotation_file = annotation_file
        self.image_dir = image_dir
        self.dataset_name = dataset_name
        self.coco = COCO(annotation_file)
        self.label_mapper = label_mapper
        
        # Lấy thông tin
        self.img_ids = self.coco.getImgIds()
        self.ann_ids = self.coco.getAnnIds()
        self.anns = self.coco.loadAnns(self.ann_ids)
        self.cats = self.coco.loadCats(self.coco.getCatIds())
        
        self.raw_cat_id_to_name = {cat['id']: cat['name'] for cat in self.cats}
        
        # Xử lý ánh xạ nhãn nếu có
        self.mapped_anns = []
        self.mapped_cat_names = set()
        for ann in self.anns:
            cat_id = ann['category_id']
            if cat_id not in self.raw_cat_id_to_name:
                continue
            cat_name = self.raw_cat_id_to_name[cat_id]
            
            if self.label_mapper:
                mapped_name = self.label_mapper(cat_id, cat_name)
                if mapped_name is None:
                    continue # Ignore this annotation
            else:
                mapped_name = cat_name
                
            self.mapped_cat_names.add(mapped_name)
            self.mapped_anns.append({
                **ann,
                'mapped_category': mapped_name
            })
            
        self.mapped_cat_names = sorted(list(self.mapped_cat_names))
        print(f"[{self.dataset_name}] Loaded {len(self.img_ids)} images and {len(self.mapped_anns)} annotations across {len(self.mapped_cat_names)} classes.")

    def set_plot_style(self):
        """Thiết lập phong cách đồ họa (Style) đẹp mắt."""
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'figure.autolayout': True
        })

    def plot_class_distribution(self, save_dir=None):
        """1. Biểu đồ phân bố số lượng vật thể theo từng nhãn (Bar Chart)."""
        self.set_plot_style()
        
        counts = Counter([ann['mapped_category'] for ann in self.mapped_anns])
        df = pd.DataFrame.from_dict(counts, orient='index', columns=['Count']).reset_index()
        df = df.rename(columns={'index': 'Category'}).sort_values('Count', ascending=False)
        
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='Count', y='Category', data=df, palette='viridis')
        plt.title(f'Class Distribution - {self.dataset_name}')
        plt.xlabel('Number of Objects')
        plt.ylabel('Category')
        
        # Hiện số lượng trên cột
        for i, v in enumerate(df['Count']):
            ax.text(v + 3, i, str(v), color='black', va='center')
            
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"{self.dataset_name}_class_distribution.png"), dpi=300)
        plt.show()
        
    def plot_bbox_size_distribution(self, save_dir=None):
        """2. Biểu đồ phân bố kích thước Bounding Box (Scatter Plot)."""
        self.set_plot_style()
        
        data = []
        for ann in self.mapped_anns:
            x, y, w, h = ann['bbox']
            data.append({
                'Category': ann['mapped_category'],
                'Width': w,
                'Height': h,
                'Area': w * h
            })
            
        df = pd.DataFrame(data)
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='Width', y='Height', hue='Category', data=df, alpha=0.6, palette='tab10')
        plt.title(f'Bounding Box Size Distribution - {self.dataset_name}')
        plt.xlabel('Width (pixels)')
        plt.ylabel('Height (pixels)')
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"{self.dataset_name}_bbox_size_scatter.png"), dpi=300)
        plt.show()

    def plot_objects_per_image(self, save_dir=None):
        """3. Biểu đồ Histogram đếm số lượng vật thể trên mỗi bức ảnh."""
        self.set_plot_style()
        
        counts = Counter([ann['image_id'] for ann in self.mapped_anns])
        obj_counts = list(counts.values())
        
        # Thêm các ảnh không có object nào
        zero_obj_imgs = len(self.img_ids) - len(counts)
        obj_counts.extend([0] * zero_obj_imgs)
        
        plt.figure(figsize=(8, 5))
        sns.histplot(obj_counts, bins=range(0, max(obj_counts) + 2), kde=False, color='coral')
        plt.title(f'Objects per Image - {self.dataset_name}')
        plt.xlabel('Number of Objects')
        plt.ylabel('Frequency (Images)')
        plt.xticks(range(0, max(obj_counts) + 2))
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"{self.dataset_name}_objects_per_image.png"), dpi=300)
        plt.show()
        
    def plot_bbox_heatmap(self, save_dir=None):
        """4. Bản đồ nhiệt (Heatmap) thể hiện vị trí không gian của rác trên ảnh."""
        self.set_plot_style()
        
        # Chuẩn hóa tọa độ tâm (center_x, center_y) về tỉ lệ 0-1
        centers_x = []
        centers_y = []
        for ann in self.mapped_anns:
            img_info = self.coco.loadImgs(ann['image_id'])[0]
            img_w, img_h = img_info['width'], img_info['height']
            x, y, w, h = ann['bbox']
            
            center_x = (x + w/2) / img_w
            center_y = (y + h/2) / img_h
            centers_x.append(center_x)
            centers_y.append(center_y)
            
        plt.figure(figsize=(8, 8))
        sns.kdeplot(x=centers_x, y=centers_y, cmap="Reds", fill=True, bw_adjust=0.5)
        plt.xlim(0, 1)
        plt.ylim(1, 0) # Lật trục Y giống ảnh
        plt.title(f'Spatial Distribution (Heatmap) - {self.dataset_name}')
        plt.xlabel('Normalized X')
        plt.ylabel('Normalized Y')
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"{self.dataset_name}_spatial_heatmap.png"), dpi=300)
        plt.show()

    def plot_aspect_ratio_distribution(self, save_dir=None):
        """5. Biểu đồ phân bố Tỉ lệ khung hình (Aspect Ratio = Width / Height). Rất quan trọng cho Anchor Box."""
        self.set_plot_style()
        
        aspect_ratios = []
        for ann in self.mapped_anns:
            w, h = ann['bbox'][2], ann['bbox'][3]
            if h > 0:
                aspect_ratios.append(w / h)
                
        plt.figure(figsize=(8, 5))
        sns.histplot(aspect_ratios, bins=50, kde=True, color='purple')
        plt.title(f'Aspect Ratio (W/H) Distribution - {self.dataset_name}')
        plt.xlabel('Aspect Ratio')
        plt.ylabel('Frequency')
        plt.xlim(0, 5) # Giới hạn lại để dễ nhìn
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"{self.dataset_name}_aspect_ratio.png"), dpi=300)
        plt.show()

    def plot_class_cooccurrence(self, save_dir=None):
        """6. Ma trận đồng xuất hiện (Co-occurrence Matrix): Xem rác nào thường đi đôi với rác nào."""
        self.set_plot_style()
        
        # Nhóm annotations theo image_id
        img_to_cats = {}
        for ann in self.mapped_anns:
            img_id = ann['image_id']
            cat = ann['mapped_category']
            if img_id not in img_to_cats:
                img_to_cats[img_id] = set()
            img_to_cats[img_id].add(cat)
            
        # Xây dựng ma trận
        n_classes = len(self.mapped_cat_names)
        matrix = np.zeros((n_classes, n_classes), dtype=int)
        
        cat_to_idx = {cat: i for i, cat in enumerate(self.mapped_cat_names)}
        
        for img_id, cats in img_to_cats.items():
            cats = list(cats)
            for i in range(len(cats)):
                for j in range(i, len(cats)):
                    idx1 = cat_to_idx[cats[i]]
                    idx2 = cat_to_idx[cats[j]]
                    matrix[idx1][idx2] += 1
                    if idx1 != idx2:
                        matrix[idx2][idx1] += 1
                        
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.mapped_cat_names, 
                    yticklabels=self.mapped_cat_names)
        plt.title(f'Class Co-occurrence Matrix - {self.dataset_name}')
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"{self.dataset_name}_cooccurrence.png"), dpi=300)
        plt.show()

    def visualize_sample_images(self, num_samples=3, save_dir=None):
        """7. Vẽ mẫu một vài bức ảnh cùng với Bounding Box."""
        sample_ids = random.sample(self.img_ids, min(num_samples, len(self.img_ids)))
        
        for img_id in sample_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            img_path = os.path.join(self.image_dir, img_info['file_name'])
            
            if not os.path.exists(img_path):
                print(f"File not found: {img_path}")
                continue
                
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Lấy các anns của hình này
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            
            for ann in anns:
                cat_id = ann['category_id']
                if cat_id not in self.raw_cat_id_to_name:
                    continue
                cat_name = self.raw_cat_id_to_name[cat_id]
                
                if self.label_mapper:
                    mapped_name = self.label_mapper(cat_id, cat_name)
                    if mapped_name is None:
                        continue
                else:
                    mapped_name = cat_name
                    
                x, y, w, h = [int(v) for v in ann['bbox']]
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, mapped_name, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                
            plt.figure(figsize=(10, 10))
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"Sample: {img_info['file_name']}")
            
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(os.path.join(save_dir, f"{self.dataset_name}_sample_{img_id}.png"), bbox_inches='tight', dpi=300)
            plt.show()

    def run_full_eda(self, save_dir=None):
        """Chạy toàn bộ đường ống EDA."""
        print(f"=== Running Full EDA for {self.dataset_name} ===")
        self.plot_class_distribution(save_dir)
        self.plot_bbox_size_distribution(save_dir)
        self.plot_objects_per_image(save_dir)
        self.plot_bbox_heatmap(save_dir)
        self.plot_aspect_ratio_distribution(save_dir)
        self.plot_class_cooccurrence(save_dir)
        self.visualize_sample_images(num_samples=3, save_dir=save_dir)
        print("=== Full EDA Completed ===")
