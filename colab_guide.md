# Hướng dẫn chạy trên Google Colab với bộ dữ liệu TLU

Tài liệu này hướng dẫn cách thiết lập môi trường và chạy mô hình trên Google Colab, tùy chỉnh đường dẫn để lưu trữ kết quả trên Google Drive.

## 1. Chuẩn bị trên Google Drive

1.  Tải thư mục dữ liệu lên Google Drive (hoặc giữ nguyên nếu đã có).
    *   Ví dụ: `/content/drive/MyDrive/AGNN/data/tlu-states/images`
    *   Đảm bảo file `split.json` nằm trong `/content/drive/MyDrive/AGNN/data/split.json`.

## 2. Các bước trên Colab

### Bước 1: Kết nối Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Bước 2: Cài đặt thư viện
```bash
!pip install ftfy regex tqdm
!pip install git+https://github.com/openai/CLIP.git
```

### Bước 3: Chạy mô hình với tùy chỉnh đường dẫn

Sử dụng lệnh sau để chạy training và evaluation:

```bash
python main.py \
    --dataset tlu \
    --root_path /content/drive/MyDrive/AGNN/data \
    --shots 16 \
    --mode twostage \
    --setting base2new \
    --results_dir /content/drive/MyDrive/GPN_results \
    --save_path /content/drive/MyDrive/GPN_models/tlu_lora
```

## Các tham số quan trọng:
- `--root_path`: Đường dẫn đến thư mục chứa `split.json` và thư mục `tlu-states`.
- `--results_dir`: Nơi lưu file kết quả `.csv`. Nên đặt trong Drive để không bị mất khi hết session.
- `--save_path`: Nơi lưu trọng số mô hình (LoRA).
- `--shots`: Số lượng ảnh huấn luyện cho mỗi lớp (ví dụ: 1, 2, 4, 8, 16).
- `--setting`: 
    - `base2new`: Huấn luyện trên các lớp Base và kiểm tra trên cả Base và Novel (phù hợp với split của bạn).
    - `standard`: Huấn luyện và kiểm tra trên cùng một tập lớp.

## Lưu ý về lỗi đường dẫn:
Nếu bạn thấy lỗi `Directory not found`, hãy kiểm tra kỹ đường dẫn trong `TLU` loader. Hiện tại code mặc định tìm:
`{root_path}/tlu-states/images` và `{root_path}/split.json`.
