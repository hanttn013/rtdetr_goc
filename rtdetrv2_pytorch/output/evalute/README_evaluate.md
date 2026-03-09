# Hướng dẫn Kiểm thử (Evaluation) trên Tập Test

Folder `evalute` được sử dụng để tiến hành test classification metrics và đánh giá tổng thể model trên tập test dataset.

## 1. File `test_model.py`
File này load cấu hình model, thay thế val_dataloader bằng `test_dataloader` đã được định nghĩa trong `custom.yml` và tiến hành evaluation model với COCO API. Script cũng xử lý in ra độ chính xác (Average Precision - AP) trên từng class riêng biệt để nhận biết chi tiết hiệu năng model.

## 2. File `detect_test.py`
File này giúp sinh ra các hình ảnh detect trực quan trên tập test dataset, lưu toàn bộ vào thư mục `output/evalute/images`. Bằng cách này bạn có thể quan sát xem model bắt hộp (bounding box) có chuẩn xác không.

## 3. Cách chạy
Di chuyển vào thư mục cài đặt gốc (chứa `configs`, `output`...) và chạy các lệnh:

```bash
# Ở folder /workspace/rtdetr_goc/rtdetrv2_pytorch

# Lệnh 1: Tính toán métrics (AP, mAP...)
python output/evalute/test_model.py -c configs/rtdetrv2/include/rtdetrv2_r18vd_custom_vehicle.yml -r output/best.pth

# Lệnh 2: Vẽ hình ảnh detect và lưu vào output/evalute/images (có thể truyền thêm ngưỡng -t 0.5)
python output/evalute/detect_test.py -c configs/rtdetrv2/include/rtdetrv2_r18vd_custom_vehicle.yml -r output/best.pth -t 0.5
```

## 4. Khái niệm Metric (Classification & Detection)
Sau khi chạy, script sẽ tổng hợp file log tại `output/evalute/test_metrics.txt` và in ra màn hình console. Nội dung bao gồm:
- **mAP (IoU=0.50:0.95)**: COCO primary metric
- **mAP (IoU=0.50)**: VOC metric (Detection và coi là giống classification base accuracy).
- Bảng chi tiết **Per-Class AP**: Thấy rõ độ chính xác trên từng loại Class ID (0, 1, 2, 3...)
