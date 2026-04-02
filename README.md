Tải các thư viện cần thiết:
pip install - r rtdetrv2_pytorch\requirements.txt

Dataset đã được lưu trong rtdetr_goc\rtdetrv2_pytorch\dataset
Khi thầy muốn fine tuning data khác chỉ cần tải dataset dạng COCO về rtdetr_goc\rtdetrv2_pytorch\dataset
Trong Configs/dataset thầy em có tạo 1 cáu custom.yml để.... chỉ cần đổi class và numclass để ....

Trong Configs/rtdetrv2/include có rtdetrv2_r18vd_custom_vehicle.yml trong này sẽ cofigs epoch chạy, cấu hình,...
Sau đó khi train thì thầy cd vào rtdetrv2_pytorch
file train.py nằm trong  tools/train.py nên khi thầy muốn train thầy gõ lệnh sau vào terminal
python3 tools/train.py -c configs/rtdetrv2/include/rtdetrv2_r18vd_custom_vehicle.yml (đay là file yml cần tạo)
sau khi train xong model sẽ có file output trong rtdetrv2_pytorch thầy kéo nó vào thư mục model bên ngoài
trong thư mục evalua có chứa sẵn eval.py chạy metric của file log.txt
eval_test.py sẽ predict trên tập test
nhưng trước khi test trên tập test phải chạy convert.py để chuyển model sang onnx (model.onnx) nằm trong thư mục model, chạy benchmark.py để tiens hành benchmark treen CPU
Thư mục streaming nhóm có chuẩn bị sẵn 2 video khi thầy chạy streaming.py theo cấu trúc sau chỉ cần đổi tên video sẽ có vdeo được detect = model
python streaming.py --model "E:\Pj\Source Code Deep Learning\rtdetr_goc\model\stronger_reg\model.onnx" --source tên video.mp4 --save output.mp4 --imgsz 640 --conf 0.2