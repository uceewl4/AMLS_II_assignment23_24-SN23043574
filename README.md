# AMLS_II_assignment23_24-SN23043574
train: yolo task=obb mode=train model=yolov8s-obb.pt data=./datasets/yolocustom-6/data.yaml epochs=200 imgsz=500 batch=6
validation: yolo task=obb mode=val model=./runs/obb/train/weights/best.pt data=./datasets/yolocustom-6/data.yaml
test: yolo task=obb mode=predict model=./runs/obb/train/weights/best.pt conf=0.2 source=./datasets/yolocustom-6/test/images save=true

