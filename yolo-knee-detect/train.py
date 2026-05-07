from ultralytics import YOLO

model = YOLO('yolo11n.pt') 

model.train(
    data='./data.yaml',
    epochs=100,
    imgsz=640,
    device=0,       # 0 genellikle laptopundaki RTX 3060'ı temsil eder
    batch=16,       # 3060'ın 6GB VRAM'i olduğunu varsayarsak 16 idealdir
    workers=8       # Veri yükleme hızı için
)