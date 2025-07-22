from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO('yolov8n.pt')

    model.train(
        data='TomatoDB_yolov8/data.yaml',  # файл разметки
        epochs=100,
        imgsz=640,
        batch=8,                  # размер батчей надо понять на чем тренируется
        workers=4,                 # число потоковнадо понять на чем тренируетсях
        project='Results_of_training_v8n',
        name='tomato_v8n',
        exist_ok=True    ,          # перезапись, если папка уже существует
        device=0                    # использовать GPU
    )