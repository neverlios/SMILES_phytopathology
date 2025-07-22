from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO('yolo11s.pt')

    model.train(
        data='TomatoDB_yolov11/data.yaml',  # файл разметки
        epochs=100,
        imgsz=640,
        batch=8,                  # размер батчей надо понять на чем тренируется
        workers=4,                 # число потоковнадо понять на чем тренируетсях
        project='Results_of_training_v11s',
        name='tomato_v11s',
        exist_ok=True    ,          # перезапись, если папка уже существует
        device=0                    # использовать GPU
    )