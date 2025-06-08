import os
import sys

# TensorFlow 로그 레벨 설정
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# 프로젝트 루트를 PYTHONPATH에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import numpy as np
from collections import deque, Counter
from tensorflow.keras.models import load_model
from src.models.cnn import build_sign_model

# 1) 모델 로드
model_path = "models/best_model.hdf5"
try:
    model = load_model(model_path)
    print(f"Loaded model from {model_path}")
except Exception:
    # 모델 생성 후 가중치 로드
    num_classes = 25  # 실제 모델 출력 노드 수에 맞춰 조정
    model = build_sign_model(num_classes=num_classes)
    model.load_weights(model_path)
    print(f"Built model and loaded weights from {model_path}")

# 2) 레이블 맵 동적 생성
num_classes = model.output_shape[-1]
label_map = {i: chr(ord('A') + i) for i in range(num_classes)}

# 3) 웹캠 실행
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam을 열 수 없습니다.")
    sys.exit(1)

# 4) 프레임 스무딩 히스토리
history = deque(maxlen=5)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 5) ROI 지정 (중앙 200x200)
    h, w = frame.shape[:2]
    x1, y1 = w//2 - 100, h//2 - 100
    crop = frame[y1:y1+200, x1:x1+200]

    # 6) 전처리: 그레이스케일 & 리사이즈
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(gray, (28, 28)).reshape(1, 28, 28, 1) / 255.0

    # 7) 예측 및 스무딩
    preds = model.predict(roi)
    current = int(np.argmax(preds))
    history.append(current)
    pred_class = Counter(history).most_common(1)[0][0]
    letter = label_map.get(pred_class, '?')

    # 8) 결과 표시
    cv2.rectangle(frame, (x1, y1), (x1+200, y1+200), (255, 0, 0), 2)
    cv2.putText(frame, f"Pred: {letter}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.imshow("ASL Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
