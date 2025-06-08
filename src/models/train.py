# src/models/train.py

import os
import sys

# (필요시) 프로젝트 루트를 PYTHONPATH에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.load_data import load_sign_mnist
from src.models.cnn import build_sign_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
# 1) 저장 폴더 만들기
os.makedirs("models", exist_ok=True)

if __name__ == "__main__":
    # 2) 데이터 로드
    X_train, X_val, X_test, y_train, y_val, y_test = load_sign_mnist(
        "data/raw/sign_mnist_train.csv",
        "data/raw/sign_mnist_test.csv"
    )

    # 3) 모델 생성 (y_train의 one-hot 차원에서 클래스 개수를 가져옵니다)
    num_classes = y_train.shape[1]
    model = build_sign_model(input_shape=(28,28,1), num_classes=num_classes)

    # 4) 콜백 설정
    checkpoint = ModelCheckpoint(
        "models/asl_cnn.h5",
        save_best_only=True,
        monitor="val_accuracy",
        verbose=1
    )
    early_stop = EarlyStopping(
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    callbacks = [checkpoint, early_stop]

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 5) 학습
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=128,
        callbacks=callbacks
    )
    print("✅ Training complete. Model saved to models/asl_cnn.h5")
