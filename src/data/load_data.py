import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def load_sign_mnist(path_train: str,
                    path_test: str,
                    val_split: float = 0.1,
                    random_state: int = 42):
    """
    CSV 파일을 로드하여 학습, 검증, 테스트 세트로 분할 후 반환합니다.

    Returns:
        X_train (np.ndarray): 학습 입력 데이터, shape=(n_train, 28, 28, 1), 값 범위 [0,1]
        X_val   (np.ndarray): 검증 입력 데이터, shape=(n_val, 28, 28, 1)
        X_test  (np.ndarray): 테스트 입력 데이터, shape=(n_test, 28, 28, 1)
        y_train (np.ndarray): 학습 레이블(one-hot), shape=(n_train, num_classes)
        y_val   (np.ndarray): 검증 레이블(one-hot)
        y_test  (np.ndarray): 테스트 레이블(one-hot)
    """
    # 학습 데이터 로드
    df_train = pd.read_csv(path_train)
    X = df_train.iloc[:, 1:].values.astype(np.float32)
    y = df_train.iloc[:, 0].values

    # 이미지 형태로 변환 및 정규화
    X = X.reshape(-1, 28, 28, 1) / 255.0

    # 학습/검증 분할
    X_train, X_val, y_train_labels, y_val_labels = train_test_split(
        X, y,
        test_size=val_split,
        random_state=random_state,
        stratify=y
    )

    # 클래스 수: 레이블의 최대값 + 1
    num_classes = int(df_train.iloc[:, 0].max()) + 1

    # 레이블 원-핫 인코딩
    y_train = to_categorical(y_train_labels, num_classes=num_classes)
    y_val   = to_categorical(y_val_labels,   num_classes=num_classes)

    # 테스트 데이터 로드
    df_test = pd.read_csv(path_test)
    X_test = df_test.iloc[:, 1:].values.astype(np.float32)
    X_test = X_test.reshape(-1, 28, 28, 1) / 255.0
    y_test = to_categorical(df_test.iloc[:, 0].values,
                             num_classes=num_classes)

    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    # 로컬 테스트 실행 (현재 작업 디렉토리가 프로젝트 루트인 경우)
    X_train, X_val, X_test, y_train, y_val, y_test = load_sign_mnist(
        "data/raw/sign_mnist_train.csv",
        "data/raw/sign_mnist_test.csv"
    )
    print("Loaded shapes:", X_train.shape, X_val.shape, X_test.shape)
    print("Value range:", X_train.min(), "~", X_train.max())
