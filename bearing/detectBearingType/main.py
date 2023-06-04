import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# 각 결함별 주파수 설정 (예시)
ball_fault_frequency = 500  # 구면 베어링 결함 주파수 (Hz)
outer_race_fault_frequency = 1000  # 외륜 베어링 결함 주파수 (Hz)
inner_race_fault_frequency = 1500  # 내륜 베어링 결함 주파수 (Hz)
roller_fault_frequency = 2000  # 롤러 베어링 결함 주파수 (Hz)

# 베어링 신호 생성 함수 정의
def generate_bearing_signal(frequency, amplitude, noise_level=0):
    # 사인 함수로 결함 신호 생성
    signal = amplitude * np.sin(2 * np.pi * frequency * t)
    # 잡음 추가
    noise = noise_level * np.random.randn(len(t))
    signal_with_noise = signal + noise
    return signal_with_noise

# 데이터셋 생성
X = []
y = []
fault_types = ["Ball", "Outer Race", "Inner Race", "Roller"]

# 구면 베어링 결함 데이터 생성
ball_fault_signal = generate_bearing_signal(ball_fault_frequency, 
amplitude=1.0, noise_level=0.1)
X.extend(ball_fault_signal)
y.extend(["Ball"] * len(ball_fault_signal))

# 외륜 베어링 결함 데이터 생성
outer_race_fault_signal = 
generate_bearing_signal(outer_race_fault_frequency, amplitude=1.0, 
noise_level=0.1)
X.extend(outer_race_fault_signal)
y.extend(["Outer Race"] * len(outer_race_fault_signal))

# 내륜 베어링 결함 데이터 생성
inner_race_fault_signal = 
generate_bearing_signal(inner_race_fault_frequency, amplitude=1.0, 
noise_level=0.1)
X.extend(inner_race_fault_signal)
y.extend(["Inner Race"] * len(inner_race_fault_signal))

# 롤러 베어링 결함 데이터 생성
roller_fault_signal = generate_bearing_signal(roller_fault_frequency, 
amplitude=1.0, noise_level=0.1)
X.extend(roller_fault_signal)
y.extend(["Roller"] * len(roller_fault_signal))

# 데이터셋을 NumPy 배열로 변환
X = np.array(X)
y = np.array(y)

# 레이블 인코딩
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

# 훈련 및 테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, 
test_size=0.2, random_state=42)

# 입력 데이터 정규화
X_train = X_train / np.max(np.abs(X_train))
X_test = X_test / np.max(np.abs(X_test))

# 모델 구성
model = Sequential()
model.add(Flatten(input_shape=(len(t),)))
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# 모델 컴파일
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', 
metrics=['accuracy'])

# 모델 훈련
model.fit(X_train, y_train, epochs=10, batch_size=32, 
validation_data=(X_test, y_test))

# 모델 평가
_, accuracy = model.evaluate(X_test, y_test)
print("Accuracy: %.2f%%" % (accuracy * 100))

# 테스트 데이터 예측
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_pred_labels = label_encoder.inverse_transform(y_pred_labels)

# 분류 보고서 출력
print(classification_report(label_encoder.inverse_transform(y_test), 
y_pred_labels))

