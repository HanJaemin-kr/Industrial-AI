import numpy as np

class ChannelFeature:
    def __init__(self, channel_number, channel_name, feature):
        self.channel_number = channel_number
        self.feature_type_name =channel_name
        self.feature = feature
    def __str__(self):
        return f"Channel {self.channel_number} > {  self.feature_type_name} : {self.feature}"


def set_Signal_Info(signal_info):
    return  signal_info['n_signal'], signal_info['sampling_rate'], signal_info['max_val'], signal_info['min_val']

def extract_high_overall(signal_info, magnitudes):
    nSignal, samplingRate, maxValue, minValue = set_Signal_Info(signal_info)
    feature = 0

    # 초 단위 신호 길이 계산
    signal_duration = nSignal / samplingRate

    # 시작 주파수 계산 (1kHz)
    start_freq = 1000 * signal_duration

    # 시작 주파수가 10kHz 이하이면 끝 주파수를 10kHz로, 그렇지 않으면 samplingRate / 2.56로 설정
    end_freq = 10000 * signal_duration if 10000 < samplingRate / 2.56 else samplingRate / 2.56

    # 주파수 범위에 대한 특징 값 계산
    for i in range(int(np.floor(start_freq)), int(np.ceil(end_freq))):
        feature += magnitudes[i] ** 2

    # Overall 값 계산
    overall_feature = np.sqrt(feature)
    return overall_feature

def extract_RMS(signal_info, magnitudes, start_freq, end_freq):
    nSignal, samplingRate, maxValue, minValue = set_Signal_Info(signal_info)

    # 초 단위 신호 길이 계산
    signal_duration = nSignal / samplingRate

    # 시작 주파수 계산 (1kHz)
    start_freq = start_freq * signal_duration

    # 시작 주파수가 10kHz 이하이면 끝 주파수를 10kHz로, 그렇지 않으면 samplingRate / 2.56로 설정
    end_freq = end_freq * signal_duration if 10000 < samplingRate / 2.56 else samplingRate / 2.56

    channel_rms = []

    # 각 채널에 대한 주파수 범위에 대한 특징 값 계산
    for channel_number, channel_magnitudes in enumerate(magnitudes, start=1):
        rms = 0
        for i in range(int(np.floor(start_freq)), int(np.ceil(end_freq))):
            m = channel_magnitudes[i]
            rms += m ** 2

        # RMS 값 계산
        rms = np.sqrt(rms / (end_freq - start_freq))

        # RMS 값 및 ChannelFeature 객체 생성 및 리스트에 추가
        channel_feature = ChannelFeature(channel_number, f"High_RMS_{start_freq/1000}k_{end_freq/1000}k", rms)
        channel_rms.append(channel_feature)

    return channel_rms
def extract_Overall(signal_info, magnitudes, start_freq, end_freq):
    nSignal, samplingRate, maxValue, minValue = set_Signal_Info(signal_info)

    # 초 단위 신호 길이 계산
    signal_duration = nSignal / samplingRate

    # 시작 주파수 계산 (1kHz)
    start_freq = start_freq * signal_duration

    # 시작 주파수가 10kHz 이하이면 끝 주파수를 10kHz로, 그렇지 않으면 samplingRate / 2.56로 설정
    end_freq = end_freq * signal_duration if 10000 < samplingRate / 2.56 else samplingRate / 2.56

    channel_features = []

    # 각 채널에 대한 주파수 범위에 대한 특징 값 계산
    for channel_number, channel_magnitudes in enumerate(magnitudes, start=1):
        feature = 0
        for i in range(int(np.floor(start_freq)), int(np.ceil(end_freq))):
            m = channel_magnitudes[i]
            feature += m ** 2

        # Overall 값 계산 및 ChannelFeature 객체 생성 및 리스트에 추가
        channel_feature = ChannelFeature(channel_number, f"Overall_{start_freq/1000}k_{end_freq/1000}k", np.sqrt(feature))
        channel_features.append(channel_feature)

    return channel_features


from scipy.stats import kurtosis
def extract_Kurtosis(signal_info, magnitudes, start_freq, end_freq):
    nSignal, samplingRate, maxValue, minValue = set_Signal_Info(signal_info)

    # 초 단위 신호 길이 계산
    signal_duration = nSignal / samplingRate

    # 시작 주파수 계산
    start_freq = start_freq * signal_duration

    # 시작 주파수가 10kHz 이하이면 끝 주파수를 10kHz로, 그렇지 않으면 samplingRate / 2.56로 설정
    end_freq = end_freq * signal_duration if 10000 < samplingRate / 2.56 else samplingRate / 2.56

    channel_kurtosis = []

    # 각 채널에 대한 주파수 범위에 대한 특징 값 계산
    for channel_number, channel_magnitudes in enumerate(magnitudes, start=1):
        # 주파수 범위 내의 신호 추출
        signal = channel_magnitudes[int(np.floor(start_freq)):int(np.ceil(end_freq))]

        # Kurtosis 계산
        kurt = kurtosis(signal)

        # Kurtosis 값 및 ChannelFeature 객체 생성 및 리스트에 추가
        channel_feature = ChannelFeature(channel_number, f"Kurtosis_{start_freq/1000}k_{end_freq/1000}k", kurt)
        channel_kurtosis.append(channel_feature)

    return channel_kurtosis
