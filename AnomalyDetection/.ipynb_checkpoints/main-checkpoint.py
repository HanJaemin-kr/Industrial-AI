import os
from extractPy import *
from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =================  setting  =====================
signal_info = {
    'n_signal': 25600,
    'sampling_rate': 25600,
    'max_val': 50,
    'min_val': -50
}

# ===================================================
'''
Amp	Amplitude ( TW )
RMS	RMS ( TW )
HighRMS	1k~10kHz Acc. RMS
EcuRMS	3k~10kHz Acc. RMS
ISoRMS	10~4kHz Vel. RMS
RMS2k	5~2kHz Vel. RMS
1X amp
2X amp
3X amp
4X amp
5X amp
SideBand 1X : bool
SideBand 2X : bool
SideBand 3X : bool
SideBand 4X : bool
SideBand 5X : bool

16 
'''
def plot_fft_results(freqs, fft_results):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.ravel()

    for i, fft_result in enumerate(fft_results):
        axs[i].plot(freqs, fft_result, label=f'Channel {i+1}')
        axs[i].set_xlabel('Frequency (Hz)')
        axs[i].set_ylabel('Magnitude')
        axs[i].set_title(f'FFT Result - Channel {i+1}')
        axs[i].legend()
        axs[i].grid(True)

    plt.tight_layout()
    plt.show()

def convert_to_dataframe(data_dict):
    data_list = []
    for key, values in data_dict.items():
        for value in values:
            # 특징의 이름을 고유하게 만듦 (예: 'Overall_1k_10k' -> 'Overall_1k_10k_Channel_1')
            # value.feature_name = f"{key}_Channel_{value.channel_number}"
            data_list.append(value.__dict__)
    df = pd.DataFrame(data_list)
    return df

def perform_fft(signal, sampling_rate):
    # 각 채널의 FFT 결과를 저장할 리스트
    fft_results = []

    # 각 채널에 대해 FFT 수행
    for channel_signal in signal:
        fft_result = np.fft.rfft(channel_signal)
        fft_results.append(np.abs(fft_result))  # 절대값

    # 주파수 벡터 생성
    freqs = np.fft.rfftfreq(len(signal[0]), 1/sampling_rate)

    return freqs, np.array(fft_results)

def main():
    baseDirPrefix = r'C:\Users\dudrn\Desktop\research\Data\\'

    #FaultList = ['beltloose_D54', 'beltloose_D96', 'beltnormal', 'coupling normal', 'Gear normal1800RPM', 'GF1800RPM', 'innerfault', 'outer fault', 'rolerfault', 'unbalance']
    FaultList = [ 'innerfault', 'outer fault', 'rolerfault', 'unbalance']

    targetPath = os.path.join(baseDirPrefix, FaultList[0], 'Vibration')

    # 디렉터리 내의 파일 목록 가져오기
    fileList = os.listdir(targetPath)

    # 파일을 하나씩 읽어오기
    for file in fileList:
        target_file = os.path.join(targetPath, file)

        mat_data = loadmat(target_file)

        # 헤더 정보 출력
        if '__header__' in mat_data:
            header_info = mat_data['__header__']


        # 데이터 읽어오기
        if 'signal' in mat_data:
            signal = mat_data['signal']
            freqs, spectrum = perform_fft(signal, signal_info['sampling_rate'])

            plot_fft_results(freqs, spectrum)
            overallData = {
                'HighRms': extract_Overall(signal_info, spectrum, 1000, 10000),
                'EcuRms': extract_Overall(signal_info, spectrum, 3000, 10000),
                'Overall_5k_20k': extract_Overall(signal_info, spectrum, 5000, 20000),
                'Overall_1k_40k': extract_Overall(signal_info, spectrum, 1000, 40000),
            }
            rmsData = {
                'rms_1k_10k': extract_RMS(signal_info, spectrum, 1000, 10000),
                'rms_3k_10k': extract_RMS(signal_info, spectrum, 3000, 10000),
                'rms_5k_20k': extract_RMS(signal_info, spectrum, 5000, 20000),
                'rms_1k_40k': extract_RMS(signal_info, spectrum, 1000, 40000),
            }
            kurtosisData = {
                'kurtosis_1k_10k': extract_Kurtosis(signal_info, spectrum, 1000, 10000),
                'kurtosis_3k_10k': extract_Kurtosis(signal_info, spectrum, 3000, 10000),
                'kurtosis_5k_20k': extract_Kurtosis(signal_info, spectrum, 5000, 20000),
                'kurtosis_1k_40k': extract_Kurtosis(signal_info, spectrum, 1000, 40000),
            }
            print('종료')
            # 각각의 데이터를 DataFrame으로 변환
            overall_df = convert_to_dataframe(overallData)
            rms_df = convert_to_dataframe(rmsData)
            kurtosis_df = convert_to_dataframe(kurtosisData)

            # 모든 DataFrame을 합침
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            result_df = pd.concat([overall_df, rms_df, kurtosis_df]).reset_index(drop=True)
            channel1_df = result_df[result_df['channel_number'] == 1]
            print(channel1_df)
            channel2_df = result_df[result_df['channel_number'] == 2]
            print(channel1_df)
            channel3_df = result_df[result_df['channel_number'] == 3]
            print(channel1_df)
            print(channel2_df)
            print(channel3_df)

        print('종료 \n\n')
        break


if __name__ == "__main__":
    main()