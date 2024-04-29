import numpy as np
import scipy.io as sio
import scipy.signal as sig
import torch
import os
from CNN.bearing_fault_type.toolbox.toolbox import getSpec

os.environ['KMP_DUPLICATE_LIB_OK']='True'
three_channel_info = {
    0: "MTR",
    1: "DE",
    2: "NDE",
}

two_channel_info = {
    0: "DE",
    1: "NDE",
}

fault_cause = ['normal', 'inner', 'outer', 'roller']

speeds = {
    600: 10,
    900: 15,
    1200: 20,
    1800: 30,
    2400: 40,
    3000: 50
}

def make_dataset(bearing_normal_dataset, bearing_fault_dataset, fault_case, save_dir, repetitions):

    setting_make_file_count = len(bearing_normal_dataset)
    setHeight = 3

    for make_file_count in range(setHeight, setting_make_file_count):

        fault_dataset = None
        normal_dataset = None
        index = make_file_count

        for h in np.arange(setHeight):
            fault_DE = bearing_fault_dataset[index]['DE']
            fault_NDE = bearing_fault_dataset[index]['NDE']
            normal_DE = bearing_normal_dataset[index]['DE']
            normal_NDE = bearing_normal_dataset[index]['NDE']
            index = index - 1

            # 모터랑 드라이브엔드 세로로 쌓음
            fault_data = np.vstack((fault_DE, fault_NDE))
            normal_data = np.vstack((normal_DE, normal_NDE))


            if fault_dataset is None:
                normal_dataset = normal_data
                fault_dataset = fault_data
            else:
                normal_dataset = np.vstack((normal_dataset, normal_data))
                fault_dataset = np.vstack((fault_dataset, fault_data))

        fault_tensor = torch.tensor(fault_dataset, dtype=torch.float32)
        normal_tensor = torch.tensor(normal_dataset, dtype=torch.float32)

        train_data = torch.stack([normal_tensor, fault_tensor], dim=0)
        print(train_data)
        file_name = f"{fault_case}_{make_file_count}.pth"
        file_path = os.path.join(save_dir, file_name)

        # 디렉터리 생성
        os.makedirs(save_dir, exist_ok=True)

        # 데이터 저장
        torch.save(train_data, file_path)

    return f"{save_dir} Sucess  : {train_data.shape}"


def load_type_dataset(target_dir, saved_dir, fault_type, repetitions, isRpmList):

    print(f"============  {fault_type} 추출 시작  ===============")
    if isRpmList:
        rpm_list = ['600', '900', '1200', '1500', '1800']
    else:
        rpm_list = ['1800']

    cutting_frequency = 1500

    for rpm in rpm_list:
        # ================= ref 값 추출 ( 노멀 파일 평균화 )====================================
        # 노멀 파일 평균화하기 !
        normal_path = os.path.join(target_dir, rpm, 'normal', 'Vibration')
        normal_files = [f for f in os.listdir(normal_path) if f.endswith('.mat')]
        normal_dataset = {}

        normal_file_path = os.path.join(normal_path, normal_files[0])
        normal_data = sio.loadmat(normal_file_path)
        try:
            normal_signal = normal_data['signal']
        except KeyError:
            normal_signal = normal_data['signals']

        if(normal_signal.shape[0] == 2):
            channel_info = two_channel_info
            channel_count = 2
        else:
            channel_info = three_channel_info
            channel_count = 3

        stacked_results = {ch: np.zeros(cutting_frequency) for ch in channel_info}

        for file_index, file_name in enumerate(normal_files):
            normal_file_path = os.path.join(normal_path, normal_files[file_index])
            normal_data = sio.loadmat(normal_file_path)

            try:
                normal_signal = normal_data['signal']
            except KeyError:
                normal_signal = normal_data['signals']

            for ch_index, ch_name in enumerate(channel_info):
                ch_signal = normal_signal[ch_index, :]
                ch_envelope = np.abs(sig.hilbert(ch_signal))
                ch_tval, _ = getSpec(ch_envelope, 25600, 1, 0, 1)
                ch_fft_result = ch_tval[:cutting_frequency]
                ch_fft_result[:5] = 0
                stacked_results[ch_name] += ch_fft_result

        # 파일 개수로 나누어 각 채널의 평균값 계산
        num_files = len(normal_files)
        for ch in channel_info:
            stacked_results[ch] /= num_files

        ref_nomal_value = stacked_results

        # ================= ========= ====================================

        target_path = os.path.join(target_dir, rpm, fault_type, 'Vibration')
        target_files = [f for f in os.listdir(target_path) if f.endswith('.mat')]
        target_dataset = {}

        # 결함 데이터에 대해서 데이터 추출 시작
        for file_index, file_name in enumerate(target_files):
            try:
                target_file_path = os.path.join(target_path, target_files[file_index])
                normal_file_path = os.path.join(normal_path, normal_files[file_index])
            except (FileNotFoundError, IndexError) as e:
                print(f"Error encountered while processing files: {e}")
                break


            target_data = sio.loadmat(target_file_path)
            normal_data = sio.loadmat(normal_file_path)

            try:
                target_signal = target_data['signal']
                normal_sigal = normal_data['signal']
            except KeyError:
                target_signal = target_data['signals']
                normal_sigal = normal_data['signals']


            result_target_spectrum = {}
            result_normal_spectrum = {}

            # 체널 별 추출
            for ch in range(0, channel_count):
                ch_normal_signal = normal_sigal[ch, :]
                normal_envelop_value = np.abs(sig.hilbert(ch_normal_signal))
                normal_tval, normal_tfreq = getSpec(normal_envelop_value, 25600, 1, 0, 1)
                normal_fft_result = normal_tval[0:cutting_frequency]
                normal_fft_result[:5] = 0
                normal_magnitude_spectrum = np.abs(normal_fft_result - ref_nomal_value[ch])

                result_normal_spectrum[channel_info[ch]] = normal_magnitude_spectrum


                ch_target_signal = target_signal[ch, :]
                tareget_envelop_value = np.abs(sig.hilbert(ch_target_signal))
                targer_tval, target_tfreq = getSpec(tareget_envelop_value, 25600, 1, 0, 1)
                target_fft_result = targer_tval[0:cutting_frequency]
                target_fft_result[:5] = 0
                target_magnitude_spectrum = np.abs(target_fft_result - ref_nomal_value[ch])
                result_target_spectrum[channel_info[ch]] = target_magnitude_spectrum

                # 정규화 수행
                min_value = np.min(target_magnitude_spectrum)
                max_value = np.max(target_magnitude_spectrum)
                target_magnitude_spectrum_normalized = (target_magnitude_spectrum - min_value) / (max_value - min_value)

                result_target_spectrum[channel_info[ch]] = target_magnitude_spectrum_normalized

            normal_dataset[file_index] = result_normal_spectrum
            target_dataset[file_index] = result_target_spectrum

        cur_save_dir = os.path.join(saved_dir, rpm, fault_type)
        res = make_dataset(normal_dataset, target_dataset, fault_type, cur_save_dir, repetitions)
        print(f"============  {fault_type} 추출 완료  ===============")
        print(res)
def main():
    prefix_dir = 'C:\\Users\\dudrn\\Desktop\\research\\CNN_Dataset\\coupling\\'
    saved_dir_prfix = "./data/"

    dir240201 = True

    if(dir240201):
        dir_list = ['240401']
    else:
        dir_list = [ '240310', '240311', '240131']
                 #    '240118', '231025', '231026']

    for target_dir in dir_list:
        data_dir = os.path.join(prefix_dir, target_dir)
        saved_dir = os.path.join(saved_dir_prfix, target_dir)
        print('현재 저장 파일 디렉터리 : ', saved_dir)

        types = [ 'inner', 'outer', 'roller', 'normal']
        for type in types:
            load_type_dataset(data_dir, saved_dir, type, 3000, dir240201)


    print("학습 데이터 추출 완료")

if __name__ == "__main__":
    main()
