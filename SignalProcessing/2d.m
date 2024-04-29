% setting
% 1x = 30 Hz 
nSignal = 25600;
samplingRate = 25600;
max_val = 50;
min_val = -50;
cutting_frequency = 3000;

% 현재 작업 디렉토리를 얻음file_index
baseDirPrefix = 'C:\Users\dudrn\Desktop\research\Data\';
dir_info = dir(baseDirPrefix);
folder_names = {dir_info([dir_info.isdir]).name};
folder_names = setdiff(folder_names, {'실험데이터(2024-01)', '.', '..'}); % '.'와 '..' 제외

for folder_index = 1:length(folder_names)
    folder_name = folder_names(folder_index);
    inputPath = fullfile(baseDirPrefix, char(folder_name), 'Vibration\');
    files = dir(fullfile(inputPath, '*.mat'));
    
    % 결과를 저장할 배열 초기화
    dataset = [];
    
    for file_index = 1:length(files)
        % 파일 읽기
        file_path = fullfile(inputPath, files(file_index).name);
        data = load(file_path);
        % 각 채널에 대한 스케일링된 스펙트럼을 저장하기 위한 임시 배열
    
        file_signals = [];
        for ch = 1:3
            [ value, frequency ] = getspec(data.signal(ch,:), data.fs, 1);
            value = value(1:cutting_frequency); % 1x1000 double
            frequency = frequency(1:cutting_frequency);
            sclae_max = max(value);
            sclae_min = min(value);
            scaled_signal = (value - sclae_min) / (sclae_max - sclae_min); % 스케일링 수행
            file_signals = [file_signals; scaled_signal];
        end
        % 임시 배열을 all_signals에 추가
         dataset = cat(3, dataset, file_signals);
    end
    
    num_channels = size(dataset, 1);
    num_frequency = size(dataset, 2);
    num_files = size(dataset, 3);
    result_array = [];
    channel1_dataset = [];
    channel2_dataset = [];
    channel3_dataset = [];

    for file_index = 1:num_files
        % Extract data for each channel for the current file
        channel1_data = dataset(1, :, file_index); % 1x1500 double
        channel2_data = dataset(2, :, file_index); % 1x1500 double
        channel3_data = dataset(3, :, file_index); % 1x1500 double
        
        channel1_dataset = cat(1, channel1_dataset, channel1_data);
        channel2_dataset = cat(1, channel2_dataset, channel2_data);
        channel3_dataset = cat(1, channel3_dataset, channel3_data);
    end
    % Resize the image (adjust the factor as needed)

        
    % Display the stacked image
    figure;
    subplot(3, 1, 1);
    imshow(channel1_dataset);
    title('Channel 1');
    
    subplot(3, 1, 2);
    imshow(channel2_dataset);
    title('Channel 2');
    
    subplot(3, 1, 3);
    imshow(channel3_dataset);
    title('Channel 3');
end
   
