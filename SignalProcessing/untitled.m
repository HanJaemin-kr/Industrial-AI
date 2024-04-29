
% ==================
% setting
% 1x = 30 Hz 
nSignal = 25600;
samplingRate = 25600;
max = 50;
min = -50;


% 현재 작업 디렉토리를 얻음
TargetDir = 'GF1800RPM';
%
baseDirPrefix = 'C:\Users\dudrn\Desktop\research\Data\';
inputPath = fullfile(baseDirPrefix, TargetDir, 'Vibration\');
files = dir(fullfile(inputPath, '*.mat'));
clear TargetDir baseDirPrefix;


% 결과를 저장할 배열 초기화
all_signals = [];

% 각 파일에 대해 반복
for i = 1:length(files)
    % 파일 읽기
    file_path = fullfile(inputPath, files(i).name);
    data = load(file_path);
    try
        all_signals = cat(3, all_signals, data.signal);
    catch
        % 다른 경우에는 건너뛰기 또는 처리를 추가로 구현
        disp(['Skipping file: ' files(i).name]);
    end
end

[nChannels, nSamples, nTrials] = size(all_signals);
rows = ceil(sqrt(nTrials));
cols = round(sqrt(nTrials));
%%
% 각 채널에 대한 스펙트로그램을 계산하고 시각화합니다.

for i = 1:1
    figure; % 새로운 그래프 창을 엽니다.
    
    % 각 반복 측정에 대한 스펙트로그램을 계산합니다.
    for j = 1:nTrials
        % 현재 채널과 반복에서의 신호를 선택합니다.
        signal = squeeze(all_signals(i, :, j));
        
        % 스펙트로그램을 계산합니다.
        [S, F, T, P] = spectrogram(signal, 256, 128, 256, samplingRate, 'yaxis');
        
        % 스펙트로그램 데이터를 dB 단위로 변환합니다.
        PdB = 10*log10(abs(P));
        
        % 현재 반복의 스펙트로그램을 subplot으로 추가합니다.
        subplot(rows, cols, j);
        imagesc(T, F, PdB);

        title(sprintf('Channel %d, Trial %d', i, j));
        colorbar; % 컬러바를 추가하여 데시벨 단위의 색상을 표시합니다.
    end
    
    % 각 채널의 스펙트로그램을 모두 표시한 후에 그래프 창의 이름을 설정합니다.
    sgtitle(sprintf('Spectrograms for Channel %d', i));
end

%%
% 기본 변수 설정

load(inputPath + string(fileNums(3).name));

feature = prctile(signal(:), 90);
subplot(2,2,1);
[ value, frequency ] = getspec(signal(2,:), fs, 1);
value(1:5) = 0;
plot(frequency, value);
%plot(signal(2,:));
title(['CH1 RMS: ' num2str(feature)]);
xlim([0,1000]);

feature = prctile(signal(:), 90);
subplot(2,2,2);
[ value, frequency ] = getspec(signal(1,:), fs, 1);
value(1:5) = 0;
plot(frequency, value);
%plot(signal(2,:));
title(['CH2 RMS: ' num2str(feature)]);
xlim([0,1000]);

feature = prctile(signal(:), 90);
subplot(2,2,3);
[ value, frequency ] = getspec(signal(3,:), fs, 1);
value(1:5) = 0;
plot(frequency, value);
%plot(signal(2,:));
title(['CH3: ' num2str(feature)]);
xlim([0,1000]);

%%
% stft 코드

% 기본 변수 설정
nSignal = 25600;
samplingRate = 25600;
max = 50;
min = -50;
load(inputPath + string(fileNums(3).name));


windowSize = 256; % 윈도우 크기 설정
overlap = round(0.5 * windowSize); % 50% 오버랩
nfft = 1024; % FFT 포인트 수

% STFT 계산
[s, f, t] = spectrogram(signal(2,:), windowSize, overlap, nfft, fs);

% STFT 결과를 dB 스케일로 변환
s_db = 10 * log10(abs(s));

%%
% stft 코드

% 기본 변수 설정
nSignal = 25600;
samplingRate = 25600;
max = 50;
min = -50;
load(inputPath + string(fileNums(3).name));


windowSize = 256; % 윈도우 크기 설정
overlap = round(0.5 * windowSize); % 50% 오버랩
nfft = 1024; % FFT 포인트 수

% STFT 계산
[s, f, t] = spectrogram(signal(2,:), windowSize, overlap, nfft, fs);

% STFT 결과를 dB 스케일로 변환
s_db = 10 * log10(abs(s));

% STFT 결과 그리기
figure;
imagesc(t, f, s_db);
axis xy; 
xlabel('Time (s)'); ylabel('Frequency (Hz)');
title(['CH1 내륜결함']);
colorbar;
ylim([0,1000]);


%%
%
figure;
load(inputPath + string(fileNums(3).name));
feature = prctile(signal(:), 90);
value(1:5) = 0;
plot(signal(1, :));
%%
figure;
load(inputPath + string(fileNums(3).name));
feature = prctile(signal(:), 90);
[~, frequency] = getspec(signal(1, :), fs, 1);
value = abs(coefficients(2, :));
value(1:5) = 0;
plot(frequency, value(1:length(frequency)));
%%
%웨이블릿 
figure;
% 기본 변수 설정
nSignal = 25600;
samplingRate = 25600;
max_val = 50;
min_val = -50;
load(inputPath + string(fileNums(3).name));

% 웨이블릿 변환 함수 정의
coefficients = cwt(signal, 1:100, 'morl');

% CH1
feature = prctile(signal(:), 90);
[~, frequency] = getspec(signal(2, :), fs, 1);
value = abs(coefficients(2, :));
value(1:5) = 0;
plot(frequency, value(1:length(frequency)));

title(['CH1 RMS: ' num2str(feature)]);
xlim([0, 1000]);
t= 200;
%%
% CH2
feature = prctile(signal(:), 90);
subplot(2, 2, 2);
[~, frequency] = getspec(signal(1, :), fs, 1);
value = abs(coefficients(1, :));
value(1:5) = 0;
plot(frequency(1:min(length(frequency), length(value))), value(1:min(length(frequency), length(value))));
%plot(signal(2,:));
title(['CH2 RMS: ' num2str(feature)]);
xlim([0, 1000]);

% CH3
feature = prctile(signal(:), 90);
subplot(2, 2, 3);
[~, frequency] = getspec(signal(3, :), fs, 1);
value = abs(coefficients(3, :));
value(1:5) = 0;
plot(frequency(1:min(length(frequency), length(value))), value(1:min(length(frequency), length(value))));
%plot(signal(2,:));
title(['CH3: ' num2str(feature)]);
xlim([0, 1000]);



