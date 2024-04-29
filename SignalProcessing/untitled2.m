
% ==================
% setting
% 1x = 30 Hz 

% 현재 작업 디렉토리를 얻음
TargetDir = 'innerfault';
%
baseDirPrefix = 'C:\Users\dudrn\Desktop\research\Data\';
inputPath = fullfile(baseDirPrefix, TargetDir, 'Vibration\');
fileNums = dir(inputPath);
clear TargetDir baseDirPrefix;

% 기본 변수 설정
nSignal = 25600;
samplingRate = 25600;
max = 50;
min = -50;
load(inputPath + string(fileNums(3).name));



%%
feature = 0; % 특징 값을 0으로 초기화
signalDuration = nSignal / samplingRate; % 초 단위 신호 길이 계산
startFreq = 1000 * signalDuration; % 시작 주파수 계산 (1kHz)

magnitudes = signal(1,:);

% 끝 주파수 계산 (10kHz or fmax)
if 10000 < samplingRate / 2.56
    endFreq = 10000 * signalDuration;
else
    endFreq = samplingRate / 2.56;
end


for i = floor(startFreq):ceil(endFreq)
    feature = feature + magnitudes(i) * magnitudes(i);
end

feature = sqrt(feature); % 1kHz ~ 10kHz의 Overall 값 계산


%% 
% 1x = 30 Hz 

figure;

subplot(2,2,1); 
[ value, frequency ] = getspec(signal(1,:), fs, 1);
value = value(1:1000);
frequency = frequency(1:1000);

% 백분위수 계산
percentile_value = prctile(value, 80);

% 상위 80% 값 추출
upper_80_percentile_values = value(value >= percentile_value);

% 하위 20% 값 0으로 설정
value(value < percentile_value) = 0;

plot(frequency, value);
feature = prctile(value, 80);
%plot(signal(1,:));
title(['CH : 1 : ' num2str(feature)]);
xlim([0,1000]);



feature = prctile(signal(:), 90);
subplot(2,2,2);
[ value, frequency ] = getspec(signal(1,:), fs, 1);
value(1:5) = 0;
plot(frequency, value);
%plot(signal(2,:));
title(['CH : 2 : ' num2str(feature)]);
xlim([0,1000]);




%%

feature = prctile(signal(:), 90);
subplot(2,2,1);
[ value, frequency ] = getspec(signal(1,:), fs, 1);
value(1:5) = 0;
plot(frequency, value);
%plot(signal(2,:));
title(['CH : 1 : 0.7M, RMS: ' num2str(feature)]);
xlim([0,1000]);

feature = prctile(signal(:), 90);
subplot(2,2,3);
[ value, frequency ] = getspec(signal(3,:), fs, 1);
plot(signal(3,:));
value(1:5) = 0;
plot(frequency, value);
%plot(signal(3,:));
title(['CH : 3 : 60.7M, RMS: ' num2str(feature)]);
xlim([0,1000]);



feature = prctile(signal(:), 90);
subplot(2,2,2);
[ value, frequency ] = getspec(signal(2,:), fs, 1);
value(1:5) = 0;
plot(frequency, value);
%plot(signal(2,:));
title(['CH : 2 : 0.7M, RMS: ' num2str(feature)]);
xlim([0,1000]);

feature = prctile(signal(:), 90);
subplot(2,2,3);
[ value, frequency ] = getspec(signal(3,:), fs, 1);
plot(signal(3,:));
value(1:5) = 0;
plot(frequency, value);
%plot(signal(3,:));
title(['CH : 3 : 60.7M, RMS: ' num2str(feature)]);
xlim([0,1000]);







%% 하모닉 
figure;
for i=1:4
    rms = sqrt(mean((signal(i,:)).^2));
    subplot(2,2,i); 
    [ value, frequency ] = getspec(signal(i,:), fs, 1);
    value(1:5) = 0;
    plot(frequency, value);
    hold on;
    xlim([0 fundamental*numHarmonics]); % set x-axis limits
    for j = 1:numHarmonics
        harmonic = j * fundamental;
        idx = find(frequency >= harmonic, 1); % find index of the harmonic
        plot([harmonic harmonic], [0 value(idx)], 'r'); % plot harmonic line
       % text(harmonic, value(idx), num2str(value(idx)), 'EdgeColor', 'k'); % add boxed text
    end
    hold off;
    title(['CH : ' num2str(i) ', RMS: ' num2str(rms)]);
end



%% stft 

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
imagesc(t, f, s_db);
axis xy; 
xlabel('Time (s)'); ylabel('Frequency (Hz)');
title(['CH1 외륜결함']);
colorbar;
ylim([0,1000]);