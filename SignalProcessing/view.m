
dataRootPath = '\\203.250.78.96\Data\[20240322] 유화교실험\datas\impact2\AE';
saveRootPath = 'D:\[20240322] 유화교실험\result';
dataList = dir(dataRootPath);
figure;

for i = 3:numel(dataList)

    disp(dataList(i).name);

    fileName = fullfile(dataRootPath, dataList(i).name);
    datfile = fopen(string(fileName), 'r'); % 데이터 파일 열기
    
    %fseek(datfile, 2, 'bof');
    % 헤더 정보를 저장할 구조체 생성
    header = struct( ...
        'fid', fread(datfile, 1, 'int32'), ...
        'nch', fread(datfile, 1, 'int32'), ...
        'fs', fread(datfile, 1, 'int32'), ...
        'nsamp', fread(datfile, 1, 'int32'), ...
        'nbit', fread(datfile, 1, 'int32'), ...
        'minval', fread(datfile, 1, 'double'), ...
        'maxval', fread(datfile, 1, 'double') ...
    );
    
    % 데이터 부분 읽기
    databit = '';
    if(header.nbit == 16)
        databit = 'int16';
    elseif(header.nbit == 24)
        databit = 'int32';
    end
    
    data = fread(datfile, [header.nsamp, header.nch], databit);
    fclose(datfile);
    
    if(length(data(:,1)) == header.nsamp)
        raw = zeros(header.nsamp,header.nch);
        for ch = 1:size(data,2)
            for i = 1:header.nsamp
                value = data(i,ch).* header.maxval / (2^(header.nbit-1)-1);
                raw(i, ch) = value;
            end
        end
    
    end
    
    
    t = (0:header.fs/header.nsamp:header.nsamp-(1/header.fs));
    
    tit1 = " AE Ch1 (교량하부구조물 near)";
    tit2 = " AE Ch2 (기존센서 near)";
    tit3 = " AE Ch3 (교량하부구조물 far)";
    tit4 = " AE Ch4 (기존센서 far)";
    
    [tval1, tfreq1] = getspec(raw(:,1),header.fs,1);
    [tval2, tfreq2] = getspec(raw(:,2),header.fs,1);
    [tval3, tfreq3] = getspec(raw(:,3),header.fs,1);
    [tval4, tfreq4] = getspec(raw(:,4),header.fs,1);
    
    

    % 4개의 서브플롯 생성
    subplot(2, 4, 1); % 첫 번째 서브플롯
    plot(t,raw(:,1));
    title('Ch1 AE Raw');
    % 바닥 20240322_154857
    % 가까운 배관 2024043322_155737
    subplot(2, 4, 2); % 두 번째 서브플롯
    plot(t,raw(:,2));
    title('Ch2 AE Raw');
    
    subplot(2, 4, 3); % 세 번째 서브플롯
    plot(t,raw(:,3));
    title('Ch3 AE Raw');
    
    subplot(2, 4, 4); % 네 번째 서브플롯
    plot(t,raw(:,4));
    title('Ch4 AE Raw');
    
    % 4개의 서브플롯 생성
    subplot(2, 4, 5); % 첫 번째 서브플롯
    plot(tfreq1,tval1);
    title('Ch1 AE FFT');
    xlim([10000 100000]);
    
    subplot(2, 4, 6); % 두 번째 서브플롯
    plot(tfreq2,tval2);
    title('Ch2 AE FFT');
    xlim([10000 100000]);
    
    subplot(2, 4, 7); % 세 번째 서브플롯 
    plot(tfreq3,tval3);
    title('Ch3 AE FFT');
    xlim([10000 100000]);
    
    subplot(2, 4, 8); % 네 번째 서브플롯
    plot(tfreq4,tval4);
    title('Ch4 AE FFT');
    xlim([10000 100000]);

    pause(1)
end
%% AE 
% ch1 : 교량하부구조물 near
% ch2 : 기존센서 near
% ch3 : 교량하부구조물 far 
% ch4 : 기존센서 far

% Vib
% ch1 : 교량하부구조물 near
% ch3 : 교량하부구조물 far





fileName = fullfile(dataRootPath,'AE/20240322_150736_AE.dat');
datfile = fopen(string(fileName), 'r'); % 데이터 파일 열기

%fseek(datfile, 2, 'bof');
% 헤더 정보를 저장할 구조체 생성
header = struct( ...
    'fid', fread(datfile, 1, 'int32'), ...
    'nch', fread(datfile, 1, 'int32'), ...
    'fs', fread(datfile, 1, 'int32'), ...
    'nsamp', fread(datfile, 1, 'int32'), ...
    'nbit', fread(datfile, 1, 'int32'), ...
    'minval', fread(datfile, 1, 'double'), ...
    'maxval', fread(datfile, 1, 'double') ...
);

% 데이터 부분 읽기
databit = '';
if(header.nbit == 16)
    databit = 'int16';
elseif(header.nbit == 24)
    databit = 'int32';
end

data = fread(datfile, [header.nsamp, header.nch], databit);
fclose(datfile);

if(length(data(:,1)) == header.nsamp)
    raw = zeros(header.nsamp,header.nch);
    for ch = 1:size(data,2)
        for i = 1:header.nsamp
            value = data(i,ch).* header.maxval / (2^(header.nbit-1)-1);
            raw(i, ch) = value;
        end
    end

end


t = (0:header.fs/header.nsamp:header.nsamp-(1/header.fs));

tit1 = " AE Ch1 (교량하부구조물 near)";
tit2 = " AE Ch2 (기존센서 near)";
tit3 = " AE Ch3 (교량하부구조물 far)";
tit4 = " AE Ch4 (기존센서 far)";

[tval1, tfreq1] = getspec(raw(:,1),header.fs,1);
[tval2, tfreq2] = getspec(raw(:,2),header.fs,1);
[tval3, tfreq3] = getspec(raw(:,3),header.fs,1);
[tval4, tfreq4] = getspec(raw(:,4),header.fs,1);


% 4개의 서브플롯 생성
subplot(2, 4, 1); % 첫 번째 서브플롯
plot(t,raw(:,1));
title('Ch1 AE Raw');


subplot(2, 4, 2); % 두 번째 서브플롯
plot(t,raw(:,2));
title('Ch2 AE Raw');

subplot(2, 4, 3); % 세 번째 서브플롯
plot(t,raw(:,3));
title('Ch3 AE Raw');

subplot(2, 4, 4); % 네 번째 서브플롯
plot(t,raw(:,4));
title('Ch4 AE Raw');

% 4개의 서브플롯 생성
subplot(2, 4, 5); % 첫 번째 서브플롯
plot(tfreq1,tval1);
title('Ch1 AE FFT');
xlim([10000 100000]);

subplot(2, 4, 6); % 두 번째 서브플롯
plot(tfreq2,tval2);
title('Ch2 AE FFT');
xlim([10000 100000]);

subplot(2, 4, 7); % 세 번째 서브플롯 
plot(tfreq3,tval3);
title('Ch3 AE FFT');
xlim([10000 100000]);

subplot(2, 4, 8); % 네 번째 서브플롯
plot(tfreq4,tval4);
title('Ch4 AE FFT');
xlim([10000 100000]);

% 
% % Ch 1
% figure,plot(t,raw(:,1));
% title(strcat(tit1," raw"));
% 
% figure,plot(tfreq1,tval1);
% title(strcat(tit1," fft"));
% xlim([10000 100000]);
% 
% 
% % Ch 2
% figure,plot(t,raw(:,2));
% title(strcat(tit2," raw"));
% 
% figure,plot(tfreq2,tval2);
% title(strcat(tit2," fft"));
% xlim([10000 100000]);
% 
% % Ch 3
% figure,plot(t,raw(:,3));
% title(strcat(tit3," raw"));
% 
% figure,plot(tfreq3,tval3);
% title(strcat(tit3," fft"));
% xlim([10000 100000]);
% 
% % Ch 4
% figure,plot(t,raw(:,4));
% title(strcat(tit4," raw"));
% 
% figure,plot(tfreq4,tval4);
% title(strcat(tit4," fft"));
% xlim([10000 100000]);
% 
% 
% 


