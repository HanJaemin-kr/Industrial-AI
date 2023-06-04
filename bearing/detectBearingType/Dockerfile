# 베이스 이미지 설정
FROM python:3.9

# 작업 디렉토리 생성
WORKDIR /app

# 필요한 파일 복사
COPY requirements.txt .

# 필요한 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY . .

# 실행할 명령어 설정
CMD [ "python", "main.py" ]

