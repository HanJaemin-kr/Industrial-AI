# Dockerfile
# 기본 이미지 선택
FROM python:3.8

# 작업 디렉토리 생성
WORKDIR /app

# 소스 코드 복사
COPY . /app

# 필요한 라이브러리 설치
RUN pip install --no-cache-dir numpy keras matplotlib tensorflow

# Docker 컨테이너 실행 시 실행할 명령어
CMD ["python", "test.py"]

