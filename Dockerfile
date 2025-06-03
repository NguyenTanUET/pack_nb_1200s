# --- STEP 1: Chọn base image
FROM python:3.10-slim

# --- STEP 2: Thiết lập working directory
WORKDIR /app

# --- STEP 3: Copy requirements và cài dependencies
COPY requirements.txt .
# Nếu trong requirements.txt có "cplex", Docker sẽ pip install cplex
# Lưu ý: cplex trên PyPI đã bao gồm cả engine (nếu bạn đã làm chạy local chỉ với "pip install cplex")
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir google-cloud-storage

# --- STEP 4: Copy toàn bộ source code và data vào container
COPY . .

# --- STEP 5: Chỉ định ENTRYPOINT hoặc CMD để container tự chạy batch job khi start
# Ở đây ta export PYTHONUNBUFFERED=1 để đảm bảo mọi print đều flush ngay
ENV PYTHONUNBUFFERED=1

# Tạo một tập hợp lệnh shell, chạy tuần tự 4 script, ghi log ra stdout
CMD [ "bash", "-lc", "\
     echo '>>> Container started, running RCPSP batch...'; \
     python rcpsp_pack.py; \
     echo '>>> rcpsp_pack.py'; \
     echo '>>> All scripts done, exiting.' \
" ]
