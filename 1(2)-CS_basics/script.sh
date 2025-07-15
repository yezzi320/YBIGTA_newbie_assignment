
# anaconda(또는 miniconda)가 존재하지 않을 경우 설치해주세요!
## TODO
curl -O https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh
bash Anaconda3-2024.02-1-Linux-x86_64.sh -b -p $HOME/anaconda3

# Conda 환셩 생성 및 활성화
## TODO
conda init bash

if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
elif [ -f "$HOME/.bash_profile" ]; then
    source "$HOME/.bash_profile"
fi

conda create -y -n myenv python=3.10
conda activate myenv

## 건드리지 마세요! ##
python_env=$(python -c "import sys; print(sys.prefix)")
if [[ "$python_env" == *"/envs/myenv"* ]]; then
    echo "[INFO] 가상환경 활성화: 성공"
else
    echo "[INFO] 가상환경 활성화: 실패"
    exit 1 
fi

# 필요한 패키지 설치
## TODO
pip install mypy

# Submission 폴더 파일 실행
cd submission || { echo "[INFO] submission 디렉토리로 이동 실패"; exit 1; }

for file in *.py; do
    ## TODO
    num=${file%.py}
    id=${num#*_}            
    python "$file" < "../input/${id}_input" > "../output/${id}_output"
done

# mypy 테스트 실행 및 mypy_log.txt 저장
## TODO
for file in *.py; do
    echo "mypy result: $file" >> ../mypy_log.txt
    mypy "$file" >> ../mypy_log.txt
    echo "------------" >> ../mypy_log.txt
done

# conda.yml 파일 생성
## TODO
conda env export > ../conda.yml

# 가상환경 비활성화
## TODO
conda deactivate
