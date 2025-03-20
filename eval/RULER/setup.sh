conda install -c nvidia cuda-nvcc
pip install -force_reinstall -r requirements.txt 
pip install flash-attn==2.6.0.post1 --no-build-isolation

pip install git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/rotary
pip install causal-conv1d==1.4.0
pip install mamba-ssm==2.2.2 

python3 -c "import nltk; nltk.download('punkt')"


# prepare data
cd scripts/data/synthetic/json/
python download_paulgraham_essay.py
bash download_qa_dataset.sh

cd ../../..