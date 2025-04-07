# iPro-MP
a BERT-based model for the prediction of multiple prokaryotic promoters
![1 流程图](https://github.com/user-attachments/assets/dbb3d177-73d2-4e08-a87c-c65aab79e96c)
# 1. Environment setup
```python
conda create -n dna python=3.8
conda activate dna

# (optional if you would like to use flash attention)
git clone https://github.com/openai/triton.git;
cd triton/python;
pip install cmake; # build-time dependency
pip install -e .

# install required packages
python3 -m pip install -r requirements.txt
```
