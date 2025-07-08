# iPro-MP
**a BERT-based model for the prediction of multiple prokaryotic promoters**
![Figure1](https://github.com/user-attachments/assets/e8639c49-1822-472a-bf98-3f954216cba2)


---

## 🌟 Features

- Supports **23 prokaryotic species**
- Built on **DNABERT**, with customizable k-mer tokenization
- Enables **cross-species prediction**, independent testing, and retraining
- Includes **pre-trained models** and benchmarking datasets
- Suitable for use in **synthetic biology** and **functional genomics**

---

## 🚀 Quick Start
# 1. File information
- Benchmark Dataset : contains **Train** and **Test** data
- iPro-MP_train.py : the source code for training models
- iPro-MP_predict.py : the code for predicting promoters

# 2. Model Downloading
- The fine-tuned models for 23 species were deposited to the Zenodo repository and are available at https://doi.org/10.5281/zenodo.15180138.
- Please download the corresponding model based on the species to be predicted. The first number indicates the **Species_ID**.

# 3. Environment setup
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
# 4. Predict
**After downloading the model and setting up the environment, please run the following command for prediction.**
```python
python iPro-MP_predict.py -i example.fasta -s species_ID -o outputfile

## -i example.fasta
This parameter specifies the example.fasta file containing the DNA sequences that you want to predict. The file should be in FASTA format.

## -s species_ID
species_ID = ["Acinetobacter baumannii ATCC 17978", # 1
            "Bradyrhizobium japonicum USDA 110",   # 2
            "Burkholderia cenocepacia J2315", # 3
            "Campylobacter jejuni RM1221", # 4
            "Campylobacter jejuni subsp. jejuni 81116", # 5
            "Campylobacter jejuni subsp. jejuni 81-176", # 6
            "Campylobacter jejuni subsp. jejuni NCTC 11168", # 7
            "Corynebacterium diphtheriae NCTC 13129", # 8
            "Corynebacterium glutamicum ATCC 13032", # 9
            "Escherichia coli str K-12 substr. MG1655", # 10
            "Haloferax volcanii DS2", # 11
            "Helicobacter pylori strain 26695", # 12
            "Nostoc sp. PCC7120",  # 13
            "Paenibacillus riograndensis SBR5", # 14
            "Pseudomonas putida KT2440",  # 15
            "Shigella flexneri 5a str. M90T", # 16
            "Sinorhizobium meliloti 1021", # 17
            "Staphylococcus aureus subsp. aureus MW2", # 18
            "Staphylococcus epidermidis ATCC 12228", # 19
            "Synechococcus elongatus PCC 7942", # 20
            "Thermococcus kodakarensis KOD1", # 21
            "Xanthomonas campestris pv. campestrie B100",  # 22
            "Bacillus subtilis subsp. subtilis str. 168"   #23
            ]
The species_ID should be an integer between 1 and 23, corresponding to one of the 23 species in the above list.

## -o outputfile
This parameter allows the user to specify the output file name where the prediction results will be saved in ./Predict_Results (.csv format). 
```

# 5. Training (obtional)
- If you want to retrain or fine-tune the model, you can modify iPro-MP_train.py file.

# 6. Citation
If you use iPro_MP in your work, please kindly cite our paper:
Wei Su, Yuhe Yang, Yafei Zhao, Shishi Yuan, Xueqin Xie, Yuduo Hao, Hongqi Zhang, Dongxin Ye, Hao Lyu, Hao Lin. iPro-MP: a BERT-based model for the prediction of multiple prokaryotic promoters

If you have any question, please contact us (<820229344@qq.com>)


