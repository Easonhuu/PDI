# PDI
A paper accepted by the 2020 BIBM conference, which is to achieve high precision in protein-drug interaction by using GAT and Transformer.  
Source code and dataset for "Structure Enhanced Protein-Drug Interaction Prediction using Transformer and Graph Embedding"
### Reqirements:
- Pytorch=1.2.0
- torchvision=0.4.0 
- cudatoolkit=10.0 (View version: `cat /usr/local/cuda/version.txt`)
- Python3
- transformers=2.7.0 (How to install: [huggingface/transformers](https://github.com/huggingface/transformers#model-architectures))
### Prepare Data:
#### Pre-training phase:  
- Large-scale protein sequences

**Location:** `./data/total_modified_enzy_seqs-v2.csv`  
**Format:** 1.9G, 4168590 lines
&nbsp;|Accession_Code|Recommended Name|EC Number|Organism|Source|No of amino acids|Sequence
:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--|
1|P15807|NaN|1|Saccharomyces cerevisiae (strain ATCC 204508 / S288c)|Swiss-Prot|274|MVKSLQLAHQLKDKKILLIGGGEVGLTRLYKLI...
2|Q5SRE7|NaN|1|Homo sapiens|Swiss-Prot|291|MACLSPSQLQKFQQDGFLVLEGFLSAEECVAMQQRI...

We just need to use the last column (protein sequences), getting sequences by `./pre-training/get_sequences.py` and saving in `./data/corpus-enz416w.big`   
**Format:** 3.0G, 4168590 lines
|Protein sequences|
|:--|
|M V K S L Q L A H Q L K D K K I L L I G G G E V G L T R L Y K L I P T G C K L T L V S...|
|M A C L S P S Q L Q K F Q Q D G F L V L E G F L S A E E C V A M Q Q R I G E I V A E M D V...|

The data is divided into train-set and dev-set by `./pre-training/train_dev_split.py`, saving in `./data/corpus-enz416w-train.tsv` and `./data/corpus-enz416w-dev.tsv`.  
Next, we need to generate protein sequences's vocabulary by `./pre-training/generate_vocab.py`, saving in `./pre-training/protein_vocab.txt`

#### Fine-tuning phase:  
- The data of protein-drug interaction (PDBBind2016)  
**Location:** `./data/pdbbind2016.pkl`  
**Format:** 4.2G, 13435 lines

&nbsp;|PDB-ID|Affinity-Value|seq|rdkit_smiles|set|contact_map
:--:|:--:|:--:|:--:|:--:|:--:|:--:|
0|3zzf|0.4|NGFSATRSTV...|CC(=O)N[C@@H](CCC(=O)O)C(=O)O|train|[array([[ True,  True,  True, ..., False, False, False], ..., [False, False, False, ...,  True,  True,  True]])]]
1|11gs|5.82|PYTVVY...|CC[C@H](C(=O)...,CC[C@@H]...|train|[array([[ True,  True,  True, ..., False, False, False], ..., [False, False, False, ...,  True,  True,  True]])]

- train_df_nums: 11877, valid_df_nums: 1000, core2016_df_nums: 290, casf2013_df_nums: 195, casf2013_df_nums: 73

### Cite:
If you use the code, please cite this paper:
```bash
@inproceedings{hu2020bibm,
  title={Structure Enhanced Protein-Drug Interaction Prediction using Transformer and Graph Embedding},
  author={Hu, Fan and Hu, Yishen and Zhang, Jianye and Wang, Dongqi and Yin, Peng},
  booktitle={2020 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  year={2020}
}
```

