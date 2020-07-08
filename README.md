# ArMod


## Dataset and Model Architecture


Aiming to develop  an Arabic MT models that can translate efficiently across different text domains, we make use of a large of parallel  
sentences datasets extracted from the Open Parallel Corpus (OPUS)(Tiedemann. 2012). OPUS contains more than 2.7 billion parallel sentences in 90 languages.
To train our models, we extract more than 285.6 M unique parallel  sentences from the whole collection [(available here)](https://github.com/EMNLP-2020/ArData). 
For each language pair we split the data into 80% for training, 10% development, and 10% test. 

## Pre-Processing

Firstly, we normalize punctuation and tokenize all sentences with the Moses tokenizer. We have also used joint Byte-Pair Encoding  (BPE) with 32 split operations for subword segmentation. 

## Model Architecture
 Our models are  mainly based on a CNN architecture [(Gehring et al., 2017)](https://arxiv.org/abs/1705.03122) . This convolutional model exploits  BPE.
 The architecture is as follows: 20 layers in the encoder and 20 layers in the decoder,  a multiplicative attention in every decoder layer, a kernel  width of 3  for both the encoder and the decoder, 
a hidden and an embedding size of 512, and 256 for the encoder and decoder layers. 



We train our proposed **Arabic&harr;13-languages** MT models on the parallel data  described [(here)](https://github.com/EMNLP-2020/ArData) using 4  GPUs for 7 days (for each model).  
For all models, the learning rate was set to 0.25, a dropout of 0.2, 512 as batch size, and a maximum tokens of 4,000. All this models are available in the table below:





| **Languge** | BG | CS | DE | EL | EN | ES |FR| HU | IT | PT | RU | TU | ZH |
| ------  | ------ | ------- | ------ | ----  | ------ | ------- | ------- | ------ | ----  | ------ | ------- | ------- | ------ |
| **AR**  &rarr;| [Link](https://drive.google.com/drive/folders/1Yq7EhhMZ3NMTE09-ddxtPQKkBp3vCuJX?usp=sharing)|  [Link](https://drive.google.com/drive/folders/1Yq7EhhMZ3NMTE09-ddxtPQKkBp3vCuJX?usp=sharing)|  [Link](https://drive.google.com/drive/folders/1Yq7EhhMZ3NMTE09-ddxtPQKkBp3vCuJX?usp=sharing) |  [Link](https://drive.google.com/drive/folders/1Yq7EhhMZ3NMTE09-ddxtPQKkBp3vCuJX?usp=sharing) |  [Link](https://drive.google.com/drive/folders/1Yq7EhhMZ3NMTE09-ddxtPQKkBp3vCuJX?usp=sharing)|[Link](https://drive.google.com/drive/folders/1Yq7EhhMZ3NMTE09-ddxtPQKkBp3vCuJX?usp=sharing)|[Link](https://drive.google.com/drive/folders/1Yq7EhhMZ3NMTE09-ddxtPQKkBp3vCuJX?usp=sharing)| [Link](https://drive.google.com/drive/folders/1Yq7EhhMZ3NMTE09-ddxtPQKkBp3vCuJX?usp=sharing)|[Link](https://drive.google.com/drive/folders/1Yq7EhhMZ3NMTE09-ddxtPQKkBp3vCuJX?usp=sharing)|[Link](https://drive.google.com/drive/folders/1Yq7EhhMZ3NMTE09-ddxtPQKkBp3vCuJX?usp=sharing)|[Link](https://drive.google.com/drive/folders/1Yq7EhhMZ3NMTE09-ddxtPQKkBp3vCuJX?usp=sharing)|[Link](https://drive.google.com/drive/folders/1Yq7EhhMZ3NMTE09-ddxtPQKkBp3vCuJX?usp=sharing)|[Link](https://drive.google.com/drive/folders/1Yq7EhhMZ3NMTE09-ddxtPQKkBp3vCuJX?usp=sharing)|
| **AR**   &larr; |[Link](https://drive.google.com/drive/folders/1Yq7EhhMZ3NMTE09-ddxtPQKkBp3vCuJX?usp=sharing)|[Link](https://drive.google.com/drive/folders/1Yq7EhhMZ3NMTE09-ddxtPQKkBp3vCuJX?usp=sharing)|[Link](https://drive.google.com/drive/folders/1Yq7EhhMZ3NMTE09-ddxtPQKkBp3vCuJX?usp=sharing)|[Link](https://drive.google.com/drive/folders/1Yq7EhhMZ3NMTE09-ddxtPQKkBp3vCuJX?usp=sharing)|[Link](https://drive.google.com/drive/folders/1Yq7EhhMZ3NMTE09-ddxtPQKkBp3vCuJX?usp=sharing)|[Link](https://drive.google.com/drive/folders/1Yq7EhhMZ3NMTE09-ddxtPQKkBp3vCuJX?usp=sharing)|[Link](https://drive.google.com/drive/folders/1Yq7EhhMZ3NMTE09-ddxtPQKkBp3vCuJX?usp=sharing)|[Link](https://drive.google.com/drive/folders/1Yq7EhhMZ3NMTE09-ddxtPQKkBp3vCuJX?usp=sharing)|[Link](https://drive.google.com/drive/folders/1Yq7EhhMZ3NMTE09-ddxtPQKkBp3vCuJX?usp=sharing)|[Link](https://drive.google.com/drive/folders/1Yq7EhhMZ3NMTE09-ddxtPQKkBp3vCuJX?usp=sharing)| [Link](https://drive.google.com/drive/folders/1Yq7EhhMZ3NMTE09-ddxtPQKkBp3vCuJX?usp=sharing)|[Link](https://drive.google.com/drive/folders/1Yq7EhhMZ3NMTE09-ddxtPQKkBp3vCuJX?usp=sharing)|[Link](https://drive.google.com/drive/folders/1Yq7EhhMZ3NMTE09-ddxtPQKkBp3vCuJX?usp=sharing)|
---

**(Tiedemann. 2012)** Jorg Tiedemann. 2012. Parallel data, tools and interfaces in opus. 2012:2214–2218.




## Training Example 


### Preprocess/binarize the data
```
src=ar
tgt=en
TEXT=Data_Input.tokenized.$src.$tgt
fairseq-preprocess --source-lang $src --target-lang $tgt \
     --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
     --destdir data-bin/$Data_Input.tokenized.$src-$tgt
```
### Training the ar-en model

```
fairseq-train data-bin/$Data_Input.tokenized.$src-$tgt  --source-lang  $tgt  --target-lang  $src \ 
--lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 --arch fconv   \
--save-dir checkpoints/Bench-fconv-$tgt-$src.sub
```

## Interactive Translation Examples

### Pre-trained Models
```
Beam=10
Best_tr=1
fairseq-interactive data-bin/$Data_Input.tokenized.$src-$tgt  \
  --path $Chk_pt_path/$Checkpoint  \
  --beam $Beam --nbest $Best_tr --tokenizer moses    --bpe subword_nmt \
  --bpe-codes $Codes  --remove-bpe  \
  --source-lang $src  --target-lang $tgt 
  
```
### Output
```

S-738	اعطيه هذا و اخبريه انني كنت امزح
T-738	tell him i was just kidding .
H-738	-0.2692852318286896	give him this and tell him i was joking .
P-738	-0.2532 -0.7831 -0.5164 -0.3192 -0.0928 -0.0160 -0.0965 -0.2905 -0.8000 -0.0025 -0.0599 -0.0014

S-4322	اعرف كل ما قام به للعثور علي عمل .
T-4322	i know all he has done to find work .
H-4322	-0.47998669743537903	i know all he did to find a job .
P-4322	-0.1533 -0.0191 -0.6882 -0.4971 -2.2777 -0.2763 -0.0512 -0.7616 -0.4628 -0.0912 -0.0014

```
