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
## Pre-trained Models

We train our proposed **Arabic&harr;13-languages** MT models on the parallel data  described [(here)](https://github.com/EMNLP-2020/ArData) using 4  GPUs for 7 days (for each model).  
For all models, the learning rate was set to 0.25, a dropout of 0.2, 512 as batch size, and a maximum tokens of 4,000. All this models are available in the table below:





| **Languge** | BG | CS | DE | EL | EN | ES |FR| HU | IT | PT | RU | TU | ZH |
| ------  | ------ | ------- | ------ | ----  | ------ | ------- | ------- | ------ | ----  | ------ | ------- | ------- | ------ |
| **AR**  &rarr;| [Link](https://drive.google.com/drive/folders/1Yq7EhhMZ3NMTE09-ddxtPQKkBp3vCuJX?usp=sharing)|  [Link](https://drive.google.com/drive/folders/1Yq7EhhMZ3NMTE09-ddxtPQKkBp3vCuJX?usp=sharing)|  [Link](https://drive.google.com/drive/folders/1Yq7EhhMZ3NMTE09-ddxtPQKkBp3vCuJX?usp=sharing) |  [Link](https://drive.google.com/drive/folders/1Yq7EhhMZ3NMTE09-ddxtPQKkBp3vCuJX?usp=sharing) |  [Link](https://drive.google.com/drive/folders/1Yq7EhhMZ3NMTE09-ddxtPQKkBp3vCuJX?usp=sharing)|[Link](https://drive.google.com/drive/folders/1Yq7EhhMZ3NMTE09-ddxtPQKkBp3vCuJX?usp=sharing)|[Link](https://drive.google.com/drive/folders/1Yq7EhhMZ3NMTE09-ddxtPQKkBp3vCuJX?usp=sharing)| [Link](https://drive.google.com/drive/folders/1Yq7EhhMZ3NMTE09-ddxtPQKkBp3vCuJX?usp=sharing)|[Link](https://drive.google.com/drive/folders/1Yq7EhhMZ3NMTE09-ddxtPQKkBp3vCuJX?usp=sharing)|[Link](https://drive.google.com/drive/folders/1Yq7EhhMZ3NMTE09-ddxtPQKkBp3vCuJX?usp=sharing)|[Link](https://drive.google.com/drive/folders/1Yq7EhhMZ3NMTE09-ddxtPQKkBp3vCuJX?usp=sharing)|[Link](https://drive.google.com/drive/folders/1Yq7EhhMZ3NMTE09-ddxtPQKkBp3vCuJX?usp=sharing)|[Link](https://drive.google.com/drive/folders/1Yq7EhhMZ3NMTE09-ddxtPQKkBp3vCuJX?usp=sharing)|
| **AR**   &larr; |[Link](https://drive.google.com/drive/folders/1Yq7EhhMZ3NMTE09-ddxtPQKkBp3vCuJX?usp=sharing)|[Link](https://drive.google.com/drive/folders/1Yq7EhhMZ3NMTE09-ddxtPQKkBp3vCuJX?usp=sharing)|[Link](https://drive.google.com/drive/folders/1Yq7EhhMZ3NMTE09-ddxtPQKkBp3vCuJX?usp=sharing)|[Link](https://drive.google.com/drive/folders/1Yq7EhhMZ3NMTE09-ddxtPQKkBp3vCuJX?usp=sharing)|[Link](https://drive.google.com/drive/folders/1Yq7EhhMZ3NMTE09-ddxtPQKkBp3vCuJX?usp=sharing)|[Link](https://drive.google.com/drive/folders/1Yq7EhhMZ3NMTE09-ddxtPQKkBp3vCuJX?usp=sharing)|[Link](https://drive.google.com/drive/folders/1Yq7EhhMZ3NMTE09-ddxtPQKkBp3vCuJX?usp=sharing)|[Link](https://drive.google.com/drive/folders/1Yq7EhhMZ3NMTE09-ddxtPQKkBp3vCuJX?usp=sharing)|[Link](https://drive.google.com/drive/folders/1Yq7EhhMZ3NMTE09-ddxtPQKkBp3vCuJX?usp=sharing)|[Link](https://drive.google.com/drive/folders/1Yq7EhhMZ3NMTE09-ddxtPQKkBp3vCuJX?usp=sharing)| [Link](https://drive.google.com/drive/folders/1Yq7EhhMZ3NMTE09-ddxtPQKkBp3vCuJX?usp=sharing)|[Link](https://drive.google.com/drive/folders/1Yq7EhhMZ3NMTE09-ddxtPQKkBp3vCuJX?usp=sharing)|[Link](https://drive.google.com/drive/folders/1Yq7EhhMZ3NMTE09-ddxtPQKkBp3vCuJX?usp=sharing)|
---

**(Tiedemann. 2012)** Jorg Tiedemann. 2012. Parallel data, tools and interfaces in opus. 2012:2214–2218.
