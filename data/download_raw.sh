# !/bin/bash
curl -L -o ./data/vivos-vietnamese-speech-corpus-for-asr.zip\
  https://www.kaggle.com/api/v1/datasets/download/kynthesis/vivos-vietnamese-speech-corpus-for-asr

# #!/bin/bash
curl -L -o ./data/demand-dataset.zip\
  https://www.kaggle.com/api/v1/datasets/download/aanhari/demand-dataset

unzip -q data/vivos-vietnamese-speech-corpus-for-asr.zip -d ./data
unzip -q data/demand-dataset.zip -d ./data

rm -rf data/demand/demand