# VotePLMs-AFP: Identification of Antifreeze Proteins Using Transformer-Embedding Features and Ensemble Learning
## 1.Introduction
We proposed a novel predictor based on transformer-embedding features and ensemble learning for the identification of AFPs, termed VotePLMs-AFP.The experimental results show that our model achieves high prediction accuracy in 10-fold cross-validation (CV) and independent set testing, outperforming existing state-of-the-art methods. Therefore, our model could serve as an effective tool for predicting AFPs.
## 2.Requirement
numpy==1.20.3<br>
pandas==1.5.3<br>
scikit_learn==1.3.2<br>
torch==1.9.0+cu102<br>
torchvision==0.9.1+cu102
## 3.Usage
run `Pre_trained_models/ProtT5_basic.py` to generate ProtT5 feature files.<br>
run `Pre_trained_models/ESM-1b_basic.py` to generate ESM-1b feature files.<br>
run `Pre_trained_models/Tape_basic.py` to generate Tape feature files.<br>
run `Independent test/5-fold CV.py` to obtain the results of the 5-fold CV of the model.<br>
run `Independent test/main.py` to obtain the results of the 10-fold CV and independent test set of the model.
