# [[현재 공사 중]🏅[Dacon] 월간 데이콘 발화자의 감정인식 AI 경진대회](https://dacon.io/competitions/official/236027/overview/description)
<img src="/img/스크린샷 2023-03-11 오후 2.47.15.png" width="99%"></img>
## Competition Info
 - Period: 2022.11.01 - 2022.12.12
 - Joined as: Team with [A.RYANG](https://github.com/nomaday)
 - TEAM_NAME: '말하기전에생각했나요?'
 - TASK: `Classification`
 - Evaluation Metric: `Macro F1 Score`
 - Environment: Colab 
 
## Result "🏅 Prized as 5th"
 - PUBLIC  : 0.54517 | 11st /259 
 - PRIVATE : 0.54712 | 10th /259 
 - Final: 5th 
 
### 수상인증서
<img src="/img/1673529536217.jpeg" width="48%"></img>

-----------------
 
 
## install
#### [torchmetrics](https://torchmetrics.readthedocs.io/en/stable/)
```python
$ pip install -qqq torchmetrics
```

#### [colorama](https://github.com/tartley/colorama)
```python
$ pip install -qqq colorama
```

#### HuggingFace Transformer
```python
$ pip install -qqq --no-cache-dir transformers sentencepiece
```


## [link도 공사중] How to train or inference in CLI? [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1EPaUyBIP4VER23AKPoLBPjW-Gdn8Bv6b?usp=share_link)


### Train 
```python

$ python train.py --base_path './data/' \
                  --model_save '/content/drive/MyDrive/깃헙/Projects/Dacon Speakers/' \
                  --sub_path '/content/drive/MyDrive/깃헙/Projects/Dacon Speakers/' \
                  --model "monologg/kobigbird-bert-base" \
                  --add_speaker_info True\
                  --make_essay_option True\
                  --make_essay_sentences 4\
                  --grad_clipping True\
                  --n_folds 5 \
                  --n_epochs 5 \
                  --device 'cuda' \
                  --train_bs 16
  
``` 
- `base_path` : Data가 저장된 경로 (Default: `./data/`)
- `sub_path`  : `submission.csv` 제출하는 경로
- `model_save`: 학습된 모델이 저장되는 경로
- `clean_text`: `mecab` tokenizer로 데이터를 tokenize 시켰다가 다시 `" ".join`으로 복원시킬 것에 대한 여부
- `test_and_ss`: `test.csv`, `sample_submission.csv`파일을 사용 여부
- `model`: Huggingface의 Pratrained Model (Default: `"tae898/emoberta-base"`)
- `add_speaker_info`: `Speaker`를 대문자화 시킨 후 `Utterance`와 결합 (`Speaker`.upper() + `:` + `Utterance`)
- `make_essay_option`: 이전 N개의 index의 `Utternce`들과 현재 index의 `Utterance`를 결합 ([`utils.py`](https://github.com/renslightsaber/Dacon_Speakers_Emotion_Recognition/blob/main/utils.py) 참고)
  - `make_essay_sentences`: 결합할 이전 `Utternce`들의 수(= N)
- `n_folds`  : Fold 수
- `n_epochs` : Epoch
- `seed` : Random Seed (Default: 2022)
- `train_bs` : Batch Size (Default: 16)
- `max_length` : Max Length (Default: 128) for HuggingFace Tokenizer
- `grad_clipping`: [Gradient Clipping](https://neptune.ai/blog/understanding-gradient-clipping-and-how-it-can-fix-exploding-gradients-problem)
- `ratio` : 데이터를 Split하여 `train`(학습) 과 `valid`(성능 평가)를 만드는 비율을 의미. 정확히는 `train`의 Size 결정
- `device`: GPU를 통한 학습이 가능하다면, `cuda` 로 설정할 수 있다.
- `learning_rate`, `weight_decay`, `min_lr`, `T_max` 등은 생략 

- [`train.py`](https://github.com/renslightsaber/Dacon_Speakers_Emotion_Recognition/blob/main/train.py) 참고!   


### 주의
 - CLI 환경에서 train 시킬 때, `tqdm`의 Progress Bar가 엄청 많이 생성된다. 아직 원인과 해결을 못 찾은 상태이다.
 - Colab과 Jupyter Notebook에서는 정상적으로 Progress Bar가 나타난다.


### Inference 
```python

$ python inference.py --base_path './data/' \
                      --model_save '/content/drive/MyDrive/ ... /Dacon Speakers/' \
                      --sub_path '/content/drive/MyDrive/ ... /Dacon Speakers/' \
                      --model "monologg/kobigbird-bert-base" \
                      --add_speaker_info True\
                      --make_essay_option True\
                      --make_essay_sentences 4\
                      --n_folds 5 \
                      --n_epochs 5 \
                      --device 'cuda' \
                      --train_bs 16

```
- `base_path` : Data가 저장된 경로 (Default: `./data/`)
- `sub_path`  : `submission.csv` 제출하는 경로
- `model_save`: 학습된 모델이 저장되는 경로
- `model`: Huggingface의 Pratrained Model (Default: `"monologg/kobigbird-bert-base"`)
- `n_folds`  : `train.py`에서 진행항 KFold 수
- `n_epochs` : train했을 때의 Epoch 수 (submission 파일명에 사용)  
- `device`: GPU를 통한 학습이 가능하다면, `cuda` 로 설정할 수 있다.


#### [link도 공사중]Jupyter Notebook Version: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mB05pvu7d83KQX6dyYlj4jxdeUSEt7pJ?usp=sharing) 



