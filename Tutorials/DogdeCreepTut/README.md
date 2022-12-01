# Sonic-the-Hedgehog-A3C-LSTM-tensorflow2
![image](https://github.com/Chang-Chia-Chi/Sonic-the-Hedgehog-A3C-LSTM-tensorflow2/blob/main/pics/zone1-act1.gif)
![image](https://github.com/Chang-Chia-Chi/Sonic-the-Hedgehog-A3C-LSTM-tensorflow2/blob/main/pics/zone1-act3.gif)     

## Introduction
Trained an AI playing Sonic the Hedgehog by Deep Reinforcement Learning model:     
**Asynchronous Advantage Actor-Critic(A3C)** with LSTM cell using tensorflow2-keras API

## Setup
1. Clone the repository:`https://github.com/Chang-Chia-Chi/Sonic-the-Hedgehog-A3C-LSTM-tensorflow2.git`    
2. Run `pip3 install -r requirements.txt` to install packages required.  
3. Sonic 1, 2 and 3 & Knuckles ROMs are available on Steam:   
    - [Sonic The Hedgehog](https://store.steampowered.com/app/71113/Sonic_The_Hedgehog/)
    - [Sonic The Hedgehog 2](http://store.steampowered.com/app/71163/Sonic_The_Hedgehog_2/)
    - [Sonic 3 & Knuckles](http://store.steampowered.com/app/71162/Sonic_3___Knuckles/)
4. Once you buy any of games, use script `python -m retro.import.sega_classics`. You'll be asked to type   
   Steam username, password and Guard code:   
   - Open a private session and log into Steam, there will be a guide code shown as below:   
   ![image](https://imageproxy.ifunny.co/crop:x-20,resize:640x,quality:90x75/images/d3b281b12de983d5670091372d2fca0b2a45b7de17f312e308e41d4ff914a5ed_1.jpg)
5. After installation, use script `python -m retro.import <path to steam folder>`
6. For more detail please follow this [link](https://contest.openai.com/2018-1/details/)

## Train
1. Open `parser.py`, change argements of `--game and --state` to which Sonic game, Zone and Act you want to train.   
   You could find all levels in these two links: -[sonic-train.csv](https://contest.openai.com/2018-1/static/sonic-train.csv), -[sonic-validation.csv](https://contest.openai.com/2018-1/static/sonic-validation.csv)
2. run `python train.py` to start training.
3. By default, it'll save model weights, model performance in distance, and video (.bk2) every 10 episodes.
4. model and training setting parameters are in `parser.py`.
5. If you want to load pre-trained weights, open `train.py` and set `pretrain=True`.`
## Test
1. Open test.py, change `game` and `state` to level you want.
2. If you don't want to record gameplay, set `record=False`
3. Run `python test.py`
4. `P.S. Uploaded pre-trained weights are trained with Impala CNN model`
## Enviroment
- **python == 3.7**
- **tensorflow == 2.3.0**

## Render .bk2 record to Mp4  
1. Run script `python3 -m retro.scripts.playback_movie <Record file name>.bk2 <output file name>.mp4`
2. For more detail please follow this [link](https://retro.readthedocs.io/en/latest/python.html#replay-files)

## Reference
1. https://contest.openai.com/2018-1/details/
2. https://retro.readthedocs.io/en/latest/python.html#replay-files
