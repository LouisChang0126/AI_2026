建立環境
```
conda env create -f environment.yml
conda activate env
```
Ant-v5
```
cd ant
python train_ant.py –algo {SAC, TD3, A2C, PPO}
python test_ant.py
python plot_reward_curve.py
python generate_video.py
```
Breakout-v5
```
cd breakout_v2
python train_breakout.py --num-envs NUM_ENV --algo {A2C,PPO} [--clip]
python test_breakout.py
python plot_reward_curve.py
python generate_video.py
```