## Install dependencies
Create environment: `conda env create -f environment.yaml`
Activate Environment: `conda activate neural_pathway`

---

# Offline Multitasks 

* `Directory: Offline\`

#### Trained agents
* [Here](https://drive.google.com/drive/folders/1OPT_x4OUt5wvPzwkhc9oi8N6ofwUHwp4?usp=share_link)

#### Run trained agents
`python run_trained_agent.py --env_name metaworld --algo_name IQL --eval_episodes 100 --seed 0 --clip_grad --log_dir trained_data --save_video`

#### Expert dataset
* [Here](https://drive.google.com/drive/folders/1oPnc12P3PCowJ-AWufGgpOMWov21PYyu?usp=share_link)

#### Train agent
`python main_parallel.py --algo_name='IQL' --env_name='metaworld' --clip_grad`
* More demonstrations of trained agents are [here](https://drive.google.com/drive/folders/14OpiARNI_hOucnEdgpTPmCR_PvKm_SdI?usp=share_link)





---
**MetaWorld MT10**

<p float="left">
  <img src="https://imgur.com/q1LLvBs.gif" width="200" />
  <img src="https://imgur.com/NA62a8y.gif" width="200" /> 
  <img src="https://imgur.com/7AjlpWC.gif" width="200" />
  <img src="https://imgur.com/3sTpeXU.gif" width="200" />
  <img src="https://imgur.com/AY15kCv.gif" width="200" />
  <img src="https://imgur.com/0imjl9P.gif" width="200" />
  <img src="https://imgur.com/0zBdb8a.gif" width="200" /> 
  <img src="https://imgur.com/zlOomqc.gif" width="200" />
  <img src="https://imgur.com/7g7LV2q.gif" width="200" />
  <img src="https://imgur.com/J70wCYm.gif" width="200" />
</p>


---

# Online Multitasks 
* `Directory: Online\`

#### Train agent

* **Train for single task**
`python train.py seed=0 hidden_dim=1024 batch_size=1024 env="halfcheetah_forward" env_type="multitask" keep_ratio=0.05 iterative_pruning=True continual_pruning=False lr=0.001 mask_update_mavg=5`
    * **Important hyper-parameter**:
        * `env` choose from `halfcheetah_forward, halfcheetah_backward, halfcheetah_jump, halfcheetah_forward_jump, halfcheetah_backward_jump`
        * `mask_update_mavg`: if >1 it uses equation $(2)$.
        * `continual_pruning` : when sets `False`, it check if performance is reached to `threshold` value. For Halfcheetah it's fixed to be episodic return 2000. 
            - if `True` and `mask_update_mavg=1` will generate `figure 3(a) (DAPD)`
            - if `False` and `mask_update_mavg=1` will generate `figure 3(a) (DAPD) with threhold`
    * **Baseline performance**:
        - `python train.py seed=0 hidden_dim=1024 batch_size=1024 env="halfcheetah_forward" env_type="multitask" keep_ratio=0.05 iterative_pruning=False continual_pruning=False lr=0.001 mask_update_mavg=1`

* **Train MetaWorld multitask**
`python run_multitask.py --seed=0 --hidden_dim=400 --batch_size=128 --keep_ratio=0.05 --lr=0.001 --mask_init_method=random --mask_update_mavg=1 --ips_threshold=800 --env_type=metaworld --grad_update_rule=pmean --env MT10 --snip_itr 1 --hidden_depth 3 --experiment B1M_HD400_D3_pretrained_mask_MT10 --num_train_steps 2000000 --optimization_type "adamW" --num_pretrain_steps=10000 --pretrain_masks`
`
    * **Important hyper-parameter**:
        * `mask_update_mavg`:  if >1 it uses equation $(2)$.
        * `ips_threshold`: define the expected threhold, which determins whether to stop searching for pathways.
        * `pretrain_masks`:
            - if `True` pretrains pathway for each task
            - if `False` configures pathway during multitask learning
        * `num_pretrain_steps`: number of steps for which pathways are pretrained before multitask training.
`
