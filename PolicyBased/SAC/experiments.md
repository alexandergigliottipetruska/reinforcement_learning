Experiments were run on several MuJoCo environments, namely HalfCheetah-v5, Ant-v5, and Humanoid-v5, to demonstrate SAC's ability to tackle high dimensional continuous control tasks.

General hyperparemeters (kept constant accross experiments) are:
- Buffer size: 1,000,000
- Discount factor $\gamma$: 0.99
- Smoothing coefficient $\tau$: 0.005
- Learning rate: 3e-4 (critic learning rate changed in Humanoid)
- Batch size: 1024 (exception is HalfCheetah)
- A warmup period of 10,000, uniform sampling improved performance according to OpenAI Spinning Up

For each experiment, some hyperparameters were trained, and so was the training duration: For example,
- Humanoid used a fixed alpha (0.05) and reward scale of 20.0, which was found to significantly improve performance over autotuning, and a critic learning rate of 1e-3.
- Ant was autotuned.
- HalfCheetah was autotuned.

At 1 million steps, Humanoid got a highest average rolling return over 100 episodes of 5202.24. HalfCheetah got 8103.09 at 1.1 million, and Ant got 4834.88 at 1.25 million.

These were expected results, and with more tuning and training length, would approach the performance of the algorithm in actual papers. 
