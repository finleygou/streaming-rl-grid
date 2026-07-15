# stream-rl-grid

一个面向持续学习实验的 Windy Grid World：每个转移只使用一次、没有 replay buffer、没有 batch、没有 episode 终止。智能体采用：

**Streaming Differential Sarsa(λ) + replacing traces + 双组 tile coding + TIDBD**。

项目参考了：

- `Streaming Deep Reinforcement Learning Finally Works` 中的逐样本即时更新与资格迹思想；
- `TIDBD: Adapting Temporal-difference Step-sizes Through Stochastic Meta-descent` 的逐特征步长更新；
- RLSS Lecture 02 的线性函数逼近和 tile coding；
- RLSS Lecture 03 的 continuing average-reward / differential TD 更新。

## 算法

动作值函数是线性的：

$$Q(s, a) = w^T x(s, a)$$


差分 Sarsa TD error 不包含折扣因子：

$$delta = reward - R_bar + Q(next_state, next_action) - Q(state, action)$$

平均奖励估计为：

$$ R_bar <- R_bar + eta * delta $$

资格迹采用 replacing traces：

$$
z <- lambda * z
z[active_features] <- 1
$$

TIDBD 为每个权重维护 $`$beta_i = log(alpha_i)$`$ 和元迹 $H_i$：

$$
beta_i <- beta_i + theta * delta * x_i * H_i
alpha_i <- exp(beta_i)
w_i <- w_i + alpha_i * delta * z_i
H_i <- H_i * max(0, 1 - alpha_i * x_i * z_i) + alpha_i * delta * z_i
$$

实现不叠加 ObGD，以免改变 TIDBD 实验含义；只设置宽松的 `beta` 数值边界并检测 NaN/Inf。

## 状态与函数逼近

智能体可观察：

$$
(current_x, current_y, goal_x, goal_y, previous_action)
$$

它看不到风阶段、奖励阶段、地图模式编号或全局时钟。

双组 tile coding 分别编码：

1. 绝对位置 `(x, y, previous_action, candidate_action)`；
2. 相对目标位置 `(goal_x-x, goal_y-y, previous_action, candidate_action)`。

另有一个 categorical bias feature。默认每组 8 个 tilings，因此正常情况下每次有 17 个激活特征。

## Continuing 环境规则

- 动作包括上、右、下、左、原地停留；
- `stay` 仍然受到风的影响；
- 主动动作和风的位移逐格执行；
- 任一步碰到边界或障碍物，整个转移取消，智能体留在动作前的位置并得到碰撞惩罚；
- 到达目标后得到目标奖励，并立即随机传送到合法非目标格；
- 环境始终返回 `terminated=False, truncated=False`；
- 到达目标、风季节变化、目标移动和地图切换均不清空资格迹；
- 地图切换时，如果智能体当前格在新地图中是障碍物，该障碍物暂不激活；智能体离开后立即激活。

## Structured non-stationarity

图形面板提供五种配置：

- `stationary`：固定风、目标、奖励和地图；
- `seasonal_wind`：风向与奖励倍率按固定周期循环；
- `moving_goal`：目标沿固定蛇形轨迹缓慢往返，遇到障碍物轨迹点则跳过；
- `hidden_context`：障碍物地图按周期切换，但模式编号不提供给智能体；
- `combined`：同时启用上述三类变化。

地图生成器保证所有合法格连通。面板中可以先点击一个障碍物，再点击一个空格来移动障碍物；破坏连通性的修改会被拒绝。

## 启动图形面板

Python 需要带 Tk 支持。安装依赖后，在仓库根目录执行：

```powershell
python run_gui.py
```

面板支持：

- 环境、奖励、调度周期和算法超参数设置；
- 地图生成、上下文地图预览和障碍物手动移动；
- 开始、暂停、继续、手动保存、停止并保存；
- checkpoint 加载并精确续训；
- 网格、平均奖励、目标到达率、碰撞率、TD error 和 TIDBD 步长实时显示。

训练在后台线程中执行，GUI 不参与智能体观测。

## 无图形界面运行

运行固定步数：

```powershell
python -m stream_rl_grid.cli --profile combined --steps 50000
```

无限运行，人工按 `Ctrl+C` 停止并自动保存：

```powershell
python -m stream_rl_grid.cli --profile combined --steps 0
```

精确续训：

```powershell
python -m stream_rl_grid.cli --resume checkpoints/<run-id>/step-000000050000.pkl --steps 0
```

固定步长基线：

```powershell
python -m stream_rl_grid.cli --profile stationary --fixed-alpha --steps 50000
```

## 多随机种子验证

比较 TIDBD 与固定步长 Differential Sarsa：

```powershell
python -m stream_rl_grid.benchmark --steps 50000 --seeds 0 1 2 3 4
```

输出包括逐运行 CSV、均值学习曲线和 95% 正态近似置信区间。主要指标是滑动窗口平均奖励，不使用 episode return。

## Checkpoint 内容

checkpoint 不只保存 `w`，还保存：

- `w, beta, H, z, R_bar`；
- 当前观测和已经选好的下一动作；
- 环境位置、目标、地图、风/奖励/地图调度相位；
- 延迟激活障碍物；
- IHT 字典与碰撞计数；
- Python 和 NumPy 随机数状态；
- 滑动指标、曲线、配置和格式版本。

保存采用临时文件 + 原子替换，避免中断时留下半个 checkpoint。

## 测试

```powershell
python -m unittest discover -s tests -v
```

## Algorithm and environment configuration

Algorithms share the interface in `stream_rl_grid/algo/base.py` and are selected with
`AgentConfig.algorithm` (`"tidbd"` or `"sarsa"`). The GUI exposes the same selector.
The policy shown by the GUI is a frozen `(height, width, 5)` epsilon-greedy probability
matrix built from the current learned parameters; it is separate from the action sampled
by the online behavior loop.

Default maps can be authored directly in `EnvironmentConfig` with
`obstacle_coordinates`, `start_position`, and `goal_position`. Wind is probabilistic:
`w_strength=0.3` means a 30% chance of one additional cell of displacement in the selected
wind direction on each transition.

测试覆盖 continuing 目标传送、碰撞回退、`stay` 的风效应、地图切换延迟障碍物、TIDBD 数值有限性，以及 checkpoint 后逐状态/逐动作/逐权重的精确续训。
