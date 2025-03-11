# README

## 简介（Introduction）
此脚本用于对 GPU 上的多个任务进行调度规划，采用了**分段线性化**（Piecewise Linear）方法来处理不同 SM(TPC) 数量对运行时间的影响，并加入了任务依赖约束与不并行执行约束，实现**最小化总完工时间（Makespan）**的目标。

> **English**:  
> This script is designed for scheduling multiple tasks on a GPU. It uses **piecewise linear** modeling to capture how the runtime depends on the number of allocated SMs (TPCs), and it also incorporates **task dependency** constraints and **non-overlapping** constraints. The goal is to **minimize the overall makespan**.

---

## 使用步骤（Usage Instructions）

1. **从 `hcpc_priority_scheduler.py` 获取数据**  
   - 获取 GPU 任务列表 `tasks`、最大并行分区数 `partitions`、可能需要指定初始时间的任务 `initial_time`，以及 GPU 任务之间的依赖 `gpu_deps`。  
   - 在本脚本中，将这些数据填入对应的变量（如下示例）：
     ```python
     # 从 hcpc_priority_scheduler.py 生成的 GPU tasks 集合
     tasks = [2, 5, 7]        

     # GPU 最大并行分区（这里为 2）
     partitions = [1, 2]      

     # 某些 GPU 任务需要指定初始时间，可在 initial_time 中进行设定
     initial_time = {
         2: 0,
         5: 0
     }

     # 依赖关系链表：例如 7 依赖于 5
     gpu_deps = {
         7: {5: {'cpu_path_length': 0, 'cpu_nodes': []}}
     }
     ```
   - **English**:  
     Retrieve the GPU task list (`tasks`), the maximum parallel partitions (`partitions`), and the GPU task dependencies (`gpu_deps`) from `hcpc_priority_scheduler.py`. Then place these datasets into the corresponding variables in this script, as shown in the example above.

2. **填入 GPU 任务的 Profiling 数据**  
   - 在本脚本的 `f_data` 中，为每个任务提供在不同 SM 数量下的运行时间（Profiling 结果），形如 `(SM数量, 运行时间)` 的元组列表：
     ```python
     f_data = {
         2: [(1, 663), (2, 390), ..., (40, 38)],
         5: [(1, 663), (2, 390), ..., (40, 38)],
         7: [(1, 663), (2, 390), ..., (40, 38)]
     }
     ```
   - 根据实际测得或分析得到的任务在 1~40 个 SM(TPC) 时的运行时间进行填写。  
   - **English**:  
     Insert profiling data for each GPU task, detailing how long it takes to run with different SM counts. Each entry in `f_data` is a list of `(SM_count, runtime)` tuples.

3. **`M` 的设置**  
   - 脚本中使用了一个名为 `M` 的变量代表**总完工时间（Makespan）**，由求解器自动决定其最优值。  
   - 由于不再显式设置上界，求解器会从 0 开始，将 `M` 视作可行解约束下的最小值来进行求解。  
   - **English**:  
     A variable named `M` is used to represent the **makespan**. The solver automatically determines its optimal value without requiring you to set an explicit upper bound.

---

## 运行与输出（Running and Output）

1. **运行脚本**  
   在终端执行（假设脚本文件名为 `gpu_partition_scheduling.py`）：
   ```bash
   python gpu_partition_scheduling.py
   ```
   若需要使用其他求解器，可在代码中自行配置 `pulp` 的 Solver（如 CBC、GLPK、CPLEX 等）。

2. **查看结果**  
   - 控制台会输出求解状态（`Optimal`, `Infeasible` 等）以及各个任务的分区分配、开始/结束时间、以及所分配的 SM 数量等。  
   - 如果求解成功并找到可行解，脚本会打印出总完工时间（Makespan）。  
   - **English**:  
     Run the script with `python gpu_partition_scheduling.py`. The console will show the solver status (`Optimal`, `Infeasible`, etc.), each task’s partition assignment, start/end times, allocated SMs, and the final makespan if a feasible solution is found.

---

## 关键点总结（Key Points Summary）

1. **从优先级调度脚本 (`hcpc_priority_scheduler.py`) 中获取任务列表与依赖信息**。  
2. **根据不同 SM 数量的 Profiling 数据**，将对应运行时间填入 `f_data`。  
3. **`M` 为脚本内的 makespan 变量**，无须手动设置上界；求解器会通过最小化 `M` 来确定调度方案。  

> **English**:  
> 1. **Obtain tasks and dependencies** from `hcpc_priority_scheduler.py`.  
> 2. **Insert runtime profiling data** for each task in `f_data`.  
> 3. **`M` is the makespan variable**; you do not need to set an upper bound. The solver will minimize `M` to find the optimal schedule.

---

## 示例输出（Example Output）

以下为脚本在某个示例下的典型输出，展示了求解器的状态、迭代过程概览，以及最终每个任务的分区分配、起止时间和分配 SM 数量。在此示例中，makespan 最终求得为 76.0。

```
Result - Optimal solution found

Objective value:                76.00000000
Enumerated nodes:               16
Total iterations:               4703
Time (CPU seconds):             1.34
Time (Wallclock seconds):       1.40

Option for printingOptions changed from normal to all
Total time (CPU seconds):       1.34   (Wallclock seconds):       1.41

Status: Optimal
Optimal Solution Found!
------------------------------------
Partitions SM allocation (x[p]):
  x[1] = 20.0
  x[2] = 20.0
------------------------------------
Task assignments and timing:
Task 2 -> partition 1
  S[2] = 0.0
  E[2] = 38.0
  F[2,1] = 38.0
Task 5 -> partition 2
  S[5] = 0.0
  E[5] = 38.0
  F[5,2] = 38.0
Task 7 -> partition 1
  S[7] = 38.0
  E[7] = 76.0
  F[7,1] = 38.0
------------------------------------
Makespan = 76.0
```

该结果表明：
- 有两个并行分区，分别分配了 20 个 SM。  
- 任务 2 和 5 并行执行，各自使用 20 SM，执行时间均为 38，结束于时间 38。  
- 任务 7 依赖任务 5，且被分配到分区 1，于时间 38 开始，76 结束。  
- 整体调度完成时间（makespan）即 76。

> **English**:  
> The above example shows typical solver output, including the solver status, number of nodes enumerated, and final schedule details. In this scenario, each partition is allocated 20 SMs, tasks 2 and 5 run in parallel up to 38, then task 7 follows task 5 on partition 1 until 76, yielding a makespan of 76.

