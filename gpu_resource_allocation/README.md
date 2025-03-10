# README

## 简介（Introduction）
此脚本用于对 GPU 上的多个任务进行调度规划，采用了**分段线性化**（Piecewise Linear）方法来处理不同 SM(TPC) 数量对运行时间的影响，并加入了任务依赖约束与不并行执行约束，实现**最小化总完工时间（Makespan）**的目标。

This script is designed for scheduling multiple tasks on a GPU. It uses **piecewise linear** modeling to capture how the runtime depends on the number of allocated SMs (TPCs), and it also incorporates **task dependency** constraints and **non-overlapping** constraints. The goal is to **minimize the overall makespan**.

---

## 使用步骤（Usage Instructions）

1. **从 `hcpc_priority_scheduler.py` 获取数据**  
   - 获取 GPU 任务列表 `tasks`、最大并行分区数 `partitions` 以及 GPU 任务依赖关系 `gpu_deps`。  
   - 在本脚本中，将这些数据分别填入对应的变量，如示例所示：
     ```python
     tasks = [2, 5, 7]        # 从 hcpc_priority_scheduler.py 生成的 GPU tasks 集合
     partitions = [1, 2]      # GPU 最大并行分区
     gpu_deps = {
         7: {5}               # 依赖关系链表
     }
     ```
   - **English**: Retrieve the GPU task list (`tasks`), the maximum parallel partitions (`partitions`), and the GPU task dependencies (`gpu_deps`) from `hcpc_priority_scheduler.py`. Then, place these datasets into the corresponding variables in this script, as shown in the example above.

2. **填入 GPU 任务的 Profiling 数据**  
   - 在本脚本的 `f_data` 中输入任务在不同 SM 数量下的运行时间（即 Profiling 结果），形如 `(SM数量, 运行时间)` 的元组列表：
     ```python
     f_data = {
         2: [(1, 663), (2, 390), ..., (40, 38)],
         5: [(1, 663), (2, 390), ..., (40, 38)],
         7: [(1, 663), (2, 390), ..., (40, 38)]
     }
     ```
   - 这里可根据每个任务在 1~40 个 SM(TPC) 时的实际测试或测算值进行填充。  
   - 若某些 GPU 任务需要指定初始时间，可在 `initial_time` 中进行设定；如果不需要或没有限制，则可统一设为 0：
     ```python
     initial_time = {
         2: 0,
         5: 0,
         7: 0
     }
     ```
   - **English**: Insert profiling data for each GPU task, detailing how long it takes to run on different numbers of SMs. Each entry in `f_data` is a list of `(SM_count, runtime)` tuples. If certain GPU tasks require a specific earliest start time, configure them in `initial_time`; otherwise, you can set them to 0 by default.

3. **设置 `M` 为在满 SM 下（如 40 SM）的串行运行总时间**  
   - 首先，根据 `f_data` 中的 `(40, runtime)` 信息，为每个任务在 **SM=40** 的情况累加出其运行时间总和，得到 `sum_serial_40`。  
   - 在脚本中，将 `M` 的上界（`upBound`）设定为 `sum_serial_40`；在最小化过程中，若实际分区调度得到的运行时间超过这个值，则认为**分区失败**或调度不可行。  
   - **English**: Determine the total runtime when **all tasks are executed sequentially on 40 SMs**. Sum up the `(40, runtime)` values from `f_data` for all tasks to get `sum_serial_40`, and then use this as the upper bound for `M`. If the schedule produced by partitioning ends up requiring more time than `sum_serial_40`, consider that partitioning to have **failed** or be infeasible.

---

## 运行与输出（Running and Output）

1. **运行脚本（Run the Script）**  
   在终端执行（假设脚本名为 `gpu_partition_scheduling.py`）：
   ```bash
   python gpu_partition_scheduling.py
   ```
   若使用其他求解器，可在代码中自行配置 `pulp` Solver（如 CBC、GLPK、CPLEX 等）。

2. **查看结果（Check Results）**  
   - 控制台会输出求解状态（`Optimal`, `Infeasible` 等）以及各个任务的分区分配、开始/结束时间、所分配的 SM 数量等。  
   - 如果求解成功并找到可行解，脚本会打印出总完工时间（Makespan）。若任务调度的最优结果超出 `sum_serial_40`，则可视为无法在此限制下完成分区调度。

**English**:  
1. **To run**: execute the script in a terminal (e.g., `python gpu_partition_scheduling.py`).  
2. **Output**: The console will show the solver status (`Optimal`, `Infeasible`, etc.), the partition assignment for each task, their start/end times, allocated SM counts, and the final makespan. If the best schedule exceeds `sum_serial_40`, treat that as a failure to partition within the specified limit.

---

## 关键点总结（Key Points Summary）

1. **从优先级调度器脚本（`hcpc_priority_scheduler.py`）获取任务列表、最大分区数和依赖信息**。  
2. **将 Profiling 数据填入 `f_data`，设置初始时间可选**。  
3. **`M` 的值或上限设为所有任务在满 SM 下串行运行的总时间**；若模型求解后运行时间超过此值，则说明分区方案不可行。

**English**:  
1. **Obtain tasks, maximum partitions, and dependencies** from `hcpc_priority_scheduler.py`.  
2. **Insert profiling data** into `f_data` and optionally set initial start times.  
3. **Set `M` to the sum of serialized runtime on maximum SM**. If the resulting schedule requires more time than `M`, treat the partitioning as infeasible.

