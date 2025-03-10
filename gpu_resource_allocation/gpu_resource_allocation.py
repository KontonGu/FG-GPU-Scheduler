import pulp

# ========================
# (1) 定义任务、分区、SM
# Define tasks, partitions, and total SM (TPC)
# ========================
tasks = [2, 5, 7]   # 需要调度的 GPU 任务编号 / GPU task IDs to schedule
partitions = [1, 2] # 可用的并行分区 / Parallel partitions available
N_SM = 40           # SM (TPC) 总数量 / Total number of SM (TPC)

# ========================
# (2) 分段线性化数据
# Piecewise linear data: (x_val, f_val) means "runtime f_val if x_val SMs allocated"
# ========================
f_data = {
    2: [(1, 663), (2, 390), (3, 227), (4, 163), (5, 130), (6, 114), (7, 98),  (8, 83),
        (9, 83),   (10, 67), (11, 67), (12, 67), (13, 67), (14, 55), (15, 55), (16, 55),
        (17, 55),  (18, 55), (19, 55), (20, 38), (40, 38)],
    5: [(1, 663), (2, 390), (3, 227), (4, 163), (5, 130), (6, 114), (7, 98),  (8, 83),
        (9, 83),   (10, 67), (11, 67), (12, 67), (13, 67), (14, 55), (15, 55), (16, 55),
        (17, 55),  (18, 55), (19, 55), (20, 38), (40, 38)],
    7: [(1, 663), (2, 390), (3, 227), (4, 163), (5, 130), (6, 114), (7, 98),  (8, 83),
        (9, 83),   (10, 67), (11, 67), (12, 67), (13, 67), (14, 55), (15, 55), (16, 55),
        (17, 55),  (18, 55), (19, 55), (20, 38), (40, 38)]
}

# ========================
# (3) 定义初始时间
# Define initial (earliest) start times for some tasks
# ========================
initial_time = {
    2: 0,   # 任务2最早开始时间 / Earliest start time for task 2
    5: 0,   # 任务5最早开始时间 / Earliest start time for task 5
}

# ========================
# (4) 依赖关系（示例：任务7依赖任务5）
# Dependency dictionary: e.g., task 7 depends on task 5
# ========================
gpu_deps = {
    7: {5}
}

# ========================
# (5) 创建线性规划问题
# Create the LP problem, objective is to minimize M (makespan)
# ========================
problem = pulp.LpProblem("GPU_Partition_Scheduling_with_Piecewise_and_Parallel", pulp.LpMinimize)

# ========================
# (6) 计算串行总时长 sum_serial_40 并定义 M
# Compute the total serialized time at x=40, use it as upper bound for M
# ========================
sum_serial_40 = 0
for i in tasks:
    # 在 f_data[i] 中查找 x=40 时的执行时间 / Find the runtime when x=40 in f_data[i]
    for (x_val, f_val) in f_data[i]:
        if x_val == 40:
            sum_serial_40 += f_val
            break

# 定义 M 为 [0, sum_serial_40] 之间的变量 / M in [0, sum_serial_40]
M = pulp.LpVariable("M", lowBound=0, upBound=sum_serial_40)

# ========================
# (7) 定义其它变量
# Define additional variables
# ========================
# y[i,p]：二元变量，表示任务i是否分配到分区p / Binary var: does task i go to partition p
y = pulp.LpVariable.dicts("y", (tasks, partitions), cat=pulp.LpBinary)

# x[p]：分区p分配的SM数量 / Number of SM assigned to partition p
x = pulp.LpVariable.dicts("x", partitions, lowBound=0, upBound=N_SM, cat=pulp.LpInteger)

# S[i]：任务i的开始时间 / Start time of task i
S = pulp.LpVariable.dicts("S", tasks, lowBound=0)

# E[i]：任务i的结束时间 / End time of task i
E = pulp.LpVariable.dicts("E", tasks, lowBound=0)

# F[i,p]：分段线性化后的执行时间 / Piecewise-linear runtime if i assigned to p
F = pulp.LpVariable.dicts("F", (tasks, partitions), lowBound=0)

# bigM：大常数，用于松弛 / A large constant for big-M constraints
bigM = 10000

# ========================
# (8) 约束：分区 SM 总和不超过 N_SM
# The sum of SMs allocated to all partitions must not exceed N_SM
# ========================
problem += pulp.lpSum(x[p] for p in partitions) <= N_SM, "SM_Capacity"

# ========================
# (9) 约束：每个任务必须且只能分配到一个分区
# Each task must be assigned exactly one partition
# ========================
for i in tasks:
    problem += pulp.lpSum(y[i][p] for p in partitions) == 1, f"OnePartition_{i}"

# ========================
# (10) 任务依赖约束：S[job] >= E[dep_job]
# Dependency: if job depends on dep_job, job cannot start before dep_job finishes
# ========================
for job, deps in gpu_deps.items():
    for dep_job in deps:
        problem += S[job] >= E[dep_job], f"Dep_{dep_job}_{job}"

# ========================
# (11) 约束：初始时间
# If a task has an earliest start time, S[i] must be >= that time
# ========================
for i in tasks:
    if i in initial_time:
        problem += S[i] >= initial_time[i], f"InitialTime_{i}"

# ========================
# (12) 分段线性化：F[i,p] 表示任务 i 分配到分区 p 时的执行时间
# Piecewise linear constraints for F[i,p], based on x[p]
# ========================
for i in tasks:
    segments = f_data[i]
    num_segments = len(segments) - 1

    # u[i,p,k]：表示使用分段k / Binary var indicating which segment k is used
    u = {p: pulp.LpVariable.dicts(f"u_{i}_{p}", range(num_segments), cat=pulp.LpBinary)
         for p in partitions}

    for p in partitions:
        # 只能选其中一段（如果任务 i 分配到 p） / Exactly one segment if assigned
        problem += pulp.lpSum(u[p][k] for k in range(num_segments)) == y[i][p], f"SegmentSum_{i}_{p}"

        for k in range(num_segments):
            x_left, f_left = segments[k]
            x_right, f_right = segments[k+1]
            slope = (f_right - f_left) / (x_right - x_left)

            # 区间限制：当 u[p][k] = 1 时，x[p] 应落在 [x_left, x_right]
            # If u[p][k] = 1, x[p] must be within [x_left, x_right]
            problem += x[p] >= x_left - bigM * (1 - u[p][k])
            problem += x[p] <= x_right + bigM * (1 - u[p][k])

            # F[i][p] 的线性逼近 / Force F[i][p] to match the piecewise slope
            expr_segment = f_left + slope * (x[p] - x_left)
            problem += F[i][p] >= expr_segment - bigM * (1 - u[p][k])
            problem += F[i][p] <= expr_segment + bigM * (1 - u[p][k])

# ========================
# (13) 约束：E[i] = S[i] + ∑ F[i,p] (当 y[i,p] = 1 时生效)
# E[i] = S[i] + sum(F[i,p]) for the chosen partition(s)
# ========================
for i in tasks:
    # E[i] 等于开始时间加上所有分区的执行时间之和 / E[i] = S[i] + sum of F[i][p]
    problem += E[i] == S[i] + pulp.lpSum(F[i][p] for p in partitions), f"EndTime_{i}"

    # 若 y[i,p] = 0，则 F[i][p] 不影响 E[i]（通过 bigM 松弛） / If y[i,p] = 0, we ignore F[i][p] for E[i]
    for p in partitions:
        problem += (E[i] - S[i] - F[i][p]) <= bigM * (1 - y[i][p])
        problem += (E[i] - S[i] - F[i][p]) >= -bigM * (1 - y[i][p])

# ========================
# (14) 不并行执行约束：同一分区上的任务不能重叠
# Non-overlapping constraint: tasks on the same partition must not overlap
# ========================
z = pulp.LpVariable.dicts("z", (tasks, tasks, partitions), cat=pulp.LpBinary)
w = pulp.LpVariable.dicts("w", (tasks, tasks, partitions), cat=pulp.LpBinary)

for p in partitions:
    for i in tasks:
        for j in tasks:
            if i < j:
                # w[i,j,p] = y[i,p] * y[j,p] / Both tasks i and j assigned to p
                problem += w[i][j][p] <= y[i][p]
                problem += w[i][j][p] <= y[j][p]
                problem += w[i][j][p] >= (y[i][p] + y[j][p] - 1)

                # z[i][j][p] + z[j][i][p] = w[i][j][p]
                # If both tasks share partition p, then exactly one of them is first
                problem += z[i][j][p] + z[j][i][p] == w[i][j][p]

                # z[i][j][p] = 1 => S[j] >= E[i] / If i is before j, S[j]>=E[i]
                problem += S[j] >= E[i] - bigM * (1 - z[i][j][p])
                problem += S[i] >= E[j] - bigM * (1 - z[j][i][p])

# ========================
# (15) Makespan 约束：E[i] <= M
# All tasks must finish by time M
# ========================
for i in tasks:
    problem += E[i] <= M, f"Makespan_{i}"

# ========================
# (16) 目标：最小化 M
# Minimize M
# ========================
problem.setObjective(M)

# ========================
# (17) 求解
# Solve the LP problem using CBC solver
# ========================
solver = pulp.PULP_CBC_CMD(msg=1)
problem.solve(solver)

# ========================
# (18) 输出结果
# Print solution results
# ========================
print("Status:", pulp.LpStatus[problem.status])
if problem.status == 1:
    print("Optimal Solution Found!")
else:
    print("Solver did not find an optimal solution.")

print("------------------------------------")
print("Partitions SM allocation (x[p]):")
for p in partitions:
    print(f"  x[{p}] = {pulp.value(x[p])}")

print("------------------------------------")
print("Task assignments and timing:")
for i in tasks:
    # 查找被分配到哪个分区 / Find the partition assigned for task i
    assigned_p = [pp for pp in partitions if pulp.value(y[i][pp]) > 0.5]
    if assigned_p:
        pstar = assigned_p[0]  # 只会有一个分区 / Only one partition is chosen
        print(f"Task {i} -> partition {pstar}")
        print(f"  S[{i}] = {pulp.value(S[i])}")
        print(f"  E[{i}] = {pulp.value(E[i])}")
        print(f"  F[{i},{pstar}] = {pulp.value(F[i][pstar])}")
    else:
        print(f"Task {i} was not assigned to any partition!")

print("------------------------------------")
print(f"sum_serial_40 (作为上界 / used as upper bound) = {sum_serial_40}")
print("Makespan =", pulp.value(M))

