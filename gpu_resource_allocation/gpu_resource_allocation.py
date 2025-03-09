import pulp

# 定义任务、分区和SM数量
tasks = [2, 5, 7]
partitions = [1, 2]  # 启发式（根据DAG计算出的最大并行分区数量）
N_SM = 40  # SM (TPC) 总数

# 分段线性化数据（SM数量和GPU任务的运行时间的函数，比如任务2在SM（TPC）为1时，运行时间为663ms）
f_data = {
    2: [(1, 663), (2, 390), (3, 227), (4, 163), (5, 130), (6, 114), (7, 98), (8, 83), (9, 83), (10, 67), (11, 67),
        (12, 67), (13, 67), (14, 55), (15, 55), (16, 55), (17, 55), (18, 55), (19, 55), (20, 38), (40, 38)],
    5: [(1, 663), (2, 390), (3, 227), (4, 163), (5, 130), (6, 114), (7, 98), (8, 83), (9, 83), (10, 67), (11, 67),
        (12, 67), (13, 67), (14, 55), (15, 55), (16, 55), (17, 55), (18, 55), (19, 55), (20, 38), (40, 38)],
    7: [(1, 663), (2, 390), (3, 227), (4, 163), (5, 130), (6, 114), (7, 98), (8, 83), (9, 83), (10, 67), (11, 67),
        (12, 67), (13, 67), (14, 55), (15, 55), (16, 55), (17, 55), (18, 55), (19, 55), (20, 38), (40, 38)]
}

# 定义初始时间（任务2和任务5已知）
initial_time = {
    2: 24,   # 任务2的初始时间为0
    5: 64,   # 任务5的初始时间为10
    7: 104
}

# 创建问题
problem = pulp.LpProblem("GPU_Partition_Scheduling_with_Piecewise_and_Parallel", pulp.LpMinimize)

# 定义变量
y = pulp.LpVariable.dicts("y", (tasks, partitions), cat=pulp.LpBinary)  # 任务分配到分区
x = pulp.LpVariable.dicts("x", partitions, lowBound=0, upBound=N_SM, cat=pulp.LpInteger)  # 分区分配的SM数量
S = pulp.LpVariable.dicts("S", tasks, lowBound=0)  # 任务的开始时间
E = pulp.LpVariable.dicts("E", tasks, lowBound=0)  # 任务的结束时间
M = pulp.LpVariable("M", lowBound=0)  # 最大完成时间（Makespan）

# 约束：总SM数量不能超过N_SM
problem += pulp.lpSum([x[p] for p in partitions]) <= N_SM

# 约束：每个任务必须分配到一个分区
for i in tasks:
    problem += pulp.lpSum([y[i][p] for p in partitions]) == 1

# 约束：任务7依赖任务5
problem += S[7] >= E[5], "Dep_5_7"

# 定义F[i,p]表示执行时间（通过分段线性化）
F = pulp.LpVariable.dicts("F", (tasks, partitions), lowBound=0)
bigM = 10000

for i in tasks:
    segments = f_data[i]
    num_segments = len(segments) - 1
    # 为区间定义二元变量 u[i,p,k]
    u = {p: pulp.LpVariable.dicts(f"u_{i}_{p}", range(num_segments), cat=pulp.LpBinary) for p in partitions}
    for p in partitions:
        problem += pulp.lpSum([u[p][k] for k in range(num_segments)]) == y[i][p]

        for k in range(num_segments):
            x_left, f_left = segments[k]
            x_right, f_right = segments[k + 1]
            slope = (f_right - f_left) / (x_right - x_left)

            # 区间限制
            problem += x[p] >= x_left - bigM * (1 - u[p][k])
            problem += x[p] <= x_right + bigM * (1 - u[p][k])

            problem += F[i][p] >= f_left + slope * (x[p] - x_left) - bigM * (1 - u[p][k])
            problem += F[i][p] <= f_left + slope * (x[p] - x_left) + bigM * (1 - u[p][k])

# 约束：E[i] = S[i] + sum_p(F[i,p])
for i in tasks:
    problem += E[i] == S[i] + pulp.lpSum(F[i][p] for p in partitions)
    for p in partitions:
        problem += E[i] - S[i] - F[i][p] <= bigM * (1 - y[i][p])
        problem += E[i] - S[i] - F[i][p] >= -bigM * (1 - y[i][p])

# 约束：任务2和任务5的初始时间
problem += S[2] >= initial_time[2], "Initial_Time_2"
problem += S[5] >= initial_time[5], "Initial_Time_5"

# 约束：任务7的初始时间（动态计算）
# 任务7的初始时间 = 任务5的结束时间
problem += S[7] >= E[5], "Initial_Time_7"

# 约束：Makespan
for i in tasks:
    problem += E[i] <= M

# 不并行执行约束
z = pulp.LpVariable.dicts("z", (tasks, tasks, partitions), cat=pulp.LpBinary)
w = pulp.LpVariable.dicts("w", (tasks, tasks, partitions), cat=pulp.LpBinary)

for p in partitions:
    for i in tasks:
        for j in tasks:
            if i < j:
                # w[i,j,p] 等价于 y[i,p]*y[j,p]
                problem += w[i][j][p] <= y[i][p]
                problem += w[i][j][p] <= y[j][p]
                problem += w[i][j][p] >= y[i][p] + y[j][p] - 1

                problem += z[i][j][p] + z[j][i][p] == w[i][j][p]

                # 不重叠约束
                problem += S[j] >= E[i] - bigM * (1 - z[i][j][p])
                problem += S[i] >= E[j] - bigM * (1 - z[j][i][p])

# 目标函数：最小化Makespan
problem.setObjective(M)

# 求解
solver = pulp.PULP_CBC_CMD(msg=1)
problem.solve(solver)

# 输出结果
print("Status:", pulp.LpStatus[problem.status])
for i in tasks:
    assigned_p = [pp for pp in partitions if pulp.value(y[i][pp]) == 1]
    if assigned_p:
        pstar = assigned_p[0]
        print(f"Task {i} assigned to partition {pstar}, x_{pstar}={pulp.value(x[pstar])}")
        print(f"F[{i},{pstar}] = {pulp.value(F[i][pstar])}")
        print(f"S[{i}] = {pulp.value(S[i])}, E[{i}] = {pulp.value(E[i])}")
    else:
        print(f"Task {i} not assigned to any partition!")
print("Makespan =", pulp.value(M))




