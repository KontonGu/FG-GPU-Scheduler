import random
from collections import defaultdict, deque

##############################
# 1) 图构建、拓扑排序
##############################

def build_graph(edges):
    """
    根据边列表构建正向图、入度字典和反向图（前驱关系），同时收集所有节点。
    edges 格式: (u, v, {})。
    """
    graph = defaultdict(list)
    in_degree = defaultdict(int)
    reverse_graph = defaultdict(list)
    nodes = set()
    for u, v, _ in edges:
        graph[u].append(v)
        in_degree[v] += 1
        # 确保 u 也在 in_degree 中
        if u not in in_degree:
            in_degree[u] = 0
        reverse_graph[v].append(u)
        nodes.add(u)
        nodes.add(v)
    return dict(graph), dict(in_degree), dict(reverse_graph), nodes

def topological_sort(graph, in_degree, nodes):
    """
    利用Kahn算法进行拓扑排序
    """
    queue = deque([n for n in nodes if in_degree[n] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in graph.get(node, []):
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)
    return topo_order

##############################
# 2) 关键路径法 (CPM)
##############################

def find_critical_path(edges, node_weight):
    """
    利用关键路径法 (CPM) 计算项目总工期，并提取一条关键路径。

    特别地，还返回:
      - EST[node] (最早开始时间)
      - EFT[node] (最早完成时间)
      - topo_order (全局拓扑序)
    
    返回：
      critical_path: 按拓扑顺序的一条关键路径（仅包含零余量节点）
      finish_time:   项目总工期
      graph, reverse_graph: 图结构
      nodes:         所有节点集合
      EST, EFT:      各节点的最早开始/完成时间
      topo_order:    整体图的拓扑序
    """
    # 1) 构建图 & 拓扑排序
    graph, in_degree, reverse_graph, nodes = build_graph(edges)
    topo_order = topological_sort(graph, in_degree.copy(), nodes)
    
    # 2) 正向遍历 => EST, EFT
    EST = {n: 0 for n in nodes}
    EFT = {n: 0 for n in nodes}
    for node in topo_order:
        EFT[node] = EST[node] + node_weight.get(node, 1)
        for succ in graph.get(node, []):
            EST[succ] = max(EST[succ], EFT[node])
    
    # 3) finish_time: 所有 sink_nodes 的 EFT 最大值
    sink_nodes = [n for n in nodes if n not in graph or not graph[n]]
    finish_time = max(EFT[n] for n in sink_nodes)

    # 4) 反向遍历 => LFT, LST
    LFT = {n: finish_time for n in nodes}
    LST = {n: finish_time for n in nodes}
    for node in reversed(topo_order):
        if node in graph and graph[node]:
            LFT[node] = min(LST[succ] for succ in graph[node])
        LST[node] = LFT[node] - node_weight.get(node, 1)
    
    # 5) 找出零余量节点 (critical_nodes)
    critical_nodes = [n for n in topo_order if EST[n] == LST[n]]
    
    # 6) 从起始节点起，拼接一条连续关键路径
    cp = []
    start_nodes = [n for n in topo_order if EST[n] == 0 and n in critical_nodes]
    if not start_nodes:
        return [], finish_time, graph, reverse_graph, nodes, EST, EFT, topo_order
    
    current = start_nodes[0]
    cp.append(current)
    while True:
        next_candidates = [
            s for s in graph.get(current, [])
            if s in critical_nodes and EST[s] == EFT[current]
        ]
        if not next_candidates:
            break
        current = next_candidates[0]
        cp.append(current)

    return cp, finish_time, graph, reverse_graph, nodes, EST, EFT, topo_order

##############################
# 3) 获取节点祖先/后继
##############################

def get_ancestors_with_reverse(reverse_graph, node):
    """
    返回 node 的所有祖先 (不含 node 自身)
    """
    ancestors = set()
    stack = [node]
    while stack:
        cur = stack.pop()
        for pred in reverse_graph.get(cur, []):
            if pred not in ancestors:
                ancestors.add(pred)
                stack.append(pred)
    return ancestors

def get_descendants(graph, node):
    """
    返回 node 的所有后继 (不含 node 自身)
    """
    descendants = set()
    stack = [node]
    while stack:
        cur = stack.pop()
        for succ in graph.get(cur, []):
            if succ not in descendants:
                descendants.add(succ)
                stack.append(succ)
    return descendants

##############################
# 4) Capacity Provider相关
##############################

def identify_capacity_providers(critical_path, reverse_graph):
    """
    对关键路径节点进行合并：
      若下一个节点的直接前驱集合 == {前一个节点} => 同一CP
    返回: CP列表, 每个元素是关键路径上的一组 [v1, v2, ...]
    """
    if not critical_path:
        return []
    cp_groups = []
    current_group = [critical_path[0]]
    for i in range(1, len(critical_path)):
        prev_node = critical_path[i-1]
        curr_node = critical_path[i]
        preds = set(reverse_graph.get(curr_node, []))
        if preds == {prev_node}:
            current_group.append(curr_node)
        else:
            cp_groups.append(current_group)
            current_group = [curr_node]
    if current_group:
        cp_groups.append(current_group)
    return cp_groups

def compute_F_G(cp_groups, graph, reverse_graph, all_nodes, critical_path):
    """
    计算每个 CP组的 F(CP_i) 和 G(CP_i).
    
    F(CP_i) = anc(下一个CP组首节点) ∩ NonCritical
    G(CP_i) = union_{v in F(CP_i)} [ all_nodes - (anc(v) ∪ desc(v)) ] ∩ NonCritical
    最后一个 CP => F, G = 空
    """
    F_dict = {}
    G_dict = {}
    critical_set = set(critical_path)
    non_critical = all_nodes - critical_set

    for i in range(len(cp_groups)):
        if i < len(cp_groups) - 1:
            next_rep = cp_groups[i+1][0]
            anc_next = get_ancestors_with_reverse(reverse_graph, next_rep)
            F_i = anc_next.intersection(non_critical)
            F_dict[i] = F_i
            # 从 non_critical 移除F_i, 避免后续重复
            non_critical = non_critical - F_i

            G_i = set()
            for v in F_i:
                anc_v = get_ancestors_with_reverse(reverse_graph, v)
                desc_v = get_descendants(graph, v)
                C_v = all_nodes - (anc_v.union(desc_v))
                C_v.discard(v)
                G_i |= (C_v.intersection(non_critical))
            G_dict[i] = G_i
        else:
            F_dict[i] = set()
            G_dict[i] = set()
    return F_dict, G_dict

##############################
# 5) CPU任务优先度分配
##############################

def compute_local_longest_paths(F_set, graph, weight):
    """
    在局部子图(只含F_set节点)中, 计算每个节点的最长路径长度(DFS+记忆)
    """
    memo = {}
    local_graph = {u: [v for v in graph.get(u, []) if v in F_set] for u in F_set}
    
    def dfs(u):
        if u in memo:
            return memo[u]
        cur_w = weight.get(u, 1)
        best = cur_w
        for succ in local_graph.get(u, []):
            best = max(best, cur_w + dfs(succ))
        memo[u] = best
        return best
    
    for node in F_set:
        dfs(node)
    return memo

def assign_cpu_priorities(cp_groups, F_dict, task_attr, graph, weight):
    """
    两阶段:
      1) CP组中的CPU => 按出现顺序分配优先度(初始p=100)
      2) 每个CP组的 F集合中的CPU => (局部最长路径, rank)降序
    返回 {node: priority}
    """
    priority = {}
    p = 100

    # 1) CP关键路径中的CPU
    for group in cp_groups:
        for node in group:
            if task_attr.get(node, {}).get('task_type') == 'CPU':
                priority[node] = p
                p -= 1

    # 2) CP的 F集合中的CPU
    for i, F_set in F_dict.items():
        if not F_set:
            continue
        cpu_F = {n for n in F_set if task_attr.get(n, {}).get('task_type') == 'CPU'}
        if not cpu_F:
            continue
        local_longest = compute_local_longest_paths(cpu_F, graph, weight)
        # 排序: (局部最长路径降序, rank降序, 节点升序)
        sorted_nodes = sorted(cpu_F, 
                              key=lambda x: (-local_longest.get(x, 0),
                                             -task_attr.get(x, {}).get('rank', 0),
                                             x))
        for node in sorted_nodes:
            priority[node] = p
            p -= 1

    return priority

##############################
# 6) GPU依赖 & 非独立GPU
##############################

def find_gpu_dependencies(reverse_graph, task_attr, all_nodes):
    """
    gpu_deps[g] = { ga1, ga2, ...} 若 g 祖先中存在其它GPU => ga.
    """
    gpu_deps = {}
    for node in all_nodes:
        if task_attr.get(node, {}).get('task_type') == 'GPU':
            anc = get_ancestors_with_reverse(reverse_graph, node)
            gpu_anc = {a for a in anc if task_attr.get(a, {}).get('task_type') == 'GPU'}
            if gpu_anc:
                gpu_deps[node] = gpu_anc
    return gpu_deps

##############################
# 7) 两GPU间 "最大CPU路径" + CPU节点列表
##############################

def compute_max_cpu_path_and_list(ga, g, graph, topo_order, node_weight, task_attr):
    """
    返回 (max_score, cpu_node_list)
      - max_score: ga->g 所经过CPU节点权重之和 的最大值
      - cpu_node_list: 该最优路径上的CPU节点ID (按顺序)
    若不可达 => None
    """
    dp = {n: (float('-inf'), None) for n in topo_order}

    # ga是GPU => 不计自身
    if ga in topo_order:
        dp[ga] = (0, None)
    else:
        return None

    # 若 ga 在拓扑序中比 g 晚 => 不可达
    try:
        ga_idx = topo_order.index(ga)
        g_idx = topo_order.index(g)
        if ga_idx >= g_idx:
            return None
    except ValueError:
        return None

    for node in topo_order:
        score, _ = dp[node]
        if score == float('-inf'):
            continue
        for succ in graph.get(node, []):
            add_w = node_weight[succ] if task_attr.get(succ, {}).get('task_type') == 'CPU' else 0
            candidate = score + add_w
            if candidate > dp[succ][0]:
                dp[succ] = (candidate, node)

    best_score, _ = dp[g] if g in dp else (float('-inf'), None)
    if best_score == float('-inf'):
        return None

    # 回溯
    path_nodes = []
    cur = g
    while cur is not None:
        path_nodes.append(cur)
        cur = dp[cur][1]
    path_nodes.reverse()

    # 保留CPU节点
    cpu_nodes = [n for n in path_nodes if task_attr.get(n, {}).get('task_type') == 'CPU']

    return (best_score, cpu_nodes)

##############################
# 8) 主程序
##############################

def main():
    random.seed(42)  # 结果可重复

    # 任务属性: ID => {rank, task_type}
    tasks_data = [
        (1,  {'rank': 0, 'task_type': 'CPU'}),
        (2,  {'rank': 1, 'task_type': 'CPU'}),
        (3,  {'rank': 1, 'task_type': 'GPU'}),
        (4,  {'rank': 1, 'task_type': 'CPU'}),
        (5,  {'rank': 1, 'task_type': 'GPU'}),
        (6,  {'rank': 1, 'task_type': 'CPU'}),
        (7,  {'rank': 2, 'task_type': 'GPU'}),
        (8,  {'rank': 2, 'task_type': 'GPU'}),
        (9,  {'rank': 2, 'task_type': 'CPU'}),
        (10, {'rank': 2, 'task_type': 'GPU'}),
        (11, {'rank': 3, 'task_type': 'GPU'}),
        (12, {'rank': 3, 'task_type': 'GPU'}),
        (13, {'rank': 3, 'task_type': 'CPU'}),
        (14, {'rank': 3, 'task_type': 'GPU'}),
        (15, {'rank': 4, 'task_type': 'CPU'}),
        (16, {'rank': 4, 'task_type': 'GPU'}),
        (17, {'rank': 4, 'task_type': 'CPU'}),
        (18, {'rank': 4, 'task_type': 'GPU'}),
        (19, {'rank': 5, 'task_type': 'GPU'}),
        (20, {'rank': 5, 'task_type': 'CPU'}),
        (21, {'rank': 5, 'task_type': 'GPU'}),
        (22, {'rank': 5, 'task_type': 'GPU'}),
        (23, {'rank': 6, 'task_type': 'CPU'})
    ]
    task_attr = {tid: attr for tid, attr in tasks_data}

    # DAG 边
    edges = [
        (1,4,{}), (1,6,{}), (1,2,{}), (1,3,{}), (1,5,{}), (1,16,{}),
        (2,8,{}), (2,9,{}),
        (3,7,{}), (3,9,{}), (3,10,{}),
        (4,8,{}), (4,10,{}),
        (5,7,{}), (5,8,{}), (5,10,{}),
        (6,7,{}), (6,9,{}),
        (7,11,{}),
        (8,13,{}), (8,14,{}),
        (9,13,{}), (9,14,{}),
        (10,12,{}), (10,13,{}),
        (11,15,{}), (11,17,{}), (11,18,{}),
        (12,15,{}),
        (13,15,{}), (13,17,{}), (13,18,{}),
        (14,15,{}), (14,17,{}),
        (15,22,{}),
        (16,22,{}),
        (17,19,{}), (17,22,{}),
        (18,20,{}), (18,21,{}), (18,22,{}),
        (19,23,{}),
        (20,23,{}),
        (21,23,{}),
        (22,23,{})
    ]

    # 随机生成节点权重(1~23 => 0号节点1和23节点为0)
    node_weight = {n: random.randint(1,10) for n in range(1,24)}
    node_weight[1] = 0
    node_weight[23] = 0

    # ======== 打印节点权重 ========
    print("\n======== 节点权重 (node_weight) ========")
    for n in sorted(node_weight):
        print(f"  节点 {n}: {node_weight[n]}")

    # ======== CPM计算 + 取回 EST, EFT, topo_order ========
    (critical_path,
     finish_time,
     graph,
     reverse_graph,
     all_nodes,
     EST, EFT,
     topo_order) = find_critical_path(edges, node_weight)

    # ======== 打印关键路径 & 总工期 ========
    print("\n======== 关键路径 & 总工期 ========")
    print(f"  关键路径: {critical_path}")
    print(f"  总工期:   {finish_time}")

    # ======== 合并 CP 组 ========
    cp_groups = identify_capacity_providers(critical_path, reverse_graph)
    print("\n======== 合并后的 Capacity Providers (CP groups) ========")
    for i, group in enumerate(cp_groups):
        print(f"  CP组 {i}: {group}")

    # ======== 计算 F 与 G ========
    F_dict, G_dict = compute_F_G(cp_groups, graph, reverse_graph, all_nodes, critical_path)
    print("\n======== 各 CP 对应的 F 与 G ========")
    for i in range(len(cp_groups)):
        print(f"  CP组 {i}:")
        print(f"    F(CP_{i}) = {F_dict[i]}")
        print(f"    G(CP_{i}) = {G_dict[i]}")

    # ======== 构建每个 CP 对应的潜在并行区域 R ========
    R_dict = {}
    for i, cp in enumerate(cp_groups):
        R = set(cp) | F_dict.get(i, set()) | G_dict.get(i, set())
        R_dict[i] = R
    print("\n======== 每个 CP组 的潜在并行区域 (R) ========")
    for i, R in R_dict.items():
        print(f"  CP组 {i} 的 R: {R}")

    # ======== 计算各 R 中 GPU 任务数量 & 最大GPU分区数量 ========
    gpu_counts = {}
    for i, R in R_dict.items():
        gpu_count = sum(
            1 for node in R 
            if task_attr.get(node, {}).get('task_type') == 'GPU'
        )
        gpu_counts[i] = gpu_count

    print("\n======== 各 R 中的 GPU 任务数量统计 ========")
    for i in sorted(gpu_counts.keys()):
        print(f"  CP组 {i} 的 R 中 GPU 任务数量: {gpu_counts[i]}")

    max_gpu_partition = max(gpu_counts.values()) if gpu_counts else 0
    print(f"\n======== GPU 最大分区数量 ========")
    print(f"  {max_gpu_partition}")

    # ======== 输出所有 GPU 任务集合 ========
    all_gpu_tasks = {
        node for node in all_nodes
        if task_attr.get(node, {}).get('task_type') == 'GPU'
    }
    print("\n======== 所有 GPU 任务集合 ========")
    print(f"  {all_gpu_tasks}")

    # ======== 发现 GPU 任务的依赖 (GPU -> 其GPU祖先) ========
    gpu_deps = find_gpu_dependencies(reverse_graph, task_attr, all_nodes)
    print("\n======== GPU 依赖关系 (GPU -> GPU祖先) ========")
    for g, ga_set in gpu_deps.items():
        print(f"  GPU 任务 {g} 依赖的 GPU 祖先: {ga_set}")

    # ======== 找到独立GPU任务 (无GPU祖先) ========
    dependent_gpu_tasks = set(gpu_deps.keys())
    independent_gpu_tasks = all_gpu_tasks - dependent_gpu_tasks
    print("\n======== 独立GPU任务 (无GPU祖先) ========")
    print(f"  {independent_gpu_tasks}")

    # ======== 生成 {独立GPU: 最早开始时间(EST)} 字典 ========
    initial_time = {gpu: EST[gpu] for gpu in sorted(independent_gpu_tasks)}
    print("\n======== 独立GPU任务的最早开始时间 (initial_time) ========")
    print("initial_time = {")
    for gpu in sorted(initial_time.keys()):
        print(f"    {gpu}: {initial_time[gpu]},")
    print("}")

    # ======== 两GPU间最大CPU路径 => 构造字典形式 (gpu_path_map) ========
    print("\n======== 两GPU间\"最大CPU路径\" 结构性数据 (gpu_deps -> gpu_path_map) ========")
    non_independent_gpu_tasks = sorted(dependent_gpu_tasks)  # 有 GPU 祖先的
    gpu_path_map = {}
    for job in non_independent_gpu_tasks:
        gpu_path_map[job] = {}
        for dep_job in sorted(gpu_deps[job]):
            res = compute_max_cpu_path_and_list(
                dep_job, 
                job, 
                graph, 
                topo_order, 
                node_weight, 
                task_attr
            )
            if res is None:
                gpu_path_map[job][dep_job] = {
                    "cpu_path_length": 0,
                    "cpu_nodes": []
                }
            else:
                length, cpu_nodes = res
                gpu_path_map[job][dep_job] = {
                    "cpu_path_length": length,
                    "cpu_nodes": cpu_nodes
                }
    print(f"  {gpu_path_map}")

    # ======== CPU 任务优先度分配 ========
    cpu_priority = assign_cpu_priorities(
        cp_groups, 
        F_dict, 
        task_attr, 
        graph, 
        node_weight
    )
    print("\n======== CPU 任务优先度分配结果 ========")
    for node in sorted(cpu_priority.keys()):
        print(f"  CPU任务 {node} => 优先度 {cpu_priority[node]}")

    print("\n======== 所有输出完毕 ========\n")


if __name__ == "__main__":
    main()




