import random
from collections import defaultdict, deque

##############################
# 图构建、拓扑排序、CPM等函数
##############################

def build_graph(edges):
    """
    根据边列表构建正向图、入度字典和反向图（前驱关系），同时收集所有节点。
    edges 格式为 (u, v, {}) ，其中第三项为占位符。
    """
    graph = defaultdict(list)
    in_degree = defaultdict(int)
    reverse_graph = defaultdict(list)
    nodes = set()
    for u, v, _ in edges:
        graph[u].append(v)
        in_degree[v] += 1
        if u not in in_degree:
            in_degree[u] = 0
        reverse_graph[v].append(u)
        nodes.add(u)
        nodes.add(v)
    return dict(graph), dict(in_degree), dict(reverse_graph), nodes

def topological_sort(graph, in_degree, nodes):
    """利用 Kahn 算法进行拓扑排序"""
    queue = deque([node for node in nodes if in_degree[node] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for neighbor in graph.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    return topo_order

def find_critical_path(edges, node_weight):
    """
    利用关键路径法 (CPM) 计算项目总工期，并提取一条关键路径。
    
    返回：
      critical_path: 按拓扑顺序排列的关键路径（仅包含零余量节点）
      finish_time: 项目总工期
      graph: 正向图
      reverse_graph: 反向图
      nodes: 所有节点集合
    """
    graph, in_degree, reverse_graph, nodes = build_graph(edges)
    topo_order = topological_sort(graph, in_degree.copy(), nodes)
    
    # 正向遍历：计算最早开始时间 EST 和最早完成时间 EFT = EST + weight
    EST = {node: 0 for node in nodes}
    EFT = {node: 0 for node in nodes}
    for node in topo_order:
        EFT[node] = EST[node] + node_weight.get(node, 1)
        for neighbor in graph.get(node, []):
            EST[neighbor] = max(EST[neighbor], EFT[node])
    
    # 项目总工期：所有没有后继节点的 EFT 的最大值
    sink_nodes = [node for node in nodes if node not in graph or not graph[node]]
    finish_time = max(EFT[node] for node in sink_nodes)
    
    # 反向遍历：计算最晚完成时间 LFT 和最晚开始时间 LST = LFT - weight
    LFT = {node: finish_time for node in nodes}
    LST = {node: finish_time for node in nodes}
    for node in reversed(topo_order):
        if node in graph and graph[node]:
            LFT[node] = min(LST[succ] for succ in graph[node])
        LST[node] = LFT[node] - node_weight.get(node, 1)
    
    # 提取零余量节点：认为 EST == LST 的节点为关键节点
    critical_nodes = [node for node in topo_order if EST[node] == LST[node]]
    
    # 构造连续关键路径：从起始节点开始沿着满足条件的后继依次构造
    cp = []
    start_nodes = [node for node in topo_order if EST[node] == 0 and node in critical_nodes]
    if not start_nodes:
        return [], finish_time, graph, reverse_graph, nodes
    current = start_nodes[0]
    cp.append(current)
    while True:
        next_candidates = [n for n in graph.get(current, [])
                           if n in critical_nodes and EST[n] == EFT[current]]
        if not next_candidates:
            break
        current = next_candidates[0]
        cp.append(current)
    
    return cp, finish_time, graph, reverse_graph, nodes

def get_ancestors_with_reverse(reverse_graph, node):
    """
    利用反向图返回 node 的所有祖先（不包含 node 本身）。
    """
    ancestors = set()
    stack = [node]
    while stack:
        current = stack.pop()
        for pred in reverse_graph.get(current, []):
            if pred not in ancestors:
                ancestors.add(pred)
                stack.append(pred)
    return ancestors

def get_descendants(graph, node):
    """
    利用正向图返回 node 的所有后继（不包含 node 本身）。
    """
    descendants = set()
    stack = [node]
    while stack:
        current = stack.pop()
        for succ in graph.get(current, []):
            if succ not in descendants:
                descendants.add(succ)
                stack.append(succ)
    return descendants

def identify_capacity_providers(critical_path, reverse_graph):
    """
    对关键路径节点进行合并：
      如果关键路径中下一个节点的直接前驱集合正好为 {前一个节点}，
      则认为它们属于同一个 Capacity Provider（CP）。
    
    返回：一个列表，每个元素是一个 CP（列表形式），例如 [[v1, v2], [v3], [v4, v5]]。
    """
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
    对于每个 Capacity Provider CP（除最后一个外），计算：
      F(CP_i) = anc(θ_{i+1}) ∩ NonCritical,
      其中 θ_{i+1} 取下一个 CP 中的第一个节点，
      NonCritical = all_nodes - (所有关键路径节点)
      
      对于每个 v ∈ F(CP_i)，定义：
      C(v) = all_nodes - (anc(v) ∪ desc(v)),
      然后 G(CP_i) = ∪_{v ∈ F(CP_i)} (C(v) ∩ NonCritical).
      
    每次计算完一个 CP 的 F 后，从 NonCritical 中删除对应的 F 节点。
    对于最后一个 CP，我们令 F 与 G 均为空集合。
    返回两个字典 F_dict 与 G_dict，其键为 CP 的索引。
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
# 新功能：CPU任务优先度分配
##############################

def compute_local_longest_paths(F_set, graph, weight):
    """
    在由 F_set 诱导的局部子图上，计算每个节点的最长路径长度。
    这里采用 DFS + 记忆化方法，节点的最长路径长度计算公式为：
         longest(u) = weight[u] + max_{v in local_graph[u]} longest(v)
    如果 u 没有后继，则 longest(u) = weight[u]。
    参数:
       F_set: 局部子图的节点集合
       graph: 原始图（只使用 F_set 内的边）
       weight: 每个节点的权重字典
    返回一个字典 {node: longest_length}。
    """
    memo = {}
    local_graph = {u: [v for v in graph.get(u, []) if v in F_set] for u in F_set}
    
    def dfs(u):
        if u in memo:
            return memo[u]
        current_weight = weight.get(u, 1)
        max_len = current_weight  # 从节点u自身开始计权重
        for v in local_graph.get(u, []):
            max_len = max(max_len, current_weight + dfs(v))
        memo[u] = max_len
        return max_len
    
    for node in F_set:
        dfs(node)
    return memo

def assign_cpu_priorities(cp_groups, F_dict, task_attr, graph, weight):
    """
    为 CPU 任务分配优先度，分两步：
    1. 对所有 Capacity Providers 中的 CPU任务，
       按照它们在 CP 组中的顺序分配优先度，即组内序号越低的任务获得更高优先度。
    2. 对于每个 CP 组的 F 集合中的 CPU任务，先在局部子图上计算最长局部路径长度，
       然后按（局部最长路径, rank）降序排序（局部最长路径越长且 rank 越高的任务优先度越高），
       最后依次分配优先度。
       
    初始优先度 p 设为 100，每分配一个任务后 p 递减。
    返回一个字典 {node: priority}（仅对 CPU任务分配）。
    """
    priority = {}
    p = 100

    # 1. 为 Capacity Providers 中的 CPU任务分配优先度（组内按序号分配）
    for group in cp_groups:
        for node in group:
            if task_attr.get(node, {}).get('task_type') == 'CPU':
                priority[node] = p
                p -= 1

    # 2. 为每个 CP 的 F 集合中的 CPU任务分配优先度
    for i, F_set in F_dict.items():
        if not F_set:
            continue
        cpu_F = {node for node in F_set if task_attr.get(node, {}).get('task_type') == 'CPU'}
        if not cpu_F:
            continue
        local_longest = compute_local_longest_paths(cpu_F, graph, weight)
        # print(f"CP组 {i} 中 CPU任务局部最长路径: {local_longest}")
        # 排序：先按局部最长路径降序，再按任务属性中的 rank 降序，最后按节点编号升序
        sorted_nodes = sorted(cpu_F, key=lambda x: (-local_longest.get(x, 0),
                                                      -task_attr.get(x, {}).get('rank', 0),
                                                      x))
        for node in sorted_nodes:
            priority[node] = p
            p -= 1

    return priority

##############################
# GPU任务依赖关系与独立GPU任务
##############################

def find_gpu_dependencies(reverse_graph, task_attr, all_nodes):
    """
    对于所有 GPU 任务，检查其祖先中是否存在 GPU 任务，
    如果存在则记录依赖关系：{ GPU_task: {GPU_ancestor, ...} }
    """
    gpu_deps = {}
    for node in all_nodes:
        if task_attr.get(node, {}).get('task_type') == 'GPU':
            ancestors = get_ancestors_with_reverse(reverse_graph, node)
            gpu_ancestors = {anc for anc in ancestors if task_attr.get(anc, {}).get('task_type') == 'GPU'}
            if gpu_ancestors:
                gpu_deps[node] = gpu_ancestors
    return gpu_deps


##############################
# 主程序
##############################

def main():
    
    random.seed(42)  # 保持随机性可重复
    
    # --- 任务属性数据 ---
    tasks_data = [(1, {'rank': 0, 'task_type': 'CPU'}), (2, {'rank': 1, 'task_type': 'CPU'}), (3, {'rank': 1, 'task_type': 'GPU'}), (4, {'rank': 1, 'task_type': 'CPU'}), (5, {'rank': 1, 'task_type': 'GPU'}), (6, {'rank': 1, 'task_type': 'CPU'}), (7, {'rank': 2, 'task_type': 'GPU'}), (8, {'rank': 2, 'task_type': 'GPU'}), (9, {'rank': 2, 'task_type': 'CPU'}), (10, {'rank': 2, 'task_type': 'GPU'}), (11, {'rank': 3, 'task_type': 'GPU'}), (12, {'rank': 3, 'task_type': 'GPU'}), (13, {'rank': 3, 'task_type': 'CPU'}), (14, {'rank': 3, 'task_type': 'GPU'}), (15, {'rank': 4, 'task_type': 'CPU'}), (16, {'rank': 4, 'task_type': 'GPU'}), (17, {'rank': 4, 'task_type': 'CPU'}), (18, {'rank': 4, 'task_type': 'GPU'}), (19, {'rank': 5, 'task_type': 'GPU'}), (20, {'rank': 5, 'task_type': 'CPU'}), (21, {'rank': 5, 'task_type': 'GPU'}), (22, {'rank': 5, 'task_type': 'GPU'}), (23, {'rank': 6, 'task_type': 'CPU'})]
    task_attr = {tid: attr for tid, attr in tasks_data}
    
    # --- 示例 DAG 边列表 ---
    edges = [(1, 4, {}), (1, 6, {}), (1, 2, {}), (1, 3, {}), (1, 5, {}), (1, 16, {}), (2, 8, {}), (2, 9, {}), (3, 7, {}), (3, 9, {}), (3, 10, {}), (4, 8, {}), (4, 10, {}), (5, 7, {}), (5, 8, {}), (5, 10, {}), (6, 7, {}), (6, 9, {}), (7, 11, {}), (8, 13, {}), (8, 14, {}), (9, 13, {}), (9, 14, {}), (10, 12, {}), (10, 13, {}), (11, 15, {}), (11, 17, {}), (11, 18, {}), (12, 15, {}), (13, 15, {}), (13, 17, {}), (13, 18, {}), (14, 15, {}), (14, 17, {}), (15, 22, {}), (16, 22, {}), (17, 19, {}), (17, 22, {}), (18, 20, {}), (18, 21, {}), (18, 22, {}), (19, 23, {}), (20, 23, {}), (21, 23, {}), (22, 23, {})]
    
    # --- 节点权重生成 ---
    node_weight = {n: random.randint(1, 10) for n in range(1, 24)}
    node_weight[1] = 0
    node_weight[23] = 0
    print("节点权重:")
    for n in sorted(node_weight):
        print(f"  节点 {n}: {node_weight[n]}")
    
    # --- 计算关键路径 (CPM) ---
    critical_path, finish_time, graph, reverse_graph, all_nodes = find_critical_path(edges, node_weight)
    print("\n关键路径:", critical_path)
    print("项目总工期:", finish_time)
    print("所有节点:", all_nodes)
    
    # --- 识别并合并 Capacity Providers ---
    cp_groups = identify_capacity_providers(critical_path, reverse_graph)
    print("\n合并后的 Capacity Providers (CP groups):")
    for i, group in enumerate(cp_groups):
        print(f"  CP组 {i}: {group}")
    
    # --- 计算 F 与 G ---
    F_dict, G_dict = compute_F_G(cp_groups, graph, reverse_graph, all_nodes, critical_path)
    print("\n各 CP 对应的 F 与 G:")
    for i in range(len(cp_groups)):
        print(f"  CP组 {i} (包含节点 {cp_groups[i]}):")
        print(f"    F(CP_{i}) = {F_dict[i]}")
        print(f"    G(CP_{i}) = {G_dict[i]}")
        print("-----------")
    
    # --- 构建每个 CP 对应的潜在并行区域 R ---
    R_dict = {}
    for i, cp in enumerate(cp_groups):
        R = set(cp) | F_dict.get(i, set()) | G_dict.get(i, set())
        R_dict[i] = R
        print(f"CP组 {i} 的潜在并行区域 R = {R}")
    
    # --- 计算每个 R 中 GPU 任务的数量 ---
    print("\n各 R 中 GPU 任务数量统计:")
    gpu_counts = {}
    for i, R in R_dict.items():
        gpu_count = sum(1 for node in R if task_attr.get(node, {}).get('task_type') == 'GPU')
        gpu_counts[i] = gpu_count
        print(f"  CP组 {i} 的 R 中 GPU 任务数量: {gpu_count}")
    
    # --- 输出 GPU 最大分区数量 ---
    max_gpu_partition = max(gpu_counts.values()) if gpu_counts else 0
    print(f"\nGPU最大分区数量: {max_gpu_partition}")

    # --- 输出 GPU任务集合 ---
    all_gpu_tasks = {node for node in all_nodes if task_attr.get(node, {}).get('task_type') == 'GPU'}
    print("\n所有 GPU 任务集合(做为GPU分区问题的输入):")
    print(all_gpu_tasks)
    
    # --- 发现 GPU 任务之间的依赖关系 ---
    gpu_deps = find_gpu_dependencies(reverse_graph, task_attr, all_nodes)
    print("\nGPU任务之间的依赖关系(做为GPU分区问题的输入):")
    print(gpu_deps)
    print("\nGPU任务之间的依赖关系:")
    for gpu, deps in gpu_deps.items():
        print(f"  GPU任务 {gpu} 依赖于 GPU任务: {deps}")

    
    # --- 打印独立的 GPU 任务 ---
    all_gpu_tasks = {node for node in all_nodes if task_attr.get(node, {}).get('task_type') == 'GPU'}
    dependent_gpu_tasks = set(gpu_deps.keys())
    independent_gpu_tasks = all_gpu_tasks - dependent_gpu_tasks
    print("\n独立的 GPU 任务:")
    for gpu in independent_gpu_tasks:
        print(f"  GPU任务 {gpu}")
    
    # --- 分配 CPU任务优先度 ---
    cpu_priority = assign_cpu_priorities(cp_groups, F_dict, task_attr, graph, node_weight)
    print("\nCPU任务优先度分配结果:")
    for node in sorted(cpu_priority.keys()):
        print(f"  CPU任务 {node} 的优先度: {cpu_priority[node]}")

if __name__ == "__main__":
    main()



