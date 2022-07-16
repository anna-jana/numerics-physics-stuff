import heapq, random, collections
random.seed(42)

def a_star(start_node, is_goal, get_estimated_distance_to_goal,
        get_distance_between, get_neighboring_nodes):
    # heap of nodes we know about but dont know the shortest path from start_node to yet
    open_nodes_heap = [(get_estimated_distance_to_goal(start_node), start_node)]
    # set of nodes for which we know the shortest path from start_node
    closed_list = set()
    # table of nodes with the predessors in the shortest paths to the nodes,
    # we know so far in the algorithm, used to construct the path at the end
    predecessors = dict()
    # shortest distance we know to a node
    distance_so_far_map = {start_node: 0}
    # cached estimated distance from node to goal
    estimated_dist_to_goal = {}
    while len(open_nodes_heap) > 0:
        # expand on the best path we have found so far
        estimated_total_dist, current_node = heapq.heappop(open_nodes_heap)
        if is_goal(current_node):
            # construct path
            path = [current_node]
            while path[-1] != start_node:
                path.append(predecessors[path[-1]])
            return path
        # we have found the shortest path from the start_node to the current_node, so we can add
        # it to the closed_list
        closed_list.add(current_node)
        # go over all neighbors of the current node
        for neighbor_node in get_neighboring_nodes(current_node):
            # skip if we have already found the shortest path from start_node to neighbor_node
            if neighbor_node in closed_list:
                continue
            # compute the value of the shortes distance from start_node to this node we have found so far
            dist_between_nodes = get_distance_between(current_node, neighbor_node)
            new_current_dist_so_far_neighbor = distance_so_far_map[current_node] + dist_between_nodes
            # if we already encountered this node, check the previous shortest path from
            # start node to it
            is_in_open = neighbor_node in map(lambda p: p[1], open_nodes_heap)
            if is_in_open:
                # new path to neighbor is longer than the one found previously
                old_current_dist_so_far_neighbor = distance_so_far_map[neighbor_node]
                if new_current_dist_so_far_neighbor >= old_current_dist_so_far_neighbor:
                    continue
            # okay the path using our current_node is indeed the shortest one from start_node
            # to the neighbor_node
            # hence set the predecessors and distance from start_node
            predecessors[neighbor_node] = current_node
            distance_so_far_map[neighbor_node] = new_current_dist_so_far_neighbor
            # also compute the estimated distance from neighbor_node to goal_node
            # and cache the result
            if neighbor_node in estimated_dist_to_goal:
                e = estimated_dist_to_goal[neighbor_node]
            else:
                e = get_estimated_distance_to_goal(neighbor_node)
                estimated_dist_to_goal[neighbor_node] = e
            # also recompute the estinmated total distances using current_node and neighbor_node
            new_estimated_total_dist_neighbor_node = new_current_dist_so_far_neighbor + e
            # update the open_nodes_heap with this new path
            if is_in_open:
                # update distance if it is already in the open_nodes_heap and we adjust
                # the heap
                for i, (_, node) in enumerate(open_nodes_heap):
                    if node == neighbor_node:
                        open_nodes_heap[i][0] = new_estimated_total_dist_neighbor_node
                        break
                heapq.heapify(open_nodes_heap)
            else:
                # insert if its not in the open_nodes_heap yet
                heapq.heappush(open_nodes_heap, (new_estimated_total_dist_neighbor_node, neighbor_node))
    # we searched exclusivly and found no path to the goal_node

    return None

def gen_graph(nnodes, nlinks):
    graph = collections.defaultdict(lambda: set())
    for i in range(nlinks):
        graph[random.randint(0, nnodes - 1)].add(random.randint(0, nnodes - 1))
    return graph


def a_star_dict(start_node, goal_node, graph):
    return a_star(start_node, lambda node: node == goal_node,
            lambda node: 0.0, lambda from_node, to_node: 1.0 if to_node in graph[from_node] else 0.0, lambda node: graph[node])

n = 10
g = gen_graph(n, 30)
for i in range(n):
    for j in range(n):
        path = a_star_dict(i, j, g)
        print(i, "->", j, ":", path)
