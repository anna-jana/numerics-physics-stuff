import itertools

def primes_to(n):
    primes = list(range(2, n + 1))
    i = 0
    while i < len(primes):
        p = primes[i]
        j = i + 1
        while j < len(primes):
            if primes[j] % p == 0:
                del primes[j]
            else:
                j += 1
        i += 1
    return primes

def prime_factors(n):
    primes = primes_to(n)
    factors = []
    while n != 1:
        for p in primes:
            if n % p == 0:
                factors.append(p)
                n //= p
                break
    return factors

def most_equal_products(factors, n):
    products = [1] * n
    factors = sorted(factors)
    i = 0
    for f in factors:
        products[i] *= f
        i = (i + 1) % n
    return products

def split_grid_points_into_boxes(num_grid_points, num_boxes):
    box_sizes = [num_grid_points // num_boxes] * num_boxes
    for i in range(num_grid_points % num_boxes):
        box_sizes[i] += 1
    return box_sizes

def get_node_boxes_sizes(grid_dimension, num_nodes):
    box_axis_lengths = most_equal_products(prime_factors(num_nodes), len(grid_dimension))
    return [split_grid_points_into_boxes(grid_axis_length, box_axis_length)
        for grid_axis_length, box_axis_length in zip(grid_dimension, box_axis_lengths)]

class DistributedGrid:
    def __init__(self, grid_dimension, num_nodes):
        self.boxes = get_node_boxes_sizes(grid_dimension, num_nodes)
        self.axis_lengths = list(map(len, self.boxes))
        self.node_id_to_box_size = dict(enumerate(itertools.product(*self.boxes)))
        self.node_id_to_box_index = dict(enumerate(itertools.product(*map(lambda b: range(len(b)), self.boxes))))
        self.box_index_to_node_id = {box_index : node_id for node_id, box_index in self.node_id_to_box_index.items()}

def check(num_grid_points_x, num_grid_points_y, num_grid_points_z, num_nodes):
    print("checking:", num_grid_points_x, "*", num_grid_points_y, "*", num_grid_points_z, "on", num_nodes, "nodes")
    boxes_z, boxes_y, boxes_x = get_node_boxes_sizes(
            (num_grid_points_x, num_grid_points_y, num_grid_points_z), num_nodes)
    print("getting:", len(boxes_x), "*", len(boxes_y), "*", len(boxes_z))
    num_grid_points = num_grid_points_x * num_grid_points_y * num_grid_points_z
    assert sum(boxes_x) * sum(boxes_y) * sum(boxes_z) == num_grid_points, \
            "total number of grid points doesnt match sum of grid points"
    assert len(boxes_x) * len(boxes_y) * len(boxes_z) == num_nodes, \
            "total number of nodes doesnt match sum of nodes"
    print("ok")

if __name__ == "__main__":
    for i in range(1, 5):
        for j in range(1, 5):
            for k in range(1, 5):
                for l in range(1, i*j*k + 1):
                    check(i, j, k, l)




