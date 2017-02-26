import numpy as np

def get_nums_avl_at(pzl, (row, col), rows, columns, boxes, all_nums):
    mask = (row == rows) | (col == columns) | (boxes[row, col] == boxes)
    return all_nums - set(pzl[mask])

def set_at(pzl, idx, num):
    pzl = pzl.copy()
    pzl[idx] = num
    return pzl

def place_numbers_at(pzl, idx, *args):
    return [set_at(pzl, idx, num) for num in get_nums_avl_at(pzl, idx, *args)]

def search_step(pzls, idx, *args):
    return reduce(lambda a, b: a + b, [place_numbers_at(pzl, idx, *args) for pzl in pzls], [])

def get_empty_cells(pzl, n_sq):
    return [(row, col) for row in xrange(n_sq) for col in xrange(n_sq) if pzl[row, col] == 0]

def solve(pzl):
    n_sq = pzl.shape[0]
    n = int(np.sqrt(n_sq))
    boxes = np.arange(n_sq).reshape((n, n)).repeat(n, axis=0).repeat(n, axis=1)
    rows = np.arange(n_sq).reshape((n_sq, 1)).repeat(n_sq, axis=1)
    columns = np.arange(n_sq).reshape((1, n_sq)).repeat(n_sq, axis=0)
    all_nums = set(range(1, n_sq + 1))
    return reduce(lambda sols, idx: search_step(sols, idx, rows, columns, boxes, all_nums),
                  get_empty_cells(pzl, n_sq),
                  [pzl])

p = np.array([[0, 5, 0, 0, 6, 0, 0, 0, 1],
              [0, 0, 4, 8, 0, 0, 0, 7, 0],
              [8, 0, 0, 0, 0, 0, 0, 5, 2],
              [2, 0, 0, 0, 5, 7, 0, 3, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 3, 0, 6, 9, 0, 0, 0, 5],
              [7, 9, 0, 0, 0, 0, 0, 0, 8],
              [0, 1, 0, 0, 0, 6, 5, 0, 0],
              [5, 0, 0, 0, 3, 0, 0, 6, 0]])

p2 = np.array([[0,0,0,0],
               [0,0,2,1],
               [3,0,0,4],
               [0,0,0,0]])

print "======== 9x9 ======="
print(solve(p)[0])

print "=== 4x4 ===="
print(solve(p2)[0])
