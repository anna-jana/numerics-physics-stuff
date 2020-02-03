# based on: https://wordsandbuttons.online/how_much_math_can_you_do_in_10_lines_of_python.html
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def eye(n): return [[float(i==j) for j in range(n)] for i in range(n)]
def trans(A): return list(zip(*A))
def mat_vec_mul(A, x): return [dot(a, x) for a in A]
def add(a,b): return [x + y for x,y in zip(a, b)]
def scale(x,a): return [x*y for y in a]
def dist(a, b): return dot(*(add(a, scale(-1, b)),)*2)**0.5
def proj(A, Pn, Pd): return add(A, scale((Pd - dot(Pn, A)) / dot(Pn, Pn), Pn))
def solve(A, b, xi): return xi if dist(mat_vec_mul(A, xi), b) < 1e-5 else solve(A[1:] + [A[0]], b[1:] + [b[0]], proj(xi, A[0], b[0]))
def inv(A): return trans([solve(A, o, [.0]*len(A)) for o in eye(len(A))])
def mat_mul(A, B): return [mat_vec_mul(A, b) for b in trans(B)]
