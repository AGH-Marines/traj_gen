import math, json

# oryginalna trasa
path = [
    [0.0, 0.0, 2.0],
    [2.0, 2.0, 2.5],
    [4.0, 0.0, 3.0],
    [2.0, -2.0, 2.5],
    [0.0, 0.0, 2.0],
    [-2.0, 2.0, 1.5],
    [-4.0, 0.0, 1.0],
    [-2.0, -2.0, 1.5],
    [0.0, 0.0, 2.0]
]

max_dist = 1
out = []

for a, b in zip(path, path[1:]):
    x1, y1, z1 = a
    x2, y2, z2 = b
    dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
    d = math.sqrt(dx**2 + dy**2 + dz**2)
    steps = max(1, math.ceil(d / max_dist))
    for s in range(steps):
        t = s / steps
        out.append([
            round(x1 + t * dx, 6),
            round(y1 + t * dy, 6),
            round(z1 + t * dz, 6)
        ])
out.append(path[-1])