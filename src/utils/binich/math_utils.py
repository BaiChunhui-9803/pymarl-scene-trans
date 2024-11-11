
import math

class Circle:
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r

def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def is_in_circle(c, p):
    return dist((c.x, c.y), p) <= c.r

def circle_from_two_points(p1, p2):
    center_x = (p1[0] + p2[0]) / 2
    center_y = (p1[1] + p2[1]) / 2
    radius = dist(p1, p2) / 2
    return Circle(center_x, center_y, radius)

def circle_from_three_points(p1, p2, p3):
    ax, ay = p1
    bx, by = p2
    cx, cy = p3
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d
    center = (ux, uy)
    radius = dist(center, p1)
    return Circle(ux, uy, radius)

def welzl(P, R):
    if len(P) == 0 or len(R) == 3:
        if len(R) == 0:
            return Circle(0, 0, 0)
        elif len(R) == 1:
            return Circle(R[0][0], R[0][1], 0)
        elif len(R) == 2:
            return circle_from_two_points(R[0], R[1])
        elif len(R) == 3:
            return circle_from_three_points(R[0], R[1], R[2])
    p = P.pop()
    d = welzl(P, R)
    if is_in_circle(d, p):
        P.append(p)
        return d
    R.append(p)
    d = welzl(P, R)
    R.pop()
    P.append(p)
    return d

def find_min_circle(points):
    P = points[:]
    return welzl(P, [])