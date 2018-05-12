import random
import threading
import time
import numpy as np
from scipy.spatial import distance

# Slightly optimized algorithm explained in :
# https://www.coursera.org/learn/algorithms-divide-conquer/lecture/nf0jk/o-n-log-n-algorithm-for-closest-pair-i-advanced-optional
class ClosestPairAlgorithm:
    def __init__(self, problem_size = 16):
        self.default_problem_size = problem_size
        self.min_boundary = 0
        self.max_boundary = 100
        self.points = self.generate_random_points(problem_size = self.default_problem_size)
        self.px = [el[0] for el in enumerate(np.sort(self.points, axis=0, kind='mergesort'))]
        self.py = [el[0] for el in enumerate(np.sort(self.points, axis=1, kind='mergesort'))]

    def generate_random_points(self, problem_size = None):
        return np.random.randint(self.min_boundary, self.max_boundary, size=(problem_size, 2))

    def closest_pair(self, arr = None):
        arr = arr if arr else self.points
        px = np.sort(self.points, axis=0, kind='mergesort')
        py = sorted(arr, key=lambda row: row[1])
        p1, p2 = self._closest_pair_recurrent(px, py)
        return p1, p2,  distance.euclidean(p1, p2)

    def brutal_force_algorithm(self, arr=None):
        (i_min, j_min) = (None, None)
        arr = arr  if arr is not None else self.points
        min_dist = self.max_boundary * 100
        for i in range(0, len(arr)):
            for j in range(i + 1, len(arr)):
                d = distance.euclidean(arr[i], arr[j])
                if d < min_dist:
                    min_dist = d
                    (i_min, j_min) = (i, j)

        return (self.points[i_min], self.points[j_min])

    def _closest_pair_recurrent(self, px, py):
        if len(px) < 4:
            return self.brutal_force_algorithm(arr = px)
        middle = len(px) // 2
        qx = px[:middle]
        rx = px[middle:]
        splitting_point = px[middle][0]
        qy, ry = [],[]

        for x in py:
            if x[0] <= splitting_point:
                qy.append(x)
            else:
                ry.append(x)

        (q1, q2) = self._closest_pair_recurrent(qx, qy)
        (r1, r2) = self._closest_pair_recurrent(rx, ry)
        q_dist = distance.euclidean(q1, q2)
        r_dist = distance.euclidean(r1, r2)
        min1, min2 = (q1, q2) if q_dist < r_dist else (r1, r2)
        min_dist = min(q_dist, r_dist)
        sy = [x for x in py if splitting_point - min_dist <= x[0] <= splitting_point + min_dist]
        for i in range(len(sy) - 1):
            for j in range(i + 1, min(i + 7, len(sy))):
                dist = distance.euclidean(sy[i], sy[j])
                if dist < min_dist:
                    min1, min2 = (sy[i], sy[j])
                    min_dist = dist
        return min1, min2


    def closest_pair_par(self, arr = None):
        arr = arr if arr else self.points
        px = np.sort(self.points, axis=0, kind='mergesort')
        py = sorted(arr, key=lambda row: row[1])
        res = {'res': None}
        self._closest_pair_recurrent_par(px, py, res)
        p1, p2 = res['res']
        return p1, p2, distance.euclidean(p1, p2)

    def _closest_pair_recurrent_par(self, px, py, res):
        if len(px) <= 12:
            res['res'] = self.brutal_force_algorithm(arr = px)
            return
        middle = len(px) // 2
        qx = px[:middle]
        rx = px[middle:]
        splitting_point = px[middle][0]
        qy, ry = [],[]

        for x in py:
            if x[0] <= splitting_point:
                qy.append(x)
            else:
                ry.append(x)
        res1 = {'res':None}
        res2 = {'res':None}
        t1 = threading.Thread(target=self._closest_pair_recurrent_par, args=(qx, qy, res1))
        t2 = threading.Thread(target=self._closest_pair_recurrent_par, args=(rx, ry, res2))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        (q1, q2) = res1['res']
        (r1, r2) = res2['res']
        q_dist = distance.euclidean(q1, q2)
        r_dist = distance.euclidean(r1, r2)
        min1, min2 = (q1, q2) if q_dist < r_dist else (r1, r2)
        min_dist = min(q_dist, r_dist)
        sy = [x for x in py if splitting_point - min_dist <= x[0] <= splitting_point + min_dist]
        for i in range(len(sy) - 1):
            for j in range(i + 1, min(i + 7, len(sy))):
                dist = distance.euclidean(sy[i], sy[j])
                if dist < min_dist:
                    min1, min2 = (sy[i], sy[j])
                    min_dist = dist
        res['res'] =  (min1, min2)


if __name__ == '__main__':
    cpa = ClosestPairAlgorithm(problem_size=2**10)

    t_brut = time.time()
    print(cpa.brutal_force_algorithm())
    print(f'brut elapsed {time.time() - t_brut}')

    t_seq = time.time()
    print(cpa.closest_pair())
    print(f'seq elapsed {time.time() - t_seq}')

    t_par = time.time()
    print(cpa.closest_pair_par())
    print(f'par elapsed {time.time() - t_par}')

