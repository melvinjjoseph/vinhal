from vinhal import *

X_train=[[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]]
y = [9, 22, 29]
w = [1.5, 2.5, 3.5, 4.5]
b = 2
print(compute_cost_multi(X_train, y, w, b))