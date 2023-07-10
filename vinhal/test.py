import vinhal

x = [10, 20, 30, 40]
y = [100, 200, 300, 400]
w = 2
b = 2

x_train = [1.0, 2.0, 3.0]
y_train = [100, 200, 300]
w_init=0
b_init=0
alpha=0.01
iters=10000

w_final, b_final, J_hist, p_hist= vinhal.gradient_descent(x_train, y_train, w_init, b_init, alpha, iters)
