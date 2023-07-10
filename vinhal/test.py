import vinhal

x_train = [1.0, 2.0, 3.0]
y_train = [300, 500, 680]
w_init=0
b_init=0
alpha=0.01
iters=10000

model = vinhal.gradient_descent(x_train, y_train, w_init, b_init, alpha, iters)
print(model.predict_x(x = 1.5))