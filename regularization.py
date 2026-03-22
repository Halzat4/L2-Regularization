import numpy as np
from sklearn.model_selection import train_test_split


np.random.seed(42)
N = 60
X = np.linspace(-3, 3, N)
y_true = np.sin(X) * 2 + 0.5
noise = np.random.normal (0, 0.6 , N)
y_hat = y_true + noise

X_train, X_test, y_train, y_test = train_test_split(
    X.reshape(-1, 1), y_hat, test_size=0.3, random_state=42
)

def loss(y_true, y_hat):
	s = 0

	for i in range(2):
		s += 1/2 * (y_true[i] - y_hat[i]) ** 2

	return s

print(loss(y_true, y_hat))

def loss_with_l2(y_true, y_hat, weights):
	z = 0

	for i in range(2):
		z += 1/2 * (y_true[i] - y_hat[i]) ** 2

	l2_penalty = 0
	for w in weights:
		l2_penalty += w ** 2

	return z + 0.1 * l2_penalty

weights = [0.5]
print(f"Loss + L2: {loss_with_l2(y_true, y_hat, weights)}")


def update_weights(X, y_true, w, alpha, lr):
	dw = 0
	for i in range(2):
		dw += (y_hat[i] - y_true[i]) * X[i]

	dw = dw + (2 * alpha * w)

	new_w = w - lr * dw
	return new_w

alpha = 0.1
learning_rate = 0.1
current_w = weights[0]
new_w = update_weights(X, y_hat, current_w, alpha, learning_rate)
w1 = []
w1.append(new_w)
print(f"Старый вес: {current_w}")
print(f"Новый вес: {new_w:.4f}")

def loss_with_l2(y_true, y_hat, weights1):
	z = 0

	for i in range(2):
		z += 1/2 * (y_true[i] - y_hat[i]) ** 2

	l2_penalty = 0
	for w in weights1:
		l2_penalty += w ** 2

	return z + 0.1 * l2_penalty

weights1 = w1
print(f"New Loss + L2: {loss_with_l2(y_true, y_hat, weights1)}")


epochs = 1000
epsilon = 1e-6
history_w = [new_w]

for epoch in range(1, epochs + 1):
	prev_w = history_w[-1]

	updated_w = update_weights(X, y_true, prev_w, alpha, learning_rate)

	history_w.append(updated_w)

	if abs(updated_w - prev_w) < epsilon:
		print(f"Остановка на эпохе {epoch}")
		break

	if epoch % 10 == 0:
		current_loss = loss_with_l2(y_true, y_hat, [updated_w])
		print(f"Эпоха {epoch}: w = {updated_w:.6f}, Loss+L2 = {current_loss:.6f}")

final_w = history_w[-1]
print(f"\nИтоговый вес после обучения: {final_w:.6f}")
final_loss = loss_with_l2(y_true, y_hat, [final_w])
print(f"Итоговый Loss+L2: {final_loss:.6f}")
