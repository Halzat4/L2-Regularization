import numpy as np
from sklearn.model_selection import train_test_split

# Подготовка данных
np.random.seed(42)
N = 60
X = np.linspace(-3, 3, N)
y_true = np.sin(X) * 2 + 0.5
noise = np.random.normal(0, 0.6, N)
y_hat = y_true + noise

# Класс для расчета функций потерь
class LinearMetrics:
    @staticmethod
    def loss(y_true, y_hat):
        s = 0
        for i in range(2): 
            s += 1/2 * (y_true[i] - y_hat[i]) ** 2
        return s

    @staticmethod
    def loss_with_l2(y_true, y_hat, weights, alpha=0.1):
        z = 0
        for i in range(2):
            z += 1/2 * (y_true[i] - y_hat[i]) ** 2
        
        l2_penalty = sum(w ** 2 for w in weights)
        return z + alpha * l2_penalty

# Класс модели
class LinearRegressionL2:
    def __init__(self, alpha=0.1, lr=0.1, weights=[0.5]):
        self.alpha = alpha
        self.lr = lr
        self.weights = weights
        self.history_w = [weights[0]]
        self.metrics = LinearMetrics()

    def update_weights(self, X, y_true, y_hat):
        dw = 0
        w = self.weights[0]
        for i in range(2):
            dw += (y_hat[i] - y_true[i]) * X[i]
        
        dw = dw + (2 * self.alpha * w)
        new_w = w - self.lr * dw
        self.weights = [new_w]
        return new_w

    def train(self, X, y_true, y_hat, epochs=1000, epsilon=1e-6):
        for epoch in range(1, epochs + 1):
            prev_w = self.weights[0]
            updated_w = self.update_weights(X, y_true, y_hat)
            self.history_w.append(updated_w)

            # Точка останова
            if abs(updated_w - prev_w) < epsilon:
                print(f"Остановка на эпохе {epoch}")
                break

            if epoch % 10 == 0:
                current_loss = self.metrics.loss_with_l2(y_true, y_hat, self.weights, self.alpha)
                print(f"Эпоха {epoch}: w = {updated_w:.6f}, Loss+L2 = {current_loss:.6f}")

# --- Использование ---

# Инициализация и первый запуск
model = LinearRegressionL2(alpha=0.1, lr=0.1, weights=[0.5])
print(f"Начальный Loss: {model.metrics.loss(y_true, y_hat)}")

# Обучение
model.train(X, y_true, y_hat)

final_w = model.weights[0]
print(f"\nИтоговый вес после обучения: {final_w:.6f}")
final_loss = model.metrics.loss_with_l2(y_true, y_hat, [final_w], alpha=0.1)
print(f"Итоговый Loss+L2: {final_loss:.6f}")
