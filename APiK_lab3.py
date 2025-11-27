import numpy as np
import matplotlib.pyplot as plt
import math

# --------------------------
# Вхідні дані (варіант 18)
# --------------------------
lam = 0.031  # 3.1 см → 0.031 м
ap = 0.10    # 10 см → 0.10 м ( H-площина )
bp = 0.16    # 16 см → 0.16 м ( E-площина )
k = 2 * np.pi / lam

print("λ =", lam, "(м)")
print("k =", k, "(рад/м)")
print("a =", ap, "м")
print("b =", bp, "м")

# --------------------------
# Розрахунок ДС
# --------------------------

FH = []
FE = []
angles = []

SGP_H = 0
SGP_E = 0
val_H = 0
val_E = 0

for theta in np.arange(0.0001, np.pi/2, 0.0005):

    FH_val = abs(np.sin((k * ap / 2) * np.sin(theta)) /
                 ((k * ap / 2) * np.sin(theta)))

    FE_val = abs(np.sin((k * bp / 2) * np.sin(theta)) /
                 ((k * bp / 2) * np.sin(theta)))

    FH.append(FH_val)
    FE.append(FE_val)
    angles.append(np.degrees(theta))

    # Пошук ширини головної пелюстки
    if 0.707 < FH_val < 0.708:
        SGP_H = 2 * np.degrees(theta)
        val_H = FH_val

    if 0.707 < FE_val < 0.708:
        SGP_E = 2 * np.degrees(theta)
        val_E = FE_val

# --------------------------
# Вивід результатів
# --------------------------
print("Ширина головної пелюстки в H-площині =", round(SGP_H, 2), "°")
print("Ширина головної пелюстки в E-площині =", round(SGP_E, 2), "°")

# --------------------------
# Побудова графіка
# --------------------------
plt.figure(figsize=(10, 6))
plt.plot(angles, FH, label="FH(θ) – H-площина", linewidth=1)
plt.plot(angles, FE, label="FE(θ) – E-площина", linewidth=1)

# Позначки ШГП
plt.plot(SGP_H/2, val_H, 'ro', label="ШГП H")
plt.plot(SGP_E/2, val_E, 'go', label="ШГП E")

plt.grid(True)
plt.xlabel("θ (градуси)")
plt.ylabel("Нормована ДС")
plt.title("Нормовані ДС рупорної антени (варіант 18)")
plt.legend()
plt.xlim(0, 90)
plt.ylim(0, 1.05)

plt.show()
