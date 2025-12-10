import numpy as np
import matplotlib.pyplot as plt

# Мої дані з варіанту №18
lambda_wave = 3.1        # довжина хвилі, см
a_p = 16                 # ширина розкриву (площина H), см
b_p = 10                 # висота розкриву (площина E), см

# sinc-функція, щоб не було помилки при x=0
def sinc(x):
    return np.sinc(x / np.pi)

# Елементний множитель (принцип Гюйгенса)
def huygens_term(theta_deg):
    theta_rad = np.deg2rad(theta_deg)
    return (1 + np.cos(theta_rad)) / 2

# Основна частина ДН в Е-площині
def main_E(theta_deg):
    theta_rad = np.deg2rad(theta_deg)
    arg = np.pi * b_p * np.sin(theta_rad) / lambda_wave
    return np.where(np.abs(arg) > 1e-10, sinc(arg), 1.0)

# Повна ДН в Е-площині
def F_E(theta_deg):
    return main_E(theta_deg) * huygens_term(theta_deg)

# Основна частина ДН в Н-площині
def main_H(theta_deg):
    theta_rad = np.deg2rad(theta_deg)
    arg_cos = np.pi * a_p * np.sin(theta_rad) / lambda_wave
    arg_den = 2 * a_p * np.sin(theta_rad) / lambda_wave
    cos_term = np.cos(arg_cos)
    den = 1 - arg_den**2
    return np.where(np.abs(den) > 1e-10, cos_term / den, 0.0)

# Повна ДН в Н-площині
def F_H(theta_deg):
    return main_H(theta_deg) * huygens_term(theta_deg)

# Нормування до максимуму (він завжди при θ = 0°)
def normalize(F):
    F_max = np.abs(F[theta >= 0][0])
    return np.abs(F) / F_max

# Кут від 0 до 90° з кроком 0.1°
theta = np.arange(0, 90.1, 0.1)

# Обчислення всіх кривих для Е-площини
main_E_values = main_E(theta)
huygens_values = huygens_term(theta)
F_E_values = F_E(theta)
F_E_norm = normalize(F_E_values)

# Те саме для Н-площини
main_H_values = main_H(theta)
F_H_values = F_H(theta)
F_H_norm = normalize(F_H_values)

# Нульові кути в Е-площині
zeros_E = np.arcsin(np.arange(1, 5) * lambda_wave / b_p)
zeros_E_deg = np.rad2deg(zeros_E[np.sin(zeros_E) <= 1])

# Нульові кути в Н-площині
zeros_H = np.arcsin((2 * np.arange(1, 4) + 1) * lambda_wave / (2 * a_p))
zeros_H_deg = np.rad2deg(zeros_H[np.sin(zeros_H) <= 1])

# Максимуми бокових пелюстків в Е-площині
maxes_E = np.arcsin((2 * np.arange(1, 5) + 1) * lambda_wave / (2 * b_p))
maxes_E_deg = np.rad2deg(maxes_E[np.sin(maxes_E) <= 1])

# Максимуми бокових пелюстків в Н-площині
maxes_H = np.arcsin((np.arange(1, 4) + 1) * lambda_wave / a_p)
maxes_H_deg = np.rad2deg(maxes_H[np.sin(maxes_H) <= 1])

# Теоретичні рівні бокових пелюстків
sidelobe_levels_E = [2 / ((2 * n + 1) * np.pi) for n in range(1, len(maxes_E_deg) + 1)]
sidelobe_levels_H = [1 / (1 - (2 * (n + 1))**2) for n in range(1, len(maxes_H_deg) + 1)]

# Графік для Е-площини
fig_E, ax_E = plt.subplots(figsize=(8, 6))
ax_E.plot(theta, F_E_norm, 'k-', label='|F_E(θ)|')
ax_E.plot(theta, main_E_values / np.max(main_E_values), 'k--', label='|F(θ)|')
ax_E.plot(theta, huygens_values / np.max(huygens_values), 'k-.', label='F_e(θ)')
ax_E.set_xlabel('θ°')
ax_E.set_ylabel('|F_E(θ)|')
ax_E.set_title('ДС рупора 10×16×λ=3.1 см, Е-площина, варіант 18')
ax_E.set_xlim(0, 90)
ax_E.set_ylim(0, 1.1)
ax_E.grid(False)

# Лінія -3 дБ і рівні бокових пелюстків
ax_E.axhline(0.5, color='k', linestyle='--', linewidth=0.5)
for lvl in sidelobe_levels_E[:len(maxes_E_deg)]:
    ax_E.axhline(lvl, color='k', linestyle='--', linewidth=0.5)

# Червоні крапки — нулі, сині — максимуми бокових
for z in zeros_E_deg:
    ax_E.plot(z, 0, 'ro')
    ax_E.text(z, -0.05, f'θ_{int(np.round(z)):02d}', ha='center', va='top', fontsize=8)
for m in maxes_E_deg:
    ax_E.plot(m, np.interp(m, theta, F_E_norm), 'bo')

# Легенда + підписи кутів
legend_labels = ['|F_E(θ)|', '|F(θ)|', 'F_e(θ)']
legend_points = [f'θ_{int(np.round(z)):02d}={z:.2f}°' for z in zeros_E_deg] + [f'θ_m{int(np.round(m)):02d}={m:.2f}°' for m in maxes_E_deg]
ax_E.legend(legend_labels + legend_points, loc='upper right')

fig_E.savefig('APiK_lab3_var18__E_plane.png')

# Графік для Н-площини
fig_H, ax_H = plt.subplots(figsize=(8, 6))
ax_H.plot(theta, F_H_norm, 'k-', label='|F_H(θ)|')
ax_H.plot(theta, main_H_values / np.max(main_H_values), 'k--', label='|F(θ)|')
ax_H.plot(theta, huygens_values / np.max(huygens_values), 'k-.', label='F_h(θ)')
ax_H.set_xlabel('θ°')
ax_H.set_ylabel('|F_H(θ)|')
ax_H.set_title('ДС рупора 10×16×λ=3.1 см, H-площина, варіант 18')
ax_H.set_xlim(0, 90)
ax_H.set_ylim(0, 1.1)
ax_H.grid(False)

ax_H.axhline(0.5, color='k', linestyle='--', linewidth=0.5)
for lvl in np.abs(sidelobe_levels_H[:len(maxes_H_deg)]):
    ax_H.axhline(lvl, color='k', linestyle='--', linewidth=0.5)

for z in zeros_H_deg:
    ax_H.plot(z, 0, 'ro')
    ax_H.text(z, -0.05, f'θ_{int(np.round(z)):02d}', ha='center', va='top', fontsize=8)
for m in maxes_H_deg:
    ax_H.plot(m, np.interp(m, theta, F_H_norm), 'bo')

legend_labels = ['|F_H(θ)|', '|F(θ)|', 'F_h(θ)']
legend_points = [f'θ_{int(np.round(z)):02d}={z:.2f}°' for z in zeros_H_deg] + [f'θ_m{int(np.round(m)):02d}={m:.2f}°' for m in maxes_H_deg]
ax_H.legend(legend_labels + legend_points, loc='upper right')

fig_H.savefig('APiK_lab3_var18_H_plane.png')

plt.show()