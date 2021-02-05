import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

f = lambda t, Y: (Y[1], - 24 * Y[0] - 11 * Y[1])  # Przekształcona funkcja

ekstremum = lambda t, Y: Y[1]  # Znajdowanie miejsca zerowego y'(t)

y_analitic = lambda t: 1.4 * np.exp(-8 * t) - 1.4 * np.exp(-3 * t)  # Obliczenia na githubie

t_extreme_analitic = -np.log(3 / 8) / 5  # Obliczenia na githubie

y0 = [0, -7]  # y(0) = 0, y'(0) = -7
# t = [0,1]
tp = 0
tk = 1
t = np.linspace(tp, tk, 51)  # 21 do wyników, 51 do wykresu
y = solve_ivp(f, [tp, tk], y0, t_eval=t, dense_output=True, events=ekstremum)  # Domyślna metoda to RK45
y_analitic_points = [y_analitic(t_point) for t_point in t]

print(f"{'t':10s}{'y(t)':22s}")
for i in range(len(y.t)):
    print(f"{y.t[i]:1.2f} | {y.y[0][i]:6.20f}")

print("\n")

plt.plot(t, y_analitic_points)
plt.plot(y.t, y.y[0], '.')

# liczenie ekstremum
t_ekstr = y.t_events[0][0]
y_ekstr = y.y_events[0][0][0]
print("Położenie ekstremum")
print(f"y({t_ekstr}) = {y_ekstr}")
print("Ekstremum policzone analitycznie")
print(f"y({t_extreme_analitic}) = {y_analitic(t_extreme_analitic)}")
plt.plot(t_ekstr, y_ekstr, "*", markersize=9)
plt.legend(["y(t) policzone analitycznie", "y(t) policzone numerycznie", "ekstremum"])
plt.title("Wykres y(t)")
plt.xlabel("t")
plt.ylabel("y")
plt.grid()
plt.show()
