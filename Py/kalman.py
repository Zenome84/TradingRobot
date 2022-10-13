
# import numpy as np
# import numpy.linalg as la
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression

# data = np.load(open('./Py/data5.npy', 'rb'))
# prices = data[:,4]
# dt = 1
# dN = 60

# p_var = 0
# for n in range(dN):
#     p_var += np.diff(prices[n::dN]).var()/dN
# m_var = 0
# for n in range(len(prices)-dN):
#     m_var += prices[n:n+dN].var()/(len(prices)-dN)

# A = np.array(
#     [[1., dt],
#      [0., 1.]]
# )
# F = np.array([[0.5*dt**2, dt]]).T
# H = np.array([[1., 0]])

# W = np.array(
#     [[p_var, 0],
#      [0., m_var]]
# )
# V = np.eye(1) * p_var
# Vi = la.inv(V)

# x = np.array([[prices[1], prices[1]-prices[0]]]).T
# d = np.array([[0.]]).T
# P = np.array(
#     [[p_var, 0.],
#      [0., m_var]]
# )

# kalman = [x]
# control = []
# for n in range(2, len(prices)):
#     xH = A @ x
#     PH = A @ P @ A.T + W
#     PHi = la.inv(PH)
#     FPF = la.inv(F.T @ PHi @ F)

#     # PXk1k1 = la.inv(PHi + H.T @ la.inv(V) @ H - PHi @ F @ FPF @ F.T @ PHi)
#     # PDkk1 = la.inv(F.T @ H.T @ la.inv(V + H @ PH @ H.T) @ H @ F)
#     # PXDk1k1 = PXk1k1 @ PHi @ F @ FPF
#     # PDXk1k1 = PDkk1 @ F.T @ PHi @ la.inv(PHi + H.T @ la.inv(V) @ H)
    
#     PD = la.inv(F.T @ H.T @ la.inv(V + H @ PH @ H.T) @ H @ F)
#     PDX = PD @ F.T @ PHi @ la.inv(PHi + H.T @ Vi @ H)
#     P = la.inv(PHi + H.T @ Vi @ H) + PDX.T @ la.inv(PD) @ PDX

#     y = prices[n]
#     delta = y - H @ xH
#     x = xH + P @ H.T @ Vi @ delta
#     d = PDX @ H.T @ Vi @ delta

#     kalman.append(x)
#     control.append(d)



#     # predict.append(x)

#     # P0 = F @ P @ F.T + Q
    
#     # y = z - H @ x0
#     # S = H @ P0 @ H.T + R
#     # K = P @ H.T @ np.linalg.inv(S)

#     # x = x0 + K @ y
#     # P = (np.eye(2) - K @ H) @ P0


# kalman = np.squeeze(np.array(kalman))
# control = np.squeeze(np.array(control))

# plt.plot(prices[1:])
# plt.plot(kalman[:,0])
# # plt.plot(control)
# plt.show()

# exit(0)

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = np.load(open('./Py/data5.npy', 'rb'))
prices = data[:,4]
dt = 1
dN = 30

p_var = 0
for n in range(dN):
    p_var += np.diff(prices[n::dN]).var()/dN
m_var = 0
for n in range(len(prices)-dN):
    m_var += prices[n:n+dN].var()/(len(prices)-dN)

A = np.array(
    [[1., dt, 0.5*dt**2],
     [0., 1., dt],
     [0., 0., 1.]]
)
# F = np.array([[0.5*dt**2, dt, 1.]]).T
H = np.array([[1., 0, 0]])

W = np.array(
    [[1.0, 0, 0],
     [0., 2.0, 0],
     [0., 0., 4.0]]
)*p_var/dN
V = np.eye(1) * p_var * dN
Vi = la.inv(V)

x = np.array([[prices[1], prices[1]-prices[0], 0.]]).T
# d = np.array([[0.]]).T
P = np.array(
    [[p_var, 0., 0.],
     [0., m_var, 0.],
     [0., 0., 1.]]
)

kalman = [x]
control = []
for n in range(2, len(prices)):
    # xH = A @ x
    # PH = A @ P @ A.T + W
    # PHi = la.inv(PH)
    # FPF = la.inv(F.T @ PHi @ F)

    # # PXk1k1 = la.inv(PHi + H.T @ la.inv(V) @ H - PHi @ F @ FPF @ F.T @ PHi)
    # # PDkk1 = la.inv(F.T @ H.T @ la.inv(V + H @ PH @ H.T) @ H @ F)
    # # PXDk1k1 = PXk1k1 @ PHi @ F @ FPF
    # # PDXk1k1 = PDkk1 @ F.T @ PHi @ la.inv(PHi + H.T @ la.inv(V) @ H)
    
    # PD = la.inv(F.T @ H.T @ la.inv(V + H @ PH @ H.T) @ H @ F)
    # PDX = PD @ F.T @ PHi @ la.inv(PHi + H.T @ Vi @ H)
    # P = la.inv(PHi + H.T @ Vi @ H) + PDX.T @ la.inv(PD) @ PDX

    # y = prices[n]
    # delta = y - H @ xH
    # x = xH + P @ H.T @ Vi @ delta
    # d = PDX @ H.T @ Vi @ delta

    # kalman.append(x)
    # control.append(d)



    # predict.append(x0)

    x0 = A @ x
    P0 = A @ P @ A.T + W
    
    y = prices[n] - H @ x0
    S = H @ P0 @ H.T + V
    K = P @ H.T @ np.linalg.inv(S)

    x = x0 + K @ y
    P = (np.eye(3) - K @ H) @ P0
    kalman.append(x)


kalman = np.squeeze(np.array(kalman))
# control = np.squeeze(np.array(control))

plt.plot(prices[1:])
plt.plot(kalman[:,0])
# plt.plot(control)
plt.show()

exit(0)
