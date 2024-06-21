import numpy as np 
import scipy.stats as stats


def DPGDPure(x, y, xm, ym, n, eps, xnew, T = 80):
    OLS_25=0.325
    OLS_75=0.575
    
    est = np.zeros((T + 1, 2))  # Khởi tạo mảng với số 0
    est[0] = [OLS_25, OLS_75]
    delta = []
    
    for t in range(T):
        eps_t = eps / T
        delta_t_sum = np.array([0.0, 0.0])

        for i in range(n):
            y_est_i = 2 * (est[t][0] * (3/4 - x[i]) + est[t][1] * (x[i] - 1/4))
            delta_t_sum[0] += 2 * (y_est_i - y[i]) * (3/4 - x[i])
            delta_t_sum[1] += 2 * (y_est_i - y[i]) * (x[i] - 1/4)

        delta_t = delta_t_sum + np.random.laplace(0, 4 * 1/eps_t, 2)
        delta.append(delta_t)
        gamma_t = 1 / np.sqrt(np.sum(np.array(delta) ** 2))

        est[t + 1] = est[t] - gamma_t * delta_t_sum

    sig = np.array([0.0, 0.0])
    for f in range(T // 2, T):
        sig += est[f]

    return (2 / T) * sig


def DPGDzCDP(x, y, xm, ym, n, eps, xnew, T = 80):
    OLS_25=0.325
    OLS_75=0.575
    est = np.zeros((T + 1, 2))  # Khởi tạo mảng với số 0
    est[0] = [OLS_25, OLS_75]
    delta = []
    
    for t in range(T):
        rho_t = eps / T
        delta_t_sum = np.array([0.0, 0.0])

        for i in range(n):
            y_est_i = 2 * (est[t][0] * (3/4 - x[i]) + est[t][1] * (x[i] - 1/4))
            delta_t_sum[0] += 2 * (y_est_i - y[i]) * (3/4 - x[i])
            delta_t_sum[1] += 2 * (y_est_i - y[i]) * (x[i] - 1/4)

        delta_t = delta_t_sum + np.random.normal(0, (2 * 1/ np.sqrt(rho_t))**2, 2)
        delta.append(delta_t)
        gamma_t = 1 / np.sqrt(np.sum(np.array(delta) ** 2))

        est[t + 1] = est[t] - gamma_t * delta_t_sum

    sig = np.array([0.0, 0.0])
    for f in range(T // 2, T):
        sig += est[f]

    return (2 / T) * sig


def DPGDApprox(x, y, xm, ym, n, eps, xnew, T = 80):
    OLS_25=0.325
    OLS_75=0.575
    est = np.zeros((T + 1, 2))  # Khởi tạo mảng với số 0
    est[0] = [OLS_25, OLS_75]
    delta = []
    
    for t in range(T):
        rho_t = eps / T
        delta_t_sum = np.array([0.0, 0.0])

        for i in range(n):
            y_est_i = 2 * (est[t][0] * (3/4 - x[i]) + est[t][1] * (x[i] - 1/4))
            delta_t_sum[0] += 2 * (y_est_i - y[i]) * (3/4 - x[i])
            delta_t_sum[1] += 2 * (y_est_i - y[i]) * (x[i] - 1/4)

        log_term = np.log(np.sqrt(np.pi * rho_t) / 2)
        if log_term < 0:
            log_term = 0 

        delta_t = delta_t_sum + np.random.normal(0, rho_t + np.sqrt(4 * rho_t * log_term), 2)
        delta.append(delta_t)
        gamma_t = 1 / np.sqrt(np.sum(np.array(delta) ** 2))

        est[t + 1] = est[t] - gamma_t * delta_t_sum

    sig = np.array([0.0, 0.0])
    for f in range(T // 2, T):
        sig += est[f]

    return (2 / T) * sig