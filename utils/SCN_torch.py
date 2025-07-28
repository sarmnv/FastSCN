import torch
import numpy as np
from time import time
from scipy.linalg import qr_insert, solve_triangular
import scipy


torch.set_grad_enabled(False)


def activation_torch(x):
    return torch.special.expit(x)


def rmse_function_torch(x):
    return torch.sqrt(torch.mean(x ** 2))


def orthogonalize(A, b):
    r1 = (A.T @ b)
    t = b - A @ r1
    r2 = torch.norm(t, dim=0)
    return t / r2


def SCN(features, labels, params, lsq='pseudoinverse', datatype=torch.float64, show=True, cond=False):

    X = features.to(datatype)
    T = labels.to(datatype)

    d = X.shape[1]
    e = T.squeeze()

    if T.ndim == 1:
        m = 1
        T = T.unsqueeze(1)
    else:
        m = T.shape[1]

    stats = {'total_time': [], 'activation_time': [], 'inv_time': [], 'rmse': [], 'cond_number': []}
    H = torch.empty((X.shape[0], params['max_neurons']), dtype=datatype)
    W = torch.empty((d, 0), dtype=datatype)
    b = torch.empty(0, dtype=datatype)
    rmse_prev = torch.inf
    beta_prev = 0
    for k in range(params['max_neurons']):

        start_time = time()

        # Генерируем случайным образом набор весов
        W_random = []
        b_random = []
        for L in params['Lambdas']:
            WL = L * (2 * torch.rand(d, params['reconfig_number'], dtype=datatype) - 1)
            bL = L * (2 * torch.rand(1, params['reconfig_number'], dtype=datatype) - 1)
            W_random.append(WL)
            b_random.append(bL)
        W_random = torch.hstack(W_random)
        b_random = torch.hstack(b_random)

        # Находим активацию
        activation_time_s = time()
        h = activation_torch(X @ W_random + b_random)
        activation_time = time() - activation_time_s

        # находим лучшие веса
        v_values = (e.T @ h) ** 2 / torch.sum(h * h, dim=0)
        if m == 1:
            best_idx = torch.argmax(v_values)
        else:
            v_values_tmp = torch.mean(v_values, dim=0)
            best_idx = torch.argmax(v_values_tmp)

        h_c = h[:, best_idx]
        W_c = W_random[:, best_idx].unsqueeze(dim=-1)
        b_c = b_random[:, best_idx]
        H[:, k] = h_c

        # Находим выходной вес
        inv_time_s = time()

        if lsq == 'pseudoinverse':
            beta = torch.linalg.inv(H[:, 0: (k + 1)].T @ H[:, 0: (k + 1)]) @ H[:, 0: (k + 1)].T @ T
        elif lsq == 'lstq':
            beta = torch.linalg.lstsq(H[:, 0: (k + 1)], T).solution
        elif lsq == 'qr':
            Q, R = torch.linalg.qr(H[:, 0: (k + 1)])
            beta = torch.linalg.solve_triangular(R, Q.T @ T, upper=True)
        elif lsq == 'qr_update':
            Ns = 2
            if k < Ns:
                beta = torch.linalg.inv(H[:, 0: (k + 1)].T @ H[:, 0: (k + 1)]) @ H[:, 0: (k + 1)].T @ T
            elif k == Ns:
                Q, R = torch.linalg.qr(H[:, 0: (k + 1)])
                QT = Q.T @ T
                beta = torch.linalg.solve_triangular(R, QT, upper=True)
            else:
                r1 = (Q.T @ h_c).squeeze()
                t = h_c.squeeze() - Q @ r1
                r2 = torch.linalg.norm(t)
                q = t / r2

                Q.data = torch.hstack([Q, q.unsqueeze(1)])
                r = torch.hstack([R, r1.unsqueeze(1)])
                r3 = torch.cat([torch.zeros((k, 1)), r2.unsqueeze(0).unsqueeze(0)])
                R.data = torch.vstack([r, r3.T])

                QT.data = torch.vstack([QT, (Q[:, -1] @ T).unsqueeze(1).T])
                beta = torch.linalg.solve_triangular(R, QT, upper=True)

        inv_time = time() - inv_time_s

        # Находим выход сети
        y = H[:, 0: (k + 1)] @ beta

        # Рассчитываем вектор ошибки
        e = T.squeeze() - y.squeeze()
        rmse = rmse_function_torch(e)

        if show:
            print(f'{k}: {rmse}')

        if rmse >= rmse_prev:
            return (W, b, beta_prev), stats

        rmse_prev = rmse
        beta_prev = beta
        W = torch.concatenate((W, W_c), dim=-1)
        b = torch.concatenate((b, b_c), dim=0)

        total_time = time() - start_time

        stats['total_time'].append(total_time)
        stats['activation_time'].append(activation_time)
        stats['inv_time'].append(inv_time)
        stats['rmse'].append(rmse)

        if cond:
            stats['cond_number'].append(torch.linalg.cond(H[:, 0: k]))

    return (W, b, beta), stats


def SCN_batch(features, labels, params, lsq='pseudoinverse', init_batch_size=None, batch_increment=0, datatype=torch.float64, show=True, cond=False):

    X = features.to(datatype)
    T = labels.to(datatype)

    d = X.shape[1]

    if not init_batch_size:
        init_batch_size = X.shape[0]

    if T.ndim == 1:
        m = 1
        T = T.unsqueeze(1)
    else:
        m = T.shape[1]

    stats = {'total_time': [], 'activation_time': [], 'inv_time': [], 'rmse': [], 'cond_number': []}
    H = torch.empty((X.shape[0], params['max_neurons']), dtype=datatype)
    W = torch.empty((d, 0), dtype=datatype)
    b = torch.empty(0, dtype=datatype)
    batch_size = init_batch_size
    for k in range(params['max_neurons']):

        start_time = time()

        indices = torch.randperm(len(T))[:batch_size]
        Xk = X[indices, :]
        Tk = T[indices, :]

        if k == 0:
            ek = Tk.squeeze()
        else:
            ek = Tk.squeeze() - y[indices, :].squeeze()

        # Генерируем случайным образом набор весов
        W_random = []
        b_random = []
        for L in params['Lambdas']:
            WL = L * (2 * torch.rand(d, params['reconfig_number'], dtype=datatype) - 1)
            bL = L * (2 * torch.rand(1, params['reconfig_number'], dtype=datatype) - 1)
            W_random.append(WL)
            b_random.append(bL)
        W_random = torch.hstack(W_random)
        b_random = torch.hstack(b_random)

        # Находим активацию
        activation_time_s = time()
        h = activation_torch(Xk @ W_random + b_random)
        activation_time = time() - activation_time_s

        # находим лучшие веса
        v_values = (ek.T @ h) ** 2 / torch.sum(h * h, dim=0)
        if m == 1:
            best_idx = torch.argmax(v_values)
        else:
            v_values_tmp = torch.mean(v_values, dim=0)
            best_idx = torch.argmax(v_values_tmp)

        W_c = W_random[:, best_idx].unsqueeze(dim=-1)
        b_c = b_random[:, best_idx]
        h_c = activation_torch(X @ W_c + b_c).squeeze()
        H[:, k] = h_c

        # Находим выходной вес
        inv_time_s = time()

        if lsq == 'pseudoinverse':
            beta = torch.linalg.inv(H[:, 0: (k + 1)].T @ H[:, 0: (k + 1)]) @ H[:, 0: (k + 1)].T @ T
        if lsq == 'pseudoinverse_batch':
            beta = torch.linalg.inv(H[indices, 0: (k + 1)].T @ H[indices, 0: (k + 1)]) @ H[indices, 0: (k + 1)].T @ Tk
        elif lsq == 'lstq':
            beta = torch.linalg.lstsq(H[:, 0: (k + 1)], T).solution
        elif lsq == 'qr':
            Q, R = torch.linalg.qr(H[:, 0: (k + 1)])
            beta = torch.linalg.solve_triangular(R, Q.T @ T, upper=True)
        elif lsq == 'qr_update':
            Ns = 2
            if k < Ns:
                beta = torch.linalg.inv(H[:, 0: (k + 1)].T @ H[:, 0: (k + 1)]) @ H[:, 0: (k + 1)].T @ T
            elif k == Ns:
                Q, R = torch.linalg.qr(H[:, 0: (k + 1)])
                QT = Q.T @ T
                beta = torch.linalg.solve_triangular(R, QT, upper=True)
            else:
                r1 = (Q.T @ h_c).squeeze()
                t = h_c.squeeze() - Q @ r1
                r2 = t.dot(t) / torch.linalg.norm(t)
                q = t / r2

                Q.data = torch.hstack([Q, q.unsqueeze(1)])
                r = torch.hstack([R, r1.unsqueeze(1)])
                r3 = torch.cat([torch.zeros((k, 1)), r2.unsqueeze(0).unsqueeze(0)])
                R.data = torch.vstack([r, r3.T])

                QT.data = torch.vstack([QT, (Q[:, -1] @ T).unsqueeze(1).T])
                beta = torch.linalg.solve_triangular(R, QT, upper=True)
        elif lsq == 'qr_update_batch':
            pass

        inv_time = time() - inv_time_s

        # Находим выход сети
        y = H[:, 0: (k + 1)] @ beta

        # Рассчитываем вектор ошибки
        e = T.squeeze() - y.squeeze()
        rmse = rmse_function_torch(e)

        if show:
            print(f'{k}: {rmse}')

        W = torch.concatenate((W, W_c), dim=-1)
        b = torch.concatenate((b, b_c), dim=0)

        batch_size += batch_increment

        total_time = time() - start_time

        stats['total_time'].append(total_time)
        stats['activation_time'].append(activation_time)
        stats['inv_time'].append(inv_time)
        stats['rmse'].append(rmse)

        if cond:
            stats['cond_number'].append(torch.linalg.cond(H[:, 0: (k + 1)]))

    return (W, b, beta), stats


def OSCN(features, labels, params, datatype=torch.float64, show=True, cond=False):

    def get_best_weights(activation, error):
        v_values = (error.T @ activation) ** 2
        v_values = torch.nan_to_num(v_values, nan=-1)
        if m == 1:
            best_idx = torch.argmax(v_values)
        else:
            v_values_tmp = torch.mean(v_values, dim=0)
            best_idx = torch.argmax(v_values_tmp)
        return best_idx

    X = features.to(datatype)
    T = labels.to(datatype)

    d = X.shape[1]
    e = T.squeeze()

    if T.ndim == 1:
        m = 1
        T = T.unsqueeze(1)
    else:
        m = T.shape[1]

    stats = {'total_time': [], 'activation_time': [], 'ort_time': [], 'rmse': [], 'cond_number': []}
    H = torch.empty((X.shape[0], params['max_neurons']), dtype=datatype)
    W = torch.empty((d, 0), dtype=datatype)
    b = torch.empty(0, dtype=datatype)
    for k in range(0, params['max_neurons']):

        start_time = time()

        # Генерируем случайным образом набор весов
        W_random = []
        b_random = []
        for L in params['Lambdas']:
            WL = L * (2 * torch.rand(d, params['reconfig_number'], dtype=datatype) - 1)
            bL = L * (2 * torch.rand(1, params['reconfig_number'], dtype=datatype) - 1)
            W_random.append(WL)
            b_random.append(bL)
        W_random = torch.hstack(W_random)
        b_random = torch.hstack(b_random)

        # Находим активацию
        activation_time_s = time()
        h = activation_torch(X @ W_random + b_random)
        activation_time = time() - activation_time_s

        ort_time_s = time()
        if k == 0:
            v = h
        else:
            v = orthogonalize(H[:, :k], h)
        ort_time = time() - ort_time_s

        best_idx = get_best_weights(v, e)
        h_c = v[:, best_idx]

        if k == 0:
            h_c = h_c / torch.norm(h_c)
            beta = e.T @ h_c
            beta = beta.unsqueeze(0)
        else:
            beta_c = e.T @ h_c
            beta = torch.concatenate((beta, beta_c.unsqueeze(0)), dim=0)

        H[:, k] = h_c

        W_c = W_random[:, best_idx].unsqueeze(dim=-1)
        b_c = b_random[:, best_idx]
        W = torch.concatenate((W, W_c), dim=-1)
        b = torch.concatenate((b, b_c), dim=0)

        # Рассчитываем вектор ошибки
        if m == 1:
            e = e - h_c * beta[-1]
        else:
            e = e - h_c.unsqueeze(1) * beta[-1, None, :]

        rmse = rmse_function_torch(e)
        if show:
            print(f'{k}: {rmse}')

        total_time = time() - start_time

        stats['total_time'].append(total_time)
        stats['activation_time'].append(activation_time)
        stats['ort_time'].append(ort_time)
        stats['rmse'].append(rmse)

        if cond:
            stats['cond_number'].append(torch.linalg.cond(H[:, 0: k]))

    # y_res = H @ beta

    H = activation_torch(X @ W + b)
    beta = torch.linalg.lstsq(H, T).solution

    # beta2 = torch.linalg.pinv(H) @ y_res

    output = H @ beta
    e = output - T
    print(rmse_function_torch(e))

    return (W, b, beta), stats
