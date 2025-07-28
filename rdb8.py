import torch
from utils.SCN_torch import SCN, SCN_batch
from data.data import RDB8
import pickle
from pathlib import Path


saves_dir = 'saves_rdb8'


torch.set_grad_enabled(False)
Path(saves_dir).mkdir(parents=True, exist_ok=True)

X, Y = RDB8(100, draw=False)

print(X.shape, Y.shape)

datatype = torch.float64
params = dict()
params['max_neurons'] = 200
params['Lambdas'] = [30, 50, 100, 150, 200, 250]
params['reconfig_number'] = 100

N = 30

print('Pseudoinverse')
for k in range(N):
    _, stats = SCN(X, Y, params, lsq='pseudoinverse', show=False, datatype=datatype)
    print(k, sum(stats['inv_time']), sum(stats['total_time']), stats['rmse'][-1])
    with open(f'{saves_dir}/rdb8_torch_pseudoinverse_{k}.pickle', 'wb') as handle:
        pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
print()

print('QR update')
for k in range(N):
    _, stats = SCN(X, Y, params, lsq='qr_update', show=False, datatype=datatype)
    print(k, sum(stats['inv_time']), sum(stats['total_time']), stats['rmse'][-1])
    with open(f'{saves_dir}/rdb8_torch_qr_update_{k}.pickle', 'wb') as handle:
        pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
print()

print('Batch QR update')
batch_size = 9000
for k in range(N):
    batch_increment = (X.shape[0] - batch_size) // params['max_neurons']
    _, stats = SCN_batch(X, Y, params, lsq='qr_update', show=False, init_batch_size=batch_size, batch_increment=batch_increment)
    print(k, sum(stats['inv_time']), sum(stats['total_time']), stats['rmse'][-1])
    with open(f'{saves_dir}/rdb8_update_batch_{batch_size}_{k}.pickle', 'wb') as handle:
        pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
print()
