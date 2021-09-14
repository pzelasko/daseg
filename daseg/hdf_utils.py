
import scipy.io as sio
import numpy as np
import scipy.sparse as sparse
import h5py


def hdf5_write_data_labels(data, labels, path, prefix='train'):
    # for sparse data, writing
    f =  h5py.File(path, 'a')
    X_dset = f.create_dataset(prefix + '_split', shape=data.shape, dtype='f', fillvalue=0, compression='gzip', compression_opts=9)
    X_dset[:] = data
    Y_dset = f.create_dataset(prefix + '_labels', labels.shape, dtype='i')
    Y_dset[:] = labels
    f.close()


def hdf5_write_compression(data, path, dataset_name):
    # for sparse data, writing
    f =  h5py.File(path, 'a')
    X_dset = f.create_dataset(dataset_name, shape=data.shape, dtype='f', fillvalue=0, compression='gzip', compression_opts=9)
    X_dset[:] = data
    f.close()


def hdf5_write(data, path, dataset_name):
    # for sparse data, writing
    f =  h5py.File(path, 'a')
    if not dataset_name in f:
        X_dset = f.create_dataset(dataset_name, shape=data.shape, dtype='f')
        X_dset[:] = data
    else:
        print(f'{dataset_name} already exists in {path} so skipping writing')
    f.close()


def hdf5_read_Siamese(path, dataset, idx1, idx2):
    # reading
    f_read = h5py.File(path, 'r')
    data_1 = f_read[dataset][idx1]
    data_2 = f_read[dataset][idx2]
    f_read.close()

    return data_1, data_2


def hdf5_read(path, dataset, idx1):
    # reading
    f_read = h5py.File(path, 'r')
    data_1 = f_read[dataset][idx1]
    f_read.close()

    return data_1


def hdf5_read_range(path, dataset, start, end):
    # reading
    f_read = h5py.File(path, 'r')
    data_1 = f_read[dataset][start:end]
    f_read.close()

    return data_1

