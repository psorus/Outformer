from torch.utils.data import Dataset
import pickle


def load_pickle(file_path):
    with open(file_path, 'rb') as handle:
        inst = pickle.load(handle)
    return inst


class EpochDataset(Dataset):
    def __init__(self, batch_size, seq_len, steps_per_epoch, hyperparameters, reuse_data_every_n, max_model_dim,
                 max_num_cluster, get_batch_method, rank, num_device, training, single_eval_pos_gen, data_path):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.steps_per_epoch = steps_per_epoch
        self.hyperparameters = hyperparameters
        self.reuse_data_every_n = reuse_data_every_n
        self.max_model_dim = max_model_dim
        self.max_num_cluster = max_num_cluster
        self.current_epoch = 0

        self.get_batch_method = get_batch_method
        self.rank = rank
        self.num_device = num_device
        self.training = training
        self.data_path = data_path
        self.single_eval_pos_gen = single_eval_pos_gen

        self.in_data = None if data_path is None else load_pickle(file_path=f'{data_path}/epoch0/in.pickle')
        self.la_data = None if data_path is None else load_pickle(file_path=f'{data_path}/epoch0/la.pickle')

    def set_epoch_and_data(self, epoch, data_dict=None):
        """
        Update the dataset to use data from the specified epoch.
        """
        self.current_epoch = epoch
        print('setting current epoch to:', epoch)

        if data_dict is not None:  # reuse saved data
            print('new data loaded...')
            self.in_data = data_dict['in']
            self.la_data = data_dict['la']

    def set_training_mode(self, training):
        self.training = training

    def set_rank(self, rank):
        self.rank = rank
        print(f'rank is successfully set to {rank} out of {self.num_device} devices')

    def __len__(self):
        return self.steps_per_epoch * self.batch_size

    def __getitem__(self, idx):
        return {'in': self.in_data[idx], 'la': self.la_data[idx]}

    def prior_batch_collate_fn(self, batch_list):
        batch = self.get_batch_method(list_of_data=batch_list, seq_len=self.seq_len,
                                      hyperparameters=self.hyperparameters,
                                      training=self.training,
                                      single_eval_pos=self.single_eval_pos_gen.generate(), )
        return batch
