import torch.utils.data as utils
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import numpy as np
import pandas as pd
import math
import time
import warnings
import csv

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

print(torch.__version__)

def PrepareDataset(input_data, BATCH_SIZE=10, seq_len=5, pred_len=1, train_propotion=0.7, valid_propotion=0.2):
    #Prepare training and testing datasets

    time_len = input_data.shape[0]

    max_data = input_data.max().max()
    data_matrix = input_data / max_data

    data_sequences, data_labels = [], []
    for i in range(time_len - seq_len - pred_len):
        data_sequences.append(data_matrix.iloc[i:i + seq_len].values)
        data_labels.append(data_matrix.iloc[i + seq_len:i + seq_len + pred_len].values)
    data_sequences, data_labels = np.asarray(data_sequences), np.asarray(data_labels)

    # shuffle and split the dataset to training and testing datasets
    sample_size = data_sequences.shape[0]
    index = np.arange(sample_size, dtype=int)
    np.random.shuffle(index)

    train_index = int(np.floor(sample_size * train_propotion))
    valid_index = int(np.floor(sample_size * (train_propotion + valid_propotion)))

    train_data, train_label = data_sequences[:train_index], data_labels[:train_index]
    valid_data, valid_label = data_sequences[train_index:valid_index], data_labels[train_index:valid_index]
    test_data, test_label = data_sequences[valid_index:], data_labels[valid_index:]

    train_data, train_label = torch.Tensor(train_data), torch.Tensor(train_label)
    valid_data, valid_label = torch.Tensor(valid_data), torch.Tensor(valid_label)
    test_data, test_label = torch.Tensor(test_data), torch.Tensor(test_label)

    train_dataset = utils.TensorDataset(train_data, train_label)
    valid_dataset = utils.TensorDataset(valid_data, valid_label)
    test_dataset = utils.TensorDataset(test_data, test_label)

    train_dataloader = utils.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    valid_dataloader = utils.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    test_dataloader = utils.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    return train_dataloader, valid_dataloader, test_dataloader, max_data

def TrainModel(model, train_dataloader, valid_dataloader, learning_rate=1e-5, num_epochs=30, patience=10,
               min_delta=0.00001):
    inputs, labels = next(iter(train_dataloader))
    [batch_size, step_size, fea_size] = inputs.size()
    input_dim = fea_size
    hidden_dim = fea_size
    output_dim = fea_size

    model.cuda()

    loss_MSE = torch.nn.MSELoss()
    loss_L1 = torch.nn.L1Loss()

    learning_rate = 1e-6
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

    use_gpu = torch.cuda.is_available()
    #use_gpu = False

    interval = 100
    losses_train = []
    losses_valid = []
    losses_epochs_train = []
    losses_epochs_valid = []

    cur_time = time.time()
    pre_time = time.time()

    # Variables for Early Stopping
    is_best_model = 0
    patient_epoch = 0

    for epoch in range(num_epochs):
        #print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        #print('-' * 10)

        trained_number = 0

        valid_dataloader_iter = iter(valid_dataloader)

        losses_epoch_train = []
        losses_epoch_valid = []

        for data in train_dataloader:
            inputs, labels = data

            if inputs.shape[0] != batch_size:
                continue

            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            model.zero_grad()

            outputs = model(inputs)

            loss_train = loss_MSE(outputs, torch.squeeze(labels))

            losses_train.append(loss_train.data)
            losses_epoch_train.append(loss_train.data)

            optimizer.zero_grad()

            loss_train.backward()

            optimizer.step()

            # validation
            try:
                inputs_val, labels_val = next(valid_dataloader_iter)
            except StopIteration:
                valid_dataloader_iter = iter(valid_dataloader)
                inputs_val, labels_val = next(valid_dataloader_iter)

            if use_gpu:
                inputs_val, labels_val = Variable(inputs_val.cuda()), Variable(labels_val.cuda())
            else:
                inputs_val, labels_val = Variable(inputs_val), Variable(labels_val)

            outputs_val = model(inputs_val)

            loss_valid = loss_MSE(outputs_val, torch.squeeze(labels_val))
            losses_valid.append(loss_valid.data)
            losses_epoch_valid.append(loss_valid.data)

            # output
            trained_number += 1

        avg_losses_epoch_train = sum(losses_epoch_train) / float(len(losses_epoch_train))
        avg_losses_epoch_valid = sum(losses_epoch_valid) / float(len(losses_epoch_valid))
        losses_epochs_train.append(avg_losses_epoch_train)
        losses_epochs_valid.append(avg_losses_epoch_valid)

        # Early Stopping
        if epoch == 0:
            is_best_model = 1
            best_model = model
            min_loss_epoch_valid = 10000.0
            if avg_losses_epoch_valid < min_loss_epoch_valid:
                min_loss_epoch_valid = avg_losses_epoch_valid
        else:
            if min_loss_epoch_valid - avg_losses_epoch_valid > min_delta:
                is_best_model = 1
                best_model = model
                min_loss_epoch_valid = avg_losses_epoch_valid
                patient_epoch = 0
            else:
                is_best_model = 0
                patient_epoch += 1
                if patient_epoch >= patience:
                    print('Early Stopped at Epoch:', epoch)
                    break

        # Print training parameters
        cur_time = time.time()
        print('Epoch: {}, train_loss: {}, valid_loss: {}, time: {}, best model: {}'.format( \
            epoch, \
            #np.around(avg_losses_epoch_train, decimals=8), \
            avg_losses_epoch_train.cpu().numpy(), \
            #np.around(avg_losses_epoch_valid, decimals=8), \
            avg_losses_epoch_valid.cpu().numpy(), \
            np.around([cur_time - pre_time], decimals=2), \
            #[cur_time - pre_time].cpu().numpy(), \
            is_best_model))
        pre_time = cur_time
    return best_model, [losses_train, losses_valid, losses_epochs_train, losses_epochs_valid]

def TestModel(model, test_dataloader, max_speed):
    inputs, labels = next(iter(test_dataloader))
    [batch_size, step_size, fea_size] = inputs.size()

    cur_time = time.time()
    pre_time = time.time()

    use_gpu = torch.cuda.is_available()

    loss_MSE = torch.nn.MSELoss()
    loss_L1 = torch.nn.MSELoss()

    tested_batch = 0

    losses_mse = []
    losses_l1 = []

    for data in test_dataloader:
        inputs, labels = data

        if inputs.shape[0] != batch_size:
            continue

        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # rnn.loop()
        hidden = model.initHidden(batch_size)

        outputs = None
        outputs = model(inputs)

        #print(outputs[0][0])
        lbl = torch.squeeze(labels)
        #print(lbl[0][0])

        loss_MSE = torch.nn.MSELoss()
        loss_L1 = torch.nn.L1Loss()
        loss_mse = loss_MSE(outputs, torch.squeeze(labels))
        loss_l1 = loss_L1(outputs, torch.squeeze(labels))

        print(loss_MSE(outputs[:,0], lbl[:,0]))
        #print(loss_MSE(outputs[0][0],lbl[0][0]))
        #print(lbl.shape)
        #print(outputs.shape)
        #print(outputs[:,0])
        #print(outputs[:,1])

        losses_mse.append(loss_mse.cpu().data.numpy())
        losses_l1.append(loss_l1.cpu().data.numpy())

        tested_batch += 1

        #print(loss_l1)
        #print(loss_l1.data)


        if tested_batch % 1000000000 == 0:
            cur_time = time.time()
            print('Tested #: {}, loss_l1: {}, loss_mse: {}, time: {}'.format( \
                tested_batch * batch_size, \
                np.around([loss_l1.data[0]], decimals=8), \
                np.around([loss_mse.data[0]], decimals=8), \
                np.around([cur_time - pre_time], decimals=8)))
            pre_time = cur_time
    losses_l1 = np.array(losses_l1)
    losses_mse = np.array(losses_mse)
    mean_l1 = np.mean(losses_l1) * max_speed
    mean_mse = np.mean(losses_mse) * max_speed
    std_l1 = np.std(losses_l1) * max_speed
    std_mse = np.std(losses_mse) * max_speed

    print('Tested: L1_mean: {}, MSE_mean: {}, L1_std : {}, MSE_std : {}'.format(mean_l1, mean_mse, std_l1, std_mse))
    return [losses_l1, losses_mse, mean_l1, std_l1]

def NodeTestModel(model, test_dataloader, max_speed):
    inputs, labels = next(iter(test_dataloader))
    [batch_size, step_size, fea_size] = inputs.size()

    cur_time = time.time()
    pre_time = time.time()

    use_gpu = torch.cuda.is_available()

    loss_MSE = torch.nn.MSELoss()
    loss_L1 = torch.nn.MSELoss()

    tested_batch = 0

    losses_0 = []
    losses_1 = []
    losses_2 = []
    losses_3 = []
    losses_4 = []
    losses_5 = []
    losses_6 = []
    losses_7 = []
    losses_8 = []
    losses_9 = []
    losses_10 = []

    losses_mse = []
    losses_l1 = []

    for data in test_dataloader:
        inputs, labels = data

        if inputs.shape[0] != batch_size:
            continue

        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # rnn.loop()
        hidden = model.initHidden(batch_size)

        outputs = None
        outputs = model(inputs)
        lbl = torch.squeeze(labels)


        loss_MSE = torch.nn.MSELoss()
        loss_L1 = torch.nn.L1Loss()

        loss_0 = loss_MSE(outputs[:, 0], lbl[:, 0])
        loss_1 = loss_MSE(outputs[:, 1], lbl[:, 1])
        loss_2 = loss_MSE(outputs[:, 2], lbl[:, 2])
        loss_3 = loss_MSE(outputs[:, 3], lbl[:, 3])
        loss_4 = loss_MSE(outputs[:, 4], lbl[:, 4])
        loss_5 = loss_MSE(outputs[:, 5], lbl[:, 5])
        loss_6 = loss_MSE(outputs[:, 6], lbl[:, 6])
        loss_7 = loss_MSE(outputs[:, 7], lbl[:, 7])
        loss_8 = loss_MSE(outputs[:, 8], lbl[:, 8])
        loss_9 = loss_MSE(outputs[:, 9], lbl[:, 9])
        loss_10 = loss_MSE(outputs[:, 10], lbl[:, 10])

        loss_mse = loss_MSE(outputs, torch.squeeze(labels))
        loss_l1 = loss_L1(outputs, torch.squeeze(labels))

        #print(loss_MSE(outputs[:,0], lbl[:,0]))
        #print(loss_MSE(outputs[0][0],lbl[0][0]))
        #print(lbl.shape)
        #print(outputs.shape[1])
        #print(outputs[:,0])
        #print(outputs[:,1])

        losses_0.append(loss_0.cpu().data.numpy())
        losses_1.append(loss_1.cpu().data.numpy())
        losses_2.append(loss_2.cpu().data.numpy())
        losses_3.append(loss_3.cpu().data.numpy())
        losses_4.append(loss_4.cpu().data.numpy())
        losses_5.append(loss_5.cpu().data.numpy())
        losses_6.append(loss_6.cpu().data.numpy())
        losses_7.append(loss_7.cpu().data.numpy())
        losses_8.append(loss_8.cpu().data.numpy())
        losses_9.append(loss_9.cpu().data.numpy())
        losses_10.append(loss_10.cpu().data.numpy())

        losses_mse.append(loss_mse.cpu().data.numpy())
        losses_l1.append(loss_l1.cpu().data.numpy())

        tested_batch += 1

        #print(loss_l1)
        #print(loss_l1.data)


        if tested_batch % 1000000000 == 0:
            cur_time = time.time()
            print('Tested #: {}, loss_l1: {}, loss_mse: {}, time: {}'.format( \
                tested_batch * batch_size, \
                np.around([loss_l1.data[0]], decimals=8), \
                np.around([loss_mse.data[0]], decimals=8), \
                np.around([cur_time - pre_time], decimals=8)))
            pre_time = cur_time

    losses_0 = np.array(losses_0)
    losses_1 = np.array(losses_1)
    losses_2 = np.array(losses_2)
    losses_3 = np.array(losses_3)
    losses_4 = np.array(losses_4)
    losses_5 = np.array(losses_5)
    losses_6 = np.array(losses_6)
    losses_7 = np.array(losses_7)
    losses_8 = np.array(losses_8)
    losses_9 = np.array(losses_9)
    losses_10 = np.array(losses_10)

    mean_0 = np.mean(losses_0)
    mean_1 = np.mean(losses_1)
    mean_2 = np.mean(losses_2)
    mean_3 = np.mean(losses_3)
    mean_4 = np.mean(losses_4)
    mean_5 = np.mean(losses_5)
    mean_6 = np.mean(losses_6)
    mean_7 = np.mean(losses_7)
    mean_8 = np.mean(losses_8)
    mean_9 = np.mean(losses_9)
    mean_10 = np.mean(losses_10)

    std_0 = np.std(losses_0)
    std_1 = np.std(losses_1)
    std_2 = np.std(losses_2)
    std_3 = np.std(losses_3)
    std_4 = np.std(losses_4)
    std_5 = np.std(losses_5)
    std_6 = np.std(losses_6)
    std_7 = np.std(losses_7)
    std_8 = np.std(losses_8)
    std_9 = np.std(losses_9)
    std_10 = np.std(losses_10)

    data = [
        [mean_0, std_0],
        [mean_1, std_1],
        [mean_2, std_2],
        [mean_3, std_3],
        [mean_4, std_4],
        [mean_5, std_5],
        [mean_6, std_6],
        [mean_7, std_7],
        [mean_8, std_8],
        [mean_9, std_9],
        [mean_10, std_10]
    ]

    with open('output.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write multiple rows
        writer.writerows(data)


    losses_l1 = np.array(losses_l1)
    losses_mse = np.array(losses_mse)
    mean_l1 = np.mean(losses_l1) * max_speed
    mean_mse = np.mean(losses_mse) * max_speed
    std_l1 = np.std(losses_l1) * max_speed
    std_mse = np.std(losses_mse) * max_speed

    #print('Tested: L1_mean: {}, MSE_mean: {}, L1_std : {}, MSE_std : {}'.format(mean_l1, mean_mse, std_l1, std_mse))
    print('Node 0 MSE: {}'.format(mean_0))
    print('Node 1 MSE: {}'.format(mean_1))
    print('Node 2 MSE: {}'.format(mean_2))
    print('Node 3 MSE: {}'.format(mean_3))
    print('Node 4 MSE: {}'.format(mean_4))
    print('Node 5 MSE: {}'.format(mean_5))
    print('Node 6 MSE: {}'.format(mean_6))
    print('Node 7 MSE: {}'.format(mean_7))
    print('Node 8 MSE: {}'.format(mean_8))
    print('Node 9 MSE: {}'.format(mean_9))
    print('Node 10 MSE: {}'.format(mean_10))
    print('============================================')
    print('Node 0 STD: {}'.format(std_0))
    print('Node 1 STD: {}'.format(std_1))
    print('Node 2 STD: {}'.format(std_2))
    print('Node 3 STD: {}'.format(std_3))
    print('Node 4 STD: {}'.format(std_4))
    print('Node 5 STD: {}'.format(std_5))
    print('Node 6 STD: {}'.format(std_6))
    print('Node 7 STD: {}'.format(std_7))
    print('Node 8 STD: {}'.format(std_8))
    print('Node 9 STD: {}'.format(std_9))
    print('Node 10 STD: {}'.format(std_10))

    return [losses_l1, losses_mse, mean_l1, std_l1]

class FilterLinear(nn.Module):
    def __init__(self, in_features, out_features, filter_square_matrix, bias=True):
        '''
        filter_square_matrix : filter square matrix, whose each elements is 0 or 1.
        '''
        super(FilterLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        use_gpu = torch.cuda.is_available()
        self.filter_square_matrix = None
        if use_gpu:
            self.filter_square_matrix = Variable(filter_square_matrix.cuda(), requires_grad=False)
        else:
            self.filter_square_matrix = Variable(filter_square_matrix, requires_grad=False)

        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    #         print(self.weight.data)
    #         print(self.bias.data)

    def forward(self, input):
        return F.linear(input, self.filter_square_matrix.matmul(self.weight), self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', bias=' + str(self.bias is not None) + ')'

class LSTM(nn.Module):
    def __init__(self, input_size, cell_size, hidden_size, output_last=True):
        """
        cell_size is the size of cell_state.
        hidden_size is the size of hidden_state, or say the output_state of each step
        """
        super(LSTM, self).__init__()

        self.cell_size = cell_size
        self.hidden_size = hidden_size
        self.fl = nn.Linear(input_size + hidden_size, hidden_size)
        self.il = nn.Linear(input_size + hidden_size, hidden_size)
        self.ol = nn.Linear(input_size + hidden_size, hidden_size)
        self.Cl = nn.Linear(input_size + hidden_size, hidden_size)

        self.output_last = output_last

    def step(self, input, Hidden_State, Cell_State):
        combined = torch.cat((input, Hidden_State), 1)
        f = torch.sigmoid(self.fl(combined))
        i = torch.sigmoid(self.il(combined))
        o = torch.sigmoid(self.ol(combined))
        C = torch.tanh(self.Cl(combined))
        Cell_State = f * Cell_State + i * C
        Hidden_State = o * torch.tanh(Cell_State)

        return Hidden_State, Cell_State

    def forward(self, inputs):
        batch_size = inputs.size(0)
        time_step = inputs.size(1)
        Hidden_State, Cell_State = self.initHidden(batch_size)

        if self.output_last:
            for i in range(time_step):
                Hidden_State, Cell_State = self.step(torch.squeeze(inputs[:, i:i + 1, :]), Hidden_State, Cell_State)
            return Hidden_State
        else:
            outputs = None
            for i in range(time_step):
                Hidden_State, Cell_State = self.step(torch.squeeze(inputs[:, i:i + 1, :]), Hidden_State, Cell_State)
                if outputs is None:
                    outputs = Hidden_State.unsqueeze(1)
                else:
                    outputs = torch.cat((outputs, Hidden_State.unsqueeze(1)), 1)
            return outputs

    def initHidden(self, batch_size):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            Cell_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            return Hidden_State, Cell_State
        else:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size))
            Cell_State = Variable(torch.zeros(batch_size, self.hidden_size))
            return Hidden_State, Cell_State

class GraphConvolutionalLSTM(nn.Module):

    def __init__(self, K, A, FFR, feature_size, Clamp_A=True, output_last=True):
        '''
        Args:
            K: K-hop graph
            A: adjacency matrix
            feature_size: the dimension of features
        '''
        super(GraphConvolutionalLSTM, self).__init__()
        self.feature_size = feature_size
        self.hidden_size = feature_size

        self.K = K

        self.A_list = []  # Adjacency Matrix List
        A = torch.FloatTensor(A)
        A_temp = torch.eye(feature_size, feature_size)
        #print(A_temp)
        #print(torch.Tensor(A))
        for i in range(K):
            A_temp = torch.matmul(A_temp, torch.Tensor(A))
            if Clamp_A:
                # confine elements of A
                A_temp = torch.clamp(A_temp, max=1.)
            self.A_list.append(torch.mul(A_temp, torch.Tensor(FFR)))
        #             self.A_list.append(A_temp)

        # a length adjustable Module List for hosting all graph convolutions
        self.gc_list = nn.ModuleList(
            [FilterLinear(feature_size, feature_size, self.A_list[i], bias=False) for i in range(K)])

        hidden_size = self.feature_size
        input_size = self.feature_size * K

        self.fl = nn.Linear(input_size + hidden_size, hidden_size)
        self.il = nn.Linear(input_size + hidden_size, hidden_size)
        self.ol = nn.Linear(input_size + hidden_size, hidden_size)
        self.Cl = nn.Linear(input_size + hidden_size, hidden_size)

        # initialize the neighbor weight for the cell state
        self.Neighbor_weight = Parameter(torch.FloatTensor(feature_size))
        stdv = 1. / math.sqrt(feature_size)
        self.Neighbor_weight.data.uniform_(-stdv, stdv)

        self.output_last = output_last

    def step(self, input, Hidden_State, Cell_State):

        x = input

        gc = self.gc_list[0](x)
        for i in range(1, self.K):
            gc = torch.cat((gc, self.gc_list[i](x)), 1)

        combined = torch.cat((gc, Hidden_State), 1)
        f = torch.sigmoid(self.fl(combined))
        i = torch.sigmoid(self.il(combined))
        o = torch.sigmoid(self.ol(combined))
        C = torch.tanh(self.Cl(combined))

        NC = torch.mul(Cell_State,
                       torch.mv(Variable(self.A_list[-1], requires_grad=False).cuda(), self.Neighbor_weight))
        Cell_State = f * NC + i * C
        Hidden_State = o * torch.tanh(Cell_State)

        return Hidden_State, Cell_State, gc

    def Bi_torch(self, a):
        a[a < 0] = 0
        a[a > 0] = 1
        return a

    def forward(self, inputs):
        batch_size = inputs.size(0)
        time_step = inputs.size(1)
        Hidden_State, Cell_State = self.initHidden(batch_size)

        outputs = None

        for i in range(time_step):
            Hidden_State, Cell_State, gc = self.step(torch.squeeze(inputs[:, i:i + 1, :]), Hidden_State, Cell_State)

            if outputs is None:
                outputs = Hidden_State.unsqueeze(1)
            else:
                outputs = torch.cat((outputs, Hidden_State.unsqueeze(1)), 1)

        if self.output_last:
            return outputs[:, -1, :]
        else:
            return outputs

    def initHidden(self, batch_size):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            Cell_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            return Hidden_State, Cell_State
        else:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size))
            Cell_State = Variable(torch.zeros(batch_size, self.hidden_size))
            return Hidden_State, Cell_State

    def reinitHidden(self, batch_size, Hidden_State_data, Cell_State_data):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(Hidden_State_data.cuda(), requires_grad=True)
            Cell_State = Variable(Cell_State_data.cuda(), requires_grad=True)
            return Hidden_State, Cell_State
        else:
            Hidden_State = Variable(Hidden_State_data, requires_grad=True)
            Cell_State = Variable(Cell_State_data, requires_grad=True)
            return Hidden_State, Cell_State


# Load and divide data
df = pd.read_csv(r'sample2.csv')
#df = pd.read_pickle('speed_matrix_2015')
train_dataloader, valid_dataloader, test_dataloader, max_dataloader = PrepareDataset(df)

# Get data dimension
inputs, labels = next(iter(train_dataloader))
[batch_size, step_size, fea_size] = inputs.size()
input_dim = fea_size
hidden_dim = fea_size
output_dim = fea_size


print("LSTM")
lstm = LSTM(input_dim, hidden_dim, output_dim, output_last=True)
lstm, lstm_loss = TrainModel(lstm, train_dataloader, valid_dataloader, num_epochs=1)
lstm_test = NodeTestModel(lstm, test_dataloader, max_dataloader )
#np.save('lstm_loss', lstm_loss)
#np.save('lstm', lstm)
"""
print("GCN-LSTM")
A = np.load('mn-adj-10.npy')
K = 2
back_length = 4
Clamp_A = False
# gclstm = GraphConvolutionalLSTM(K, torch.Tensor(A), FFR[back_length], A.shape[0], Clamp_A=Clamp_A, output_last = True)
gclstm = GraphConvolutionalLSTM(K, torch.Tensor(A), torch.Tensor(A), A.shape[0], Clamp_A=Clamp_A, output_last=True)
gclstm, gclstm_loss = TrainModel(gclstm, train_dataloader, valid_dataloader, num_epochs=1)
gclstm_test = NodeTestModel(gclstm, test_dataloader, max_dataloader )
#np.save('gclstm_loss', gclstm_loss)
#np.save('gclstm', gclstm)
"""


