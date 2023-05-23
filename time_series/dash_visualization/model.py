import pandas as pd
from pmdarima.arima import auto_arima
from pmdarima.model_selection import train_test_split
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


data = pd.read_csv('../data/processed/train.csv')


time = data.Date
target = data.target_value
TRAIN_SIZE = len(target) - 7
train_arima, test_arima = train_test_split(target, train_size=TRAIN_SIZE)

model_arima = auto_arima(train_arima, m=30, trace=True)
forecasts_arima = model_arima.predict(n_periods=7)




def transform_data(arr, seq_len):
    x, y = [], []
    for i in range(arr.shape[0] - seq_len - 7):
        x_i = arr[i : i + seq_len, 0]
        y_i = arr[i + seq_len : i + seq_len + 7, 0]
        x.append(x_i)
        y.append(y_i)

    return x, y

df = pd.read_csv('../data/processed/train.csv')
target = df[['target_value']]



seq_len = 10
x_train = torch.tensor(target.iloc[:350].values).float()
x_test = torch.tensor(target.iloc[350:359].values).float()
y_test = torch.tensor(target.iloc[359:].values).float()

# scaler = StandardScaler()
# x_train = torch.tensor(scaler.fit_transform(x_train)).float()
# x_test = torch.tensor(scaler.transform(x_test)).float()
# y_test = torch.tensor(scaler.transform(y_test)).float()


x_train, y_train = transform_data(x_train, seq_len)


class OwnLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(OwnLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x.view(1, len(x), 1).float())
        y_pred = self.linear(lstm_out.view(len(x), -1).float())
        return y_pred[-7]


input_dim = 1
hidden_dim = 64
output_dim = 7


model = OwnLSTM(input_dim, hidden_dim, output_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 40

for epoch in range(num_epochs):
    for i in range(len(x_train)):
        optimizer.zero_grad()
        y_pred = model(x_train[i])
        loss = criterion(y_pred, y_train[i])
        loss.backward()
        optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')


y_pred = model(x_test).detach().numpy()
# y_pred = scaler.inverse_transform(model(x_test).detach().numpy().reshape(1,-1))
df = pd.read_csv('../data/processed/train.csv')
time = df.Date
lstm_train = df.target_value.iloc[:359]
lstm_test = df.target_value.iloc[359:]
lstm_pred = y_pred