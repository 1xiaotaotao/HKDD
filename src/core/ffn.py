import torch.nn as nn

class FFNAmy(nn.Module):
	def __init__(self, input_dim, dropout=0.2):
		super().__init__()
		self.fc1 = nn.Linear(input_dim, input_dim//2)
		self.fc2 = nn.Linear(input_dim//2, input_dim)
		# self.fc = nn.Linear(input_dim, 64)
		self.dropout = nn.Dropout(dropout)
		self.gelu = nn.GELU()


	def forward(self, x):
		x_fc1 = self.gelu(self.fc1(x))
		x_fc2 = self.gelu(self.fc2(x_fc1))   
		# x_out = self.fc(self.dropout(x_fc2))
		x_out = self.dropout(x_fc2)
		return x_out
    
    
    
class FFNCb(nn.Module):
	def __init__(self, input_dim: int, dropout=0.1):
		super().__init__()
		self.fc1 = nn.Linear(input_dim, input_dim * 2)
		self.fc2 = nn.Linear(input_dim * 2, input_dim)
		# self.fc = nn.Linear(input_dim, 64)
		self.dropout = nn.Dropout(dropout)#not use
		self.relu = nn.ReLU()

	def forward(self, x):
		x_fc1 = self.relu(self.fc1(x))
		x_fc2 = self.relu(self.fc2(x_fc1))
		# x_out = self.fc(x_fc2)
		x_out = x_fc2
		return x_out