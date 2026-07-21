import torch
import torch.nn as nn
from ..config import AmyloidConfig

class AveragePooling(nn.Module):
	def __init__(self):
		super().__init__()
	
	@staticmethod
	def forward(output):
		pooled_output = output.mean(dim=1)
		return pooled_output

class AttentionModule(nn.Module):
	def __init__(self, input_dim=128):
		super().__init__()
		self.fc1 = nn.Linear(input_dim, input_dim)
		self.fc2 = nn.Linear(input_dim, input_dim // 2)
		self.fc3 = nn.Linear(input_dim // 2, input_dim)
		self.leakyRelu = nn.LeakyReLU()
		self.sigmoid = nn.Sigmoid()

	def forward(self, Fc):
		Q1 = self.leakyRelu(self.fc1(Fc))
		Q2 = self.leakyRelu(self.fc2(Q1))
		Wc = self.sigmoid(self.fc3(Q2))
		Fa = Fc * Wc
		return Fa, Wc

class DeepModel(nn.Module):
	def __init__(self, config: AmyloidConfig, dim, encoder_dim=64):
		super().__init__()
		self.model_config = config.model
		self.model = self.model_config.deep_model
		dataset = config.dataset
		semantic = self.model_config.semantic_feature_extractor
		if self.model_config.deep_feature_mode == 'semantic':
			self.feature_names = [semantic]
		elif self.model_config.deep_feature_mode == 'structure':
			self.feature_names = ['saprot']
		else:
			self.feature_names = [semantic, 'saprot']
		self.encoders = nn.ModuleDict(
			{name: nn.Linear(dim[name], encoder_dim) for name in self.feature_names}
		)
		self.model_dim = encoder_dim * len(self.feature_names)
		if self.model == 'ffn':
			if dataset == 'Amy':
				from .ffn import FFNAmy
				self.deep_model_layer = FFNAmy(self.model_dim)
			else:
				from .ffn import FFNCb
				self.deep_model_layer = FFNCb(self.model_dim)
		elif self.model == 'lstm':
			self.deep_model_layer = nn.LSTM(
				self.model_dim, self.model_dim, num_layers=2, batch_first=True)
		elif self.model == 'transformer':
			layer = nn.TransformerEncoderLayer(
				d_model=self.model_dim, nhead=4, batch_first=True)
			self.deep_model_layer = nn.TransformerEncoder(layer, num_layers=1)
		else:
			try:
				from mamba_ssm import Mamba
			except ImportError:
				raise ImportError('mamba_ssm import error')
			self.deep_model_layer = Mamba(
				d_model=self.model_dim, d_state=16, expand=2, d_conv=4)
		self.layer_norm = nn.LayerNorm(self.model_dim)
		self.average_pool = AveragePooling()
		self.fc = nn.Linear(self.model_dim, encoder_dim)
		
	def forward(self, x):
		encoded = [self.encoders[name](x[name]) for name in self.feature_names] # (batch, 512, 64*(2))
		x = encoded[0] if len(encoded) == 1 else torch.cat(encoded, dim=-1) # (batch, 512, 64(*2))
		x_out = self.deep_model_layer(self.layer_norm(x))[0] if self.model == 'lstm' \
			else self.deep_model_layer(self.layer_norm(x)) # (batch, 512, 64(*2))
		pooled_out = self.average_pool(x_out) # (batch, 64(*2))
		return self.fc(pooled_out) # (batch, 64)
		

class Model(nn.Module):
	def __init__(self, config: AmyloidConfig, dim, encoder_dim=64, num_classes=1):
		super().__init__()
		model_config = config.model
		self.deep_model = DeepModel(config, dim, encoder_dim)
		self.feature_mode = model_config.feature_mode
		self.use_attention = model_config.use_attention
		self.joint_loss = model_config.joint_loss
		self.know_encoder = nn.Linear(dim['handcrafted'], encoder_dim)
		self.attention = AttentionModule(encoder_dim * 2)
		self.sigmoid = nn.Sigmoid()
		self.layer_norm = nn.LayerNorm(encoder_dim * 2)
		self.classifier_deep = nn.Linear(encoder_dim, num_classes)
		self.classifier_know = nn.Linear(encoder_dim, num_classes)
		self.classifier_fusion = nn.Linear(encoder_dim * 2, num_classes)
		self.classifier_attn = nn.Linear(encoder_dim * 2, num_classes)
		self.weight_deep = nn.Parameter(torch.tensor(1.0 / 3))
		self.weight_know = nn.Parameter(torch.tensor(1.0 / 3))
		self.weight_final = nn.Parameter(torch.tensor(1.0 / 3))
	
	def forward(self, x):
		x_deep = self.deep_model(x) # (batch, 64)
		x_know = self.know_encoder(x['handcrafted']) # (batch, 64)
		x_fusion = self.layer_norm(torch.cat([x_deep, x_know], dim=-1))  # (batch, 128)
		x_attn, attn_w = self.attention(x_fusion)  # (batch, 128)
		logits_deep = self.classifier_deep(x_deep).squeeze(-1)
		logits_know = self.classifier_know(x_know).squeeze(-1)
		logits_fusion = self.classifier_fusion(x_fusion).squeeze(-1)
		logits_attn = self.classifier_attn(x_attn).squeeze(-1)
		w_deep = self.sigmoid(self.weight_deep)
		w_know = self.sigmoid(self.weight_know)
		w_final = self.sigmoid(self.weight_final)
		if self.feature_mode == 'deep':
			return logits_deep
		elif self.feature_mode == 'know':
			return logits_know
		else:
			if not self.joint_loss:
				if self.use_attention:
					return logits_attn
				return logits_fusion
			else:
				if not self.use_attention:
					return {
						'deep': {'logit': logits_deep, 'weight': w_deep},
						'know': {'logit': logits_know, 'weight': w_know},
						'final': {'logit': logits_fusion, 'weight': w_final},
					}
				else:
					return {
						'deep': {'logit': logits_deep, 'weight': w_deep},
						'know': {'logit': logits_know, 'weight': w_know},
						'final': {'logit': logits_attn, 'weight': w_final},
						'attn_weight': attn_w,
						'umap_features': {'know': x_know, 'deep': x_deep, 'fusion': x_fusion, 'attn': x_attn}
					}
		
		

