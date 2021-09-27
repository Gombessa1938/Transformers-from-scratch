import torch 
import torch.nn as nn 

class selfAttention(nn.Module):
	def __init__(self,embed_size,heads):
		super(selfAttention,self).__init__()
		self.embed_size = embed_size
		self.heads = heads 
		self.head_dim = embed_size // heads

		assert (self.head_dim * heads == embed_size),"embed size needs to be div by heads"

		self.values = nn.Linear(self.head_dim,self.head_dim,bias= False)
		self.keys = nn.Linear(self.head_dim,self.head_dim,bias=False)
		self.queries = nn.Linear(self.head_dim,self.head_dim,bias=False)

		self.fc_out = nn.Linear(heads*self.head_dim,embed_size)

	def forward(self,values,keys,quey,mask):
		N = quey.shape[0]
		value_len,key_len,query_len = values.shape[1],keys.shape[1],query.shape[1]

		values = values.reshape(N,value_len,self.heads,self.head_dim)
		keys = keys.reshape(N,key_len,self.heads,self.head_dim)
		queries = query.reshape(N,query_len,self.heads,self.head_dim)

		values = self.values(values)
		keys = self.keys(keys)
		queries = self.queries(queries)

		energy = torch.einsum('nqhd,nkhd-->nhqk',[queries,keys])



		#queries shape ( N q_len,heads, heads_dim)
		#key shape( N, key_len,heads, heads_dim)
		#energy shape(N,heads, query_len,key_len)
		#torch.bmm(batch matrixmul)
		if mask is not None:
			energy = energy.masked_fill(mask == 0,float("-1e20"))

		attention = torch.softmax(energy / (self.embed_size **(1/2)),dim=3)

		out = torch.einsum('nhql,nlhd-->nqhd',[attention,values]).reshape(N,query_len,self.heads*self.head_dim)


		#attention shape:(N,heads,query_len,key_len)
		#vlues shape:(N,value_len,heads,heads_dim)
		#(N,query_len,heads,head_dim)
		out = self.fc_out(out)

		return out  




class Transformer_Block(nn.Module):
	def __init__(self,embed_size,heads,dropout,forward_expansion):
		super(Transformer_Block,self).__init__()
		self.attention = selfAttention(embed_size,heads)
		self.norm1 = nn.LayerNorm(embed_size)
		self.norm2 = nn.LayerNorm(embed_size)

		self.feed_forward = nn.Sequential(
			nn.Linear(embed_size,forward_expansion*embed_size),
			nn.ReLU(),
			nn.Linear(forward_expansion*embed_size,embed_size)
			)
		self.dropout = nn.Dropout(dropout)

	def forward(self,value,key,query,mask):
		attention = self.attention(value,key,query,mask)
		x = self.dropout(self.norm1(attention + query))
		forward = self.feed_forwad(x)
		out = self.dropout(self.norm2(forward + x))
		return out 


class Encoder(nn.Module):
	def __init__(self,src_vocab_size,embed_size,num_layers,heads,device,forward_expansion,dropout,max_lenght,):
		super(Encoder,self).__init__()
		self.embed_size = embed_size
		self.device = device
		self.word_embedding = nn.Embedding(src_vocab_size,embed_size)
		self.position_embedding = nn.Embedding(max_lenght,embed_size)

		self.layers = nn.ModuleList([Transformer_Block(embed_size,heads,dropout=dropout,forward_expansion=forward_expansion) for _ in range(num_layers)])


		self.dropout = nn.Dropout(dropout)

	def forward(self,x,mask):

		N,seq_length = x.shape
		positions = torch.arange(0,seq_length).expand(N,seq_length).to(self.device)

		out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
		for layer in self.layers:
			out = layer(out,out,out,mask)

		return out 	


class DecodeBlock(nn.Module):
	def __init__(self,embed_size,heads,forward_expansion,dropout,device):
		super(DecodeBlock).__init__()

		self.attention= selfAttention(embed_size,heads)
		self.norm = nn.LayerNorm(embed_size)
		self.Transformer_Block = Transformer_Block(embed_size,heads,dropout,forward_expansion)
		self.dropout =nn.Dropout(dropout)

	def forwad(self,x,value,key,src_mask,target_mask):
		attention = self.attention(x,x,x,target_mask)
		query = self.dropout(self.norm(attention+x))
		out = self.Transformer_Block(value,key,query,src_mask)
		return out

class Decoder(nn.Module):
	def __init__(self,target_vocab_size,embed_size,num_layers,heads,forward_expansion,dropout,device,max_lenght):
		super(Decoder,self).__init__()
		self.device = device
		self.word_embedding = nn.Embedding(target_vocab_size,embed_size)
		self.position_embedding = nn.Embedding(max_lenght,embed_size)

		self.layers = nn.ModuleList([DecodeBlock(embed_size,heads,forward_expansion,dropout,device)
			for _ in range(num_layers)])

		self.fc_out = nn.Linear(embed_size,target_vocab_size)
		self.dropout = nn.Dropout(dropout)

	def forward(self,x,encoder_out,src_mask,target_mask):
		N,seq_length = x.shape
		position = torch.arange(0,seq_length).expand(N,seq_length).to(self.device)
		x = self.dropout((self.word_embedding(x)+ self.position_embedding(positions)))


		for layer in self.layers:
			x = layer(x,encoder_out,encoder_out,src_mask,target_mask)
		out = self.fc_out(x)
		return out 





class Transformer(nn.Module):
	def __init__(self,src_vocab_size,
		target_vocab_size,src_pad_idx,
		target_pad_idx,embed_size=256,
		num_layers = 6,forward_expansion = 4,
		heads=8,dropout=0,device = 'cuda',max_lenght=100):
		super(Transformer,self).__init__()

		self.encoder = Encoder(src_vocab_size,embed_size,num_layers,heads,device,forward_expansion,dropout,max_lenght)
		self.decoder = Decoder(target_vocab_size,embed_size,num_layers,heads,forward_expansion,dropout,max_lenght)

		self.src_pad_idx = src_pad_idx
		self.target_pad_idx = target_pad_idx
		self.device = device

	def make_src_mask(self,src):
		src_mark= (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

		return src_mask.to(self.device)

	def make_target_mask(self,targets):
		N,target_len = targets.shape
		target_mask = torch.tril(torch.one((target_len,target_len))).expand(N,1,target_len,target_len)

		return target_mask.to(self.device)

	def forward(self,src,target):
		src_mask = self,make_src_mask(src)
		target_mask = self.make_target_mask(target)
		enc_src = self.encoder(src,src_mask)
		out = self.decoder(target,enc_src,src_mask,target_mask)

		return out 

