import math
import torch
import torch.nn as nn
import torch.nn.functional as F 
from transformers import AutoTokenizer
from transformers.models.bert.modeling_bert import BertForMaskedLM
import torch_geometric.nn as gnn
from transformers.models.bert.configuration_bert import BertConfig
from .modeling import MyBertForMaskedLM
from .util import pad_batch, unpad_batch
import torch_geometric.utils as utils
from einops import rearrange
# from convert import tan_proj

class TransformerEncoderLayer(nn.TransformerEncoderLayer):

    def __init__(self, d_model, nhead=8, dim_feedforward=512, dropout=0.1,
                 activation="relu", batch_norm=True, pre_norm=False,
                 gnn_type="gcn", **kwargs):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)

        self.self_attn = Attention(d_model, nhead, dropout=dropout,
                                   bias=False, gnn_type=gnn_type, **kwargs)
        self.batch_norm = batch_norm
        self.pre_norm = pre_norm
        if batch_norm:
            self.norm1 = nn.BatchNorm1d(d_model)
            self.norm2 = nn.BatchNorm1d(d_model)

    def forward(self, x, dag_rr_edge_index, return_attn=False):

        if self.pre_norm:
            x = self.norm1(x)

        x2, attn = self.self_attn(
            x,
            dag_rr_edge_index,
            return_attn=return_attn
        )

        x = x + self.dropout1(x2)
        if self.pre_norm:
            x = self.norm2(x)
        else:
            x = self.norm1(x)
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(x2)

        if not self.pre_norm:
            x = self.norm2(x)
        return x


class Attention(gnn.MessagePassing):
    """Multi-head DAG attention using PyG interface
    accept Batch data given by PyG
    """

    def __init__(self, embed_dim, num_heads=8, dropout=0., bias=False, symmetric=False, gnn_type="gcn", **kwargs):

        super().__init__(node_dim=0, aggr='add')
        self.embed_dim = embed_dim
        self.bias = bias
        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.scale = head_dim ** -0.5
        self.gnn_type = gnn_type
        # self.structure_extractor = StructureExtractor(embed_dim, gnn_type=gnn_type, **kwargs)
        self.attend = nn.Softmax(dim=-1)
        self.batch_first = True

        self.symmetric = symmetric
        if symmetric:
            self.to_qk = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.to_tqk = nn.Linear(embed_dim, embed_dim, bias=bias)
        else:
            self.to_qk = nn.Linear(embed_dim, embed_dim * 2, bias=bias)
            self.to_tqk = nn.Linear(embed_dim, embed_dim * 2, bias=bias)
        self.to_v = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

        self.attn_sum = None

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.to_qk.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

        if self.bias:
            nn.init.constant_(self.to_qk.bias, 0.)
            nn.init.constant_(self.to_v.bias, 0.)

    def forward(self,
                x,
                dag_rr_edge_index,
                return_attn=False):


        v = self.to_v(x)

        x_struct = x

        # Compute query and key matrices
        if self.symmetric:
            qk = self.to_qk(x_struct)
            qk = (qk, qk)
        else:
            qk = self.to_qk(x_struct).chunk(2, dim=-1)

        # Compute self-attention
        attn = None
        out = self.propagate(dag_rr_edge_index, v=v, qk=qk, edge_attr=None, size=None,
                             return_attn=return_attn)
        if return_attn:
            attn = self._attn
            self._attn = None
            attn = torch.sparse_coo_tensor(
                dag_rr_edge_index,
                attn,
            ).to_dense().transpose(0, 1)
        out = rearrange(out, 'n h d -> n (h d)')

        return self.out_proj(out), attn

    def message(self, v_j, qk_j, qk_i, edge_attr, index, ptr, size_i, return_attn):
        """Self-attention based on MPNN """
        qk_i = rearrange(qk_i, 'n (h d) -> n h d', h=self.num_heads)
        qk_j = rearrange(qk_j, 'n (h d) -> n h d', h=self.num_heads)
        v_j = rearrange(v_j, 'n (h d) -> n h d', h=self.num_heads)
        attn = (qk_i * qk_j).sum(-1) * self.scale
        if edge_attr is not None:
            attn = attn + edge_attr
        attn = utils.softmax(attn, index, ptr, size_i)
        if return_attn:
            self._attn = attn
        attn = self.attn_dropout(attn)

        return v_j * attn.unsqueeze(-1)

    def self_attn(self, qk, v, ptr, mask_dag_, return_attn=False):
        """ Self attention based on mask matrix"""

        qk, mask = pad_batch(qk, ptr, return_mask=True)
        k, q = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qk)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        dots = dots.masked_fill(
            mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        # DAG mask
        mask_dag_ = mask_dag_.reshape(dots.shape[0], mask_dag_.shape[1], mask_dag_.shape[1])
        mask_dag_ = mask_dag_[:, :dots.shape[2], :dots.shape[3]]
        dots = dots.masked_fill(
            mask_dag_.unsqueeze(1),
            float('-inf'),
        )
        dots = self.attend(dots)
        dots = self.attn_dropout(dots)
        v = pad_batch(v, ptr)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        out = torch.matmul(dots, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = unpad_batch(out, ptr)

        if return_attn:
            return out, dots
        return out, None


class GraphTransformerEncoder(nn.TransformerEncoder):
    def forward(self, x, dag_rr_edge_index, return_attn=False):
        output = x
        for mod in self.layers:
            output = mod(output, dag_rr_edge_index, return_attn=return_attn
            )
        if self.norm is not None:
            output = self.norm(output)
        return output

class diffusion_graph(nn.Module):
    def __init__(self,args) -> None:
        super().__init__()
        bert_config = BertConfig.from_pretrained(args.model_config)
        self.base_model = MyBertForMaskedLM(bert_config)
        self.edge_model = MyBertForMaskedLM(bert_config)
        self.node_model = MyBertForMaskedLM(bert_config)
        self.max_len = args.max_n
        self.max_step = args.diff_step
        self.time_embed = nn.Embedding(self.max_step,self.base_model.config.hidden_size)
        self.embedding_lap_pos_enc = nn.Linear(args.pos_enc_dim,self.base_model.config.hidden_size)
        self.args = args
        self.position = torch.arange(args.max_n).unsqueeze(1).to(args.device)
        self.div_term = torch.exp(torch.arange(0, self.base_model.config.hidden_size, 2) * (-math.log(10000.0) / self.base_model.config.hidden_size)).to(args.device)
        self.poe = torch.zeros(args.max_n, self.base_model.config.hidden_size).to(args.device)
        self.poe[:, 0::2] = torch.sin(self.position * self.div_term).to(args.device)
        self.poe[:, 1::2] = torch.cos(self.position * self.div_term).to(args.device)
        nn.init.constant_(self.time_embed.weight, 0)
        self.mlp = nn.Sequential(
                nn.Linear(self.base_model.config.hidden_size*2, self.base_model.config.hidden_size*4), 
                nn.ReLU(), 
                nn.Linear(self.base_model.config.hidden_size*4, 1),
                nn.Sigmoid()
                )
        self.encoder_layer = TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=512, dropout=0., batch_norm=False,
            gnn_type="gcn")


        self.encoder_edge = GraphTransformerEncoder(self.encoder_layer, 4)# num_layers
        self.encoder_node = GraphTransformerEncoder(self.encoder_layer, 4)# num_layers

        self.embedding = nn.Embedding(70, 256)

        self.dropout = nn.Dropout(0.1)

    def construct_batch_adjacency_matrices(self, node_lists, mask = 69):
        """
        根据多个节点列表批量构造邻接矩阵，其中每个节点只能连接到其后的节点，
        并且如果节点值为69，则该节点不可以与其他节点相连。

        参数：
        node_lists (list of torch.Tensor): 多个节点列表，每个列表的形状为 [n]

        返回：
        adjacency_matrices (torch.Tensor): 批量邻接矩阵，形状为 [batch_size, max_num_nodes, max_num_nodes]
        """
        batch_size = len(node_lists)
        max_num_nodes = max(len(node_list) for node_list in node_lists)

        # 初始化全0的批量邻接矩阵
        adjacency_matrices = torch.zeros((batch_size, max_num_nodes, max_num_nodes), dtype=torch.float32,device=node_lists.device)

        for b, node_list in enumerate(node_lists):
            num_nodes = len(node_list)
            for i in range(num_nodes):
                if node_list[i] == mask:
                    continue
                for j in range(i + 1, num_nodes):
                    adjacency_matrices[b, i, j] = 1

        return adjacency_matrices

    def combine(self, node, edge, filter=False, mask=69):
        batch_size, node_num, _ = edge.shape
        all_edges = []
        all_index = []
        num_nodes = 0

        for batch_index in range(batch_size):
            A = edge[batch_index]
            if filter:
                single_index = node[batch_index][node[batch_index]!=mask]
            else:
                single_index = node[batch_index]
            source_nodes, target_nodes = A.nonzero(as_tuple=True)
            source_nodes += num_nodes  # 调整节点索引以区分不同的图
            target_nodes += num_nodes
            edges = torch.stack([source_nodes, target_nodes], dim=0)
            num_nodes += len(single_index)
            all_edges.append(edges)
            all_index.append(single_index)

        # 将所有边索引合并成一个张量
        dag_rr_edge_index = torch.cat(all_edges, dim=1)
        new_input_ids = torch.cat(all_index)

        return new_input_ids, dag_rr_edge_index

    def forward(self,input_ids,attention_mask,edge_matrix,lap_pos_enc,weight_tensor,t=None):
        batch_size, node_num, _ = edge_matrix.shape
        input_shape = input_ids.size()
        seq_length = input_shape[1]
        
        position_ids = self.base_model.bert.embeddings.position_ids[:, 0 : seq_length]
        position_embeddings = self.base_model.bert.embeddings.position_embeddings(position_ids)
        # position_embeddings = self.poe[:][position_ids].to(self.args.device)
        h_lap_pos_enc = self.embedding_lap_pos_enc(lap_pos_enc) 
        word_emb = self.base_model.bert.embeddings.word_embeddings(input_ids)

        # word_emb = o_p

        if self.args.use_lap:
            word_emb = word_emb + h_lap_pos_enc
        if t is None:
            diffusion_steps = torch.randint(0,self.max_step,size = (input_shape[0],),device=input_ids.device)
        else:
            diffusion_steps = torch.ones(size = (input_shape[0],),device=input_ids.device).long()*t



        noise = torch.randn_like(word_emb)/math.sqrt(self.base_model.config.hidden_size)
        alpha = 1 - torch.sqrt((diffusion_steps+1)/self.max_step).view(-1,1,1)
        noisy_word = torch.sqrt(alpha)*word_emb+torch.sqrt(1-alpha)*noise#  + token_type_embeddings


            
        time_embedding = self.time_embed(diffusion_steps).unsqueeze(1)
        noisy_word = noisy_word+position_embeddings+time_embedding

        middle_noisy_word = self.base_model.bert.embeddings.LayerNorm(noisy_word)

        extended_attention_mask = self.base_model.bert.get_extended_attention_mask(attention_mask, input_shape,device=attention_mask.device)

        #----------------
        node_out = self.node_model.bert.encoder(
            middle_noisy_word,
            attention_mask=extended_attention_mask
        )
        node_log = self.node_model.cls.predictions(node_out[0])
        node_pre = torch.argmax(node_log, dim=-1)
        #-------------adjust
        # node_pre = input_ids
        #-------------
        edge_mask = self.construct_batch_adjacency_matrices(node_pre)
        node_pre_, edge_mask_ = self.combine(node_pre, edge_mask)
        x_node = self.embedding(node_pre_)
        # x_node = self.dropout(x_node)
        output_edge = self.encoder_edge(
            x_node,
            edge_mask_,
            return_attn=False
        )

        edge_out = self.edge_model.bert.encoder(
            middle_noisy_word+position_embeddings,
            attention_mask=extended_attention_mask
        )
        edge_out = edge_out[0]
        e_emb = edge_out.unsqueeze(2).repeat(1, 1, self.args.max_n, 1)
        e_emb_t = edge_out.unsqueeze(2).repeat(1, 1, self.args.max_n, 1).transpose(1, 2)
        ee_emb_cat = torch.cat([e_emb, e_emb_t], dim=-1)
        edge_mats = self.mlp(ee_emb_cat)
        edge_mats = edge_mats.squeeze()
        edge_mats = torch.triu(edge_mats, diagonal=1)
        edge_mats = (edge_mats > self.args.thre).bool().float()
        #-------------adjust
        # node_mask = torch.randint(1, 70, (8, 50), dtype=torch.float32, device=edge_mats.device)
        node_mask = torch.zeros(input_ids.size(), dtype=torch.float32, device=edge_mats.device)
        # edge_mats = edge_matrix
        #------------
        _, edge_pre_ = self.combine(node_mask, edge_mats)
        # x_node_edge = self.embedding(node_mask_)
        # x_node_edge = self.dropout(x_node_edge)
        middle_noisy_word = middle_noisy_word.view(-1,self.base_model.config.hidden_size)
        output_node = self.encoder_node(
            # x_node_edge,
            middle_noisy_word,
            edge_pre_,
            return_attn=False
        )

        node_encoder_outputs = output_node.view(batch_size, node_num,-1)
        edge_encoder_outputs = output_edge.view(batch_size, node_num,-1)

        prediction_scores = self.node_model.cls.predictions(node_encoder_outputs)
        loss = F.cross_entropy(prediction_scores.view(-1, self.node_model.config.vocab_size), input_ids.flatten(),
                               ignore_index=self.args.PAD_TYPE, reduction='mean')


        e_emb = edge_encoder_outputs.unsqueeze(2).repeat(1, 1, self.args.max_n, 1)
        e_emb_t = edge_encoder_outputs.unsqueeze(2).repeat(1, 1, self.args.max_n, 1).transpose(1, 2)
        ee_emb_cat = torch.cat([e_emb, e_emb_t], dim=-1)
        ee_p = self.mlp(ee_emb_cat)
        ee_p = ee_p.squeeze()
        dis_mat = torch.triu(ee_p, diagonal=1)
        edge_loss_fun = torch.nn.MSELoss()
        edge_loss = edge_loss_fun(dis_mat,edge_matrix)

        return loss,prediction_scores,diffusion_steps,edge_encoder_outputs,edge_loss

    @torch.no_grad()
    def sampler(self,device,k=1,N=100):
        import time
        
        start_time = time.time()
        noisy_word = torch.normal(0,1,(N,self.max_len,self.base_model.config.hidden_size)).to(device) / math.sqrt(self.base_model.config.hidden_size)
        attention_mask = torch.ones(N,self.max_len).long().to(device)
        extended_attention_mask = self.base_model.bert.get_extended_attention_mask(attention_mask, attention_mask.shape,device=device)

        position_ids = self.base_model.bert.embeddings.position_ids[:, 0 : self.max_len]
        # position_embeddings = self.base_model.bert.embeddings.position_embeddings(position_ids)
        position_embeddings = self.poe[:][position_ids].to(self.args.device)


        for t in range(self.max_step-1,0,-k):
            diffusion_steps = torch.ones(size = (N,),device=device).long()*t
            time_embedding = self.time_embed(diffusion_steps).unsqueeze(1)

            model_input = noisy_word +position_embeddings+time_embedding # +token_type_embeddings
            model_input = self.base_model.bert.embeddings.LayerNorm(model_input)

            # ----------------
            node_out = self.node_model.bert.encoder(
                model_input,
                attention_mask=extended_attention_mask
            )
            node_log = self.node_model.cls.predictions(node_out[0])
            node_pre = torch.argmax(node_log, dim=-1)
            edge_mask = self.construct_batch_adjacency_matrices(node_pre)
            node_pre_, edge_mask_ = self.combine(node_pre, edge_mask)
            x_node = self.embedding(node_pre_)
            # x_node = self.dropout(x_node)
            output_edge = self.encoder_edge(
                x_node,
                edge_mask_,
                return_attn=False
            )

            edge_out = self.edge_model.bert.encoder(
                model_input + position_embeddings,
                attention_mask=extended_attention_mask
            )
            edge_out = edge_out[0]
            e_emb = edge_out.unsqueeze(2).repeat(1, 1, self.args.max_n, 1)
            e_emb_t = edge_out.unsqueeze(2).repeat(1, 1, self.args.max_n, 1).transpose(1, 2)
            ee_emb_cat = torch.cat([e_emb, e_emb_t], dim=-1)
            edge_mats = self.mlp(ee_emb_cat)
            edge_mats = edge_mats.squeeze()
            edge_mats = torch.triu(edge_mats, diagonal=1)
            edge_mats = (edge_mats > self.args.thre).bool().float()
            node_mask = torch.zeros((N,self.max_len,self.base_model.config.hidden_size), dtype=torch.float32, device=edge_mats.device)
            _, edge_pre_ = self.combine(node_mask, edge_mats)
            # x_node_edge = self.embedding(node_mask_)
            # x_node_edge = self.dropout(x_node_edge)
            model_input = model_input.view(-1, self.base_model.config.hidden_size)
            output_node = self.encoder_node(
                # x_node_edge,
                model_input,
                edge_pre_,
                return_attn=False
            )

            node_encoder_output = output_node.view(N,self.max_len,self.base_model.config.hidden_size)
            edge_encoder_output = output_edge.view(N,self.max_len,self.base_model.config.hidden_size)

            prediction_scores = self.node_model.cls.predictions(node_encoder_output)

            pred = torch.argmax(prediction_scores,-1).long()
            denoised_word = self.base_model.bert.embeddings.word_embeddings(pred)
            # denoised_word = prediction_scores.softmax(-1) @ self.model.bert.embeddings.word_embeddings.weight.unsqueeze(0)
        
            alpha_tk = 1 - math.sqrt((t+1-k)/self.max_step)#+1e-5
            alpha_t = 1 - math.sqrt((t+1)/self.max_step)+1e-5

            noise = (noisy_word - math.sqrt(alpha_t)*denoised_word)/math.sqrt(1-alpha_t)
            if self.args.edge_loop:
                noisy_word = math.sqrt(alpha_tk)*edge_encoder_output + math.sqrt(1-alpha_tk)*noise
            else:
                # noisy_word = math.sqrt(alpha_tk)*(noisy_word/math.sqrt(alpha_t) + (math.sqrt((1-alpha_tk)/alpha_tk) - math.sqrt((1-alpha_t)/alpha_t))*noise)
                noisy_word = math.sqrt(alpha_tk)*denoised_word + math.sqrt(1-alpha_tk)*noise
            print(f"\rnoise level {t}  {time.time()-start_time:.2f}",end='')
        if self.args.node_stochastic:
            pred = torch.multinomial(prediction_scores.softmax(-1).reshape(-1,self.args.num_vertex_type),1).reshape(self.args.decode_num,self.args.max_n)
        else:
            pred = torch.argmax(prediction_scores,-1).long()

        return pred,edge_encoder_output