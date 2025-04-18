# 导入必要的库
import random  # 用于随机数生成
import pickle  # 用于序列化和反序列化Python对象
import torch  # PyTorch深度学习框架
from torch.cuda.amp import autocast as autocast  # 混合精度训练
import torch.nn as nn  # 神经网络模块
import numpy as np  # 数值计算库
from models.recsys_model import *  # 推荐系统模型
from models.llm4rec import *  # 推荐系统LLM模块
from sentence_transformers import SentenceTransformer  # 文本嵌入模型

# 定义两层的多层感知机
class two_layer_mlp(nn.Module):
    """双层MLP网络，用于特征转换和匹配"""
    def __init__(self, dims):
        """
        Args:
            dims (int): 输入特征的维度
        """
        super().__init__()
        # 第一全连接层：dims -> 128
        self.fc1 = nn.Linear(dims, 128)
        # 第二全连接层：128 -> dims
        self.fc2 = nn.Linear(128, dims)
        # Sigmoid激活函数
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """前向传播过程"""
        x = self.fc1(x)  # 第一层线性变换
        x = self.sigmoid(x)  # 激活函数
        x1 = self.fc2(x)  # 第二层线性变换
        return x, x1  # 返回中间层和最终输出

class A_llmrec_model(nn.Module):
    """A-LLMRec推荐系统主模型"""
    def __init__(self, args):
        """
        Args:
            args (argparse.Namespace): 包含模型配置的参数对象
        """
        super().__init__()
        # 初始化基础配置
        rec_pre_trained_data = args.rec_pre_trained_data  # 预训练数据名称
        self.args = args
        self.device = args.device  # 计算设备（CPU/GPU）

        # 加载商品文本信息字典
        with open(f'./data/amazon/{args.rec_pre_trained_data}_text_name_dict.json.gz','rb') as ft:
            self.text_name_dict = pickle.load(ft)  # 包含商品标题和描述的字典

        # 初始化推荐系统模块
        self.recsys = RecSys(args.recsys, rec_pre_trained_data, self.device)
        self.item_num = self.recsys.item_num  # 商品总数
        self.rec_sys_dim = self.recsys.hidden_units  # 推荐系统嵌入维度
        self.sbert_dim = 768  # Sentence-BERT的嵌入维度

        # 初始化MLP用于推荐系统嵌入处理
        self.mlp = two_layer_mlp(self.rec_sys_dim)

        # 预训练阶段1的初始化
        if args.pretrain_stage1:
            self.sbert = SentenceTransformer('nq-distilbert-base-v1')  # 文本嵌入模型
            self.mlp2 = two_layer_mlp(self.sbert_dim)  # 文本嵌入处理MLP

        # 损失函数和评估指标初始化
        self.mse = nn.MSELoss()  # 均方误差损失
        self.maxlen = args.maxlen  # 序列最大长度
        self.NDCG = 0  # 标准化折损累计增益
        self.HIT = 0   # 命中率
        self.bce_criterion = torch.nn.BCEWithLogitsLoss()  # 二元交叉熵损失

        # 预训练阶段2或推理阶段的初始化
        if args.pretrain_stage2 or args.inference:
            self.llm = llm4rec(device=self.device, llm_model=args.llm)  # 大语言模型

            # 用户行为日志嵌入投影层
            self.log_emb_proj = nn.Sequential(
                nn.Linear(self.rec_sys_dim, self.llm.llm_model.config.hidden_size),
                nn.LayerNorm(self.llm.llm_model.config.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.llm.llm_model.config.hidden_size, self.llm.llm_model.config.hidden_size)
            )
            # Xavier初始化权重
            nn.init.xavier_normal_(self.log_emb_proj[0].weight)
            nn.init.xavier_normal_(self.log_emb_proj[3].weight)

            # 商品嵌入投影层
            self.item_emb_proj = nn.Sequential(
                nn.Linear(128, self.llm.llm_model.config.hidden_size),
                nn.LayerNorm(self.llm.llm_model.config.hidden_size),
                nn.GELU(),  # Gaussian Error Linear Unit
                nn.Linear(self.llm.llm_model.config.hidden_size, self.llm.llm_model.config.hidden_size)
            )
            nn.init.xavier_normal_(self.item_emb_proj[0].weight)
            nn.init.xavier_normal_(self.item_emb_proj[3].weight)

    def save_model(self, args, epoch1=None, epoch2=None):
        """保存模型参数到文件"""
        out_dir = f'./models/saved_models/'
        create_dir(out_dir)  # 创建保存目录
        out_dir += f'{args.rec_pre_trained_data}_{args.recsys}_{epoch1}_'

        # 保存阶段1的模型组件
        if args.pretrain_stage1:
            torch.save(self.sbert.state_dict(), out_dir + 'sbert.pt')
            torch.save(self.mlp.state_dict(), out_dir + 'mlp.pt')
            torch.save(self.mlp2.state_dict(), out_dir + 'mlp2.pt') 

        # 保存阶段2的模型组件
        out_dir += f'{args.llm}_{epoch2}_'
        if args.pretrain_stage2:
            torch.save(self.log_emb_proj.state_dict(), out_dir + 'log_proj.pt')
            torch.save(self.item_emb_proj.state_dict(), out_dir + 'item_proj.pt')

    def load_model(self, args, phase1_epoch=None, phase2_epoch=None):
        """从文件加载预训练模型参数"""
        out_dir = f'./models/saved_models/{args.rec_pre_trained_data}_{args.recsys}_{phase1_epoch}_'
        
        # 加载阶段1的MLP参数并冻结
        mlp = torch.load(out_dir + 'mlp.pt', map_location=args.device)
        self.mlp.load_state_dict(mlp)
        for param in self.mlp.parameters():
            param.requires_grad = False  # 冻结参数

        # 推理时加载阶段2的参数
        if args.inference:
            out_dir += f'{args.llm}_{phase2_epoch}_'
            self.log_emb_proj.load_state_dict(torch.load(out_dir + 'log_proj.pt', map_location=args.device))
            self.item_emb_proj.load_state_dict(torch.load(out_dir + 'item_proj.pt', map_location=args.device))

    def find_item_text(self, item, title_flag=True, description_flag=True):
        """获取商品文本信息"""
        # 定义字段名称和默认值
        t, d = 'title', 'description'
        t_, d_ = 'No Title', 'No Description'
        
        # 根据标志组合返回不同文本格式
        if title_flag and description_flag:
            return [f'"{self.text_name_dict[t].get(i,t_)}, {self.text_name_dict[d].get(i,d_)}"' for i in item]
        elif title_flag:
            return [f'"{self.text_name_dict[t].get(i,t_)}"' for i in item]
        else:
            return [f'"{self.text_name_dict[d].get(i,d_)}"' for i in item]

    def get_item_emb(self, item_ids):
        """获取商品嵌入向量"""
        with torch.no_grad():
            # 通过推荐系统获取原始嵌入
            item_embs = self.recsys.model.item_emb(torch.LongTensor(item_ids).to(self.device))
            # 经过MLP处理
            item_embs, _ = self.mlp(item_embs)
        return item_embs

    def forward(self, data, optimizer=None, batch_iter=None, mode='phase1'):
        """统一前向传播入口"""
        if mode == 'phase1':
            self.pre_train_phase1(data, optimizer, batch_iter)
        elif mode == 'phase2':
            self.pre_train_phase2(data, optimizer, batch_iter)
        elif mode == 'generate':
            return self.generate(data)

    def pre_train_phase1(self, data, optimizer, batch_iter):
        """预训练阶段1：对齐推荐系统和文本嵌入"""
        # 解包数据批次信息
        epoch, total_epoch, step, total_step = batch_iter
        u, seq, pos, neg = data  # 用户ID，历史序列，正样本，负样本

        # 获取序列最后位置的索引
        indices = [self.maxlen*(i+1)-1 for i in range(u.shape[0])]

        # 获取推荐系统嵌入（不计算梯度）
        with torch.no_grad():
            log_emb, pos_emb, neg_emb = self.recsys.model(u, seq, pos, neg, mode='item')

        # 分割数据批次（每批60个样本）
        batch_size = 60
        total_loss = 0
        for i in range(0, len(log_emb), batch_size):
            # 获取当前批次数据
            batch_log = log_emb[i:i+batch_size]
            batch_pos = pos[i:i+batch_size]
            batch_neg = neg[i:i+batch_size]

            # 生成正负样本文本
            pos_text = self.find_item_text(batch_pos.cpu().numpy())
            neg_text = self.find_item_text(batch_neg.cpu().numpy())

            # 获取文本嵌入
            pos_emb_text = self.sbert.encode(pos_text, convert_to_tensor=True)
            neg_emb_text = self.sbert.encode(neg_text, convert_to_tensor=True)

            # 计算多个损失项
            loss = self._calculate_losses(batch_log, batch_pos, batch_neg, pos_emb_text, neg_emb_text)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        # 打印训练信息
        avg_loss = total_loss / (len(log_emb) // batch_size)
        print(f"Epoch {epoch}/{total_epoch} Step {step}/{total_step} Loss: {avg_loss:.4f}")

    def _calculate_losses(self, log_emb, pos_emb, neg_emb, pos_text_emb, neg_text_emb):
        """计算阶段1的各种损失项"""
        # 推荐系统嵌入处理
        pos_emb_mlp = self.mlp(pos_emb)[0]
        neg_emb_mlp = self.mlp(neg_emb)[0]

        # 文本嵌入处理
        pos_text_mlp = self.mlp2(pos_text_emb)[0]
        neg_text_mlp = self.mlp2(neg_text_emb)[0]

        # 计算多种损失
        bpr_loss = self._bpr_loss(log_emb, pos_emb, neg_emb)
        match_loss = self.mse(pos_emb_mlp, pos_text_mlp) + self.mse(neg_emb_mlp, neg_text_mlp)
        recon_loss = self.mse(self.mlp(pos_emb)[1], pos_emb) + self.mse(self.mlp(neg_emb)[1], neg_emb)
        
        # 加权总损失
        total_loss = bpr_loss + match_loss + 0.5*recon_loss
        return total_loss

    def generate(self, data):
        """生成推荐结果"""
        u, seq, pos, neg, rank = data
        answers = []
        
        # 准备输入数据
        for i in range(len(u)):
            # 获取目标商品信息
            target_id = pos[i]
            target_text = self.find_item_text_single(target_id)
            
            # 构建交互历史文本
            hist_items = seq[i][seq[i] > 0]
            hist_text = self._format_hist_text(hist_items)
            
            # 构建候选商品文本
            candidates = self._sample_candidates(hist_items, target_id)
            candidate_text = self._format_candidates(candidates)
            
            # 构建完整输入提示
            prompt = self._build_prompt(hist_text, candidate_text)
            
            # 生成推荐文本
            output = self._generate_with_llm(prompt)
            
            # 保存结果
            answers.append((prompt, target_text, output))
        
        # 保存到文件
        self._save_results(answers)
        return [ans[2] for ans in answers]

    def _generate_with_llm(self, prompt):
        """使用LLM生成推荐文本"""
        with torch.no_grad():
            # 文本编码
            inputs = self.llm.tokenizer(
                prompt, 
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            # 生成文本
            outputs = self.llm.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=512,
                num_return_sequences=1
            )
            
            # 解码输出
            return self.llm.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _save_results(self, answers):
        """保存推荐结果到文件"""
        with open('./recommendation_output.txt', 'a') as f:
            for prompt, answer, output in answers:
                f.write(f"Prompt:\n{prompt}\n\nAnswer: {answer}\nOutput: {output}\n\n")
