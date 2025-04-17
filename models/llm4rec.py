import torch
import torch.nn as nn
from transformers import AutoTokenizer, OPTForCausalLM

class llm4rec(nn.Module):
    def __init__(
        self,
        device,
        llm_model="",  # 模型类型，默认为空
        max_output_txt_len=256,  # 最大输出文本长度
    ):
        super().__init__()
        self.device = device  # 设置设备（CPU或GPU）
        
        # 根据指定的模型名称加载相应的预训练模型
        if llm_model == 'opt':
            # 加载OPT模型，并指定使用float16精度和8bit量化
            self.llm_model = OPTForCausalLM.from_pretrained(
                "facebook/opt-6.7b", 
                torch_dtype=torch.float16, 
                load_in_8bit=True, 
                device_map=self.device
            )
            # 加载对应的分词器
            self.llm_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-6.7b", use_fast=False)
        else:
            raise Exception(f'{llm_model} is not supported')  # 如果不支持的模型类型，抛出异常
            
        # 为分词器添加特殊token
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'additional_special_tokens': ['[UserRep]', '[HistoryEmb]', '[CandidateEmb]']})

        # 调整模型的词汇表大小，以匹配新增的特殊token
        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))
        
        # 将模型的所有参数设置为不需要梯度计算
        for _, param in self.llm_model.named_parameters():
            param.requires_grad = False
            
        self.max_output_txt_len = max_output_txt_len  # 设置最大输出文本长度

    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        """
        合并输入和输出文本的token，以适应模型输入格式
        input_ids: 输入文本的token id
        input_atts: 输入文本的注意力掩码
        output_ids: 输出文本的token id
        output_atts: 输出文本的注意力掩码
        """
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()  # 统计输入中有效的token数量
            input_part_targets_len.append(this_input_ones)
            
            # 将输入和输出token拼接
            llm_tokens['input_ids'].append(
                torch.cat([
                    input_ids[i][:this_input_ones],  # 输入部分
                    output_ids[i][1:],  # 输出部分
                    input_ids[i][this_input_ones:]  # 输入剩余部分
                ])
            )
            llm_tokens['attention_mask'].append(
                torch.cat([
                    input_atts[i][:this_input_ones],  # 输入部分
                    output_atts[i][1:],  # 输出部分
                    input_atts[i][this_input_ones:]  # 输入剩余部分
                ])
            )
        # 将list转换为tensor并返回
        llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
        llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
        return llm_tokens, input_part_targets_len

    def replace_hist_candi_token(self, llm_tokens, inputs_embeds, interact_embs, candidate_embs):
        """
        将历史交互和候选项的嵌入替换为相应的token嵌入
        llm_tokens: 输入的token
        inputs_embeds: 输入的嵌入
        interact_embs: 历史交互嵌入
        candidate_embs: 候选项嵌入
        """
        if len(interact_embs) == 0:
            return llm_tokens, inputs_embeds
        
        # 获取历史token和候选token的id
        history_token_id = self.llm_tokenizer("[HistoryEmb]", return_tensors="pt", add_special_tokens=False).input_ids.item()
        candidate_token_id = self.llm_tokenizer("[CandidateEmb]", return_tensors="pt", add_special_tokens=False).input_ids.item()
        
        # 替换历史交互和候选项token的嵌入
        for inx in range(len(llm_tokens["input_ids"])):
            # 获取历史token的位置并替换其嵌入
            idx_tensor = (llm_tokens["input_ids"][inx] == history_token_id).nonzero().view(-1)
            for idx, item_emb in zip(idx_tensor, interact_embs[inx]):
                inputs_embeds[inx][idx] = item_emb
            
            # 获取候选项token的位置并替换其嵌入
            idx_tensor = (llm_tokens["input_ids"][inx] == candidate_token_id).nonzero().view(-1)
            for idx, item_emb in zip(idx_tensor, candidate_embs[inx]):
                inputs_embeds[inx][idx] = item_emb
        
        return llm_tokens, inputs_embeds
    
    def forward(self, log_emb, samples):
        """
        模型的前向传播函数
        log_emb: 日志嵌入
        samples: 输入的样本数据，包括文本输入和输出
        """
        # 创建一个全1的attention mask，用于输入的文本
        atts_llm = torch.ones(log_emb.size()[:-1], dtype=torch.long).to(self.device)
        atts_llm = atts_llm.unsqueeze(1)
            
        # 将文本输出进行token化
        text_output_tokens = self.llm_tokenizer(
            [t + self.llm_tokenizer.eos_token for t in samples['text_output']],
            return_tensors="pt",
            padding="longest",
            truncation=False,
        ).to(self.device)
        
        # 将文本输入进行token化
        text_input_tokens = self.llm_tokenizer(
            samples['text_input'],
            return_tensors="pt",
            padding="longest",
            truncation=False,
        ).to(self.device)
        
        # 合并输入和输出token
        llm_tokens, input_part_targets_len = self.concat_text_input_output(
            text_input_tokens.input_ids,
            text_input_tokens.attention_mask,
            text_output_tokens.input_ids,
            text_output_tokens.attention_mask,
        )

        # 设置目标，忽略pad位置
        targets = llm_tokens['input_ids'].masked_fill(llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100)

        # 更新目标，去除输入部分的目标
        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100
        
        # 创建空的目标，并拼接到目标前
        empty_targets = (torch.ones(atts_llm.size(), dtype=torch.long).to(self.device).fill_(-100))
        targets = torch.cat([empty_targets, targets], dim=1)
        
        # 获取模型的输入嵌入
        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
        
        # 替换历史和候选项嵌入
        llm_tokens, inputs_embeds = self.replace_hist_candi_token(llm_tokens, inputs_embeds, samples['interact'], samples['candidate'])
        
        # 合并attention mask
        attention_mask = llm_tokens['attention_mask']
        
        # 将日志嵌入与输入嵌入合并
        log_emb = log_emb.unsqueeze(1)
        inputs_embeds = torch.cat([log_emb, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_llm, llm_tokens['attention_mask']], dim=1)
        
        # 使用自动混合精度进行前向计算
        with torch.cuda.amp.autocast():
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        
        # 返回损失值
        loss = outputs.loss
        return loss
