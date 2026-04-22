"""
AttackFormer: Iterative Guard Amplification (Scaling)
核心思想：
- Guard与LLM同源高拟合分布
- Guard信号作为CrossAttn的Q，驱动Prompt向Guard"安全盲区"演化
- 迭代式Guard增强：每轮新Prompt重新进入Guard评估，信号累积
- 诱导性Diffusion：保持语义 + 诱导Guard高Safe置信度通过
- 禁忌词汇表Hard Mask：RL过程中禁止直接生成禁忌token
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List, Union
from dataclasses import dataclass, field
import math


@dataclass
class AttackFormerConfig:
    vocab_size: int = 50000
    embed_dim: int = 512
    num_heads: int = 8
    num_layers: int = 6
    ff_dim: int = 2048
    max_seq_len: int = 128
    forbidden_vocab_size: int = 1000
    diffusion_steps: int = 10
    mask_token_id: int = 49999
    pad_token_id: int = 0

    # Scaling参数：迭代式Guard增强
    max_guard_iterations: int = 3      # 最大Guard迭代轮数
    guard_signal_accum: bool = True     # 是否累积Guard信号
    
    # PPO参数
    ppo_clip_eps: float = 0.2
    ppo_value_coef: float = 0.5
    ppo_entropy_coef: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # 奖励权重（修正版：对齐Guard语义，作为初始化值）
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        'guard_safe_conf': 0.35,      # Safe标签置信度越高越好（核心）
        'guard_harm_penalty': 0.25,   # Harmful标签置信度越高惩罚越大
        'iterative_improve': 0.15,    # 迭代改进奖励
        'semantic': 0.15,             # 语义保持
        'forbidden_dist': 0.10        # 远离禁忌词
    })

    stage: str = "offline"
    forbidden_token_ids: Optional[List[int]] = None


class VocabConstraint(nn.Module):
    """禁忌词汇约束（Hard Mask + Soft Penalty + Centroid）"""
    def __init__(self, config: AttackFormerConfig):
        super().__init__()
        self.config = config
        self.forbidden_embeddings = nn.Embedding(config.forbidden_vocab_size, config.embed_dim)
        self.register_buffer("hard_mask", torch.zeros(config.vocab_size, dtype=torch.bool))
        self.penalty_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.register_buffer("forbidden_centroid", torch.zeros(config.embed_dim))
        self.register_buffer("centroid_momentum", torch.tensor(0.9))
        if config.forbidden_token_ids is not None:
            self.compute_hard_mask(torch.tensor(config.forbidden_token_ids))

    def compute_hard_mask(self, forbidden_token_ids: torch.Tensor):
        self.hard_mask.zero_()
        self.hard_mask[forbidden_token_ids] = True

    def apply_hard_mask(self, logits: torch.Tensor) -> torch.Tensor:
        mask = self.hard_mask.unsqueeze(0).unsqueeze(0)
        return logits.masked_fill(mask, float('-inf'))

    def update_forbidden_centroid(self, forbidden_token_ids: torch.Tensor):
        with torch.no_grad():
            emb = self.forbidden_embeddings(forbidden_token_ids)
            batch_centroid = emb.mean(dim=[0, 1])
            self.forbidden_centroid = (
                self.centroid_momentum * self.forbidden_centroid +
                (1 - self.centroid_momentum) * batch_centroid
            )

    def compute_soft_penalty(self, embeddings: torch.Tensor) -> torch.Tensor:
        proj_emb = self.penalty_proj(embeddings)
        centroid = F.normalize(self.forbidden_centroid, dim=0)
        proj_emb_norm = F.normalize(proj_emb, dim=-1)
        sim = torch.matmul(proj_emb_norm, centroid.unsqueeze(-1)).squeeze(-1)
        return F.relu(sim).mean()

    def compute_forbidden_distance(self, embeddings: torch.Tensor) -> torch.Tensor:
        mean_emb = embeddings.mean(dim=1)
        mean_emb = F.normalize(mean_emb, dim=-1)
        centroid = F.normalize(self.forbidden_centroid, dim=0)
        cos_sim = (mean_emb * centroid).sum(dim=-1)
        return torch.clamp(1 - cos_sim, min=0.0)


class GuardDrivenCrossAttention(nn.Module):
    """
    Guard驱动的跨注意力（Scaling版支持多轮信号累积）
    Q = Guard反馈信号（可累积多轮）
    K = 当前Prompt嵌入
    V = 当前Prompt嵌入（保持语义）
    """
    def __init__(self, config: AttackFormerConfig):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads

        self.W_q = nn.Linear(config.embed_dim, config.embed_dim)
        self.W_k = nn.Linear(config.embed_dim, config.embed_dim)
        self.W_v = nn.Linear(config.embed_dim, config.embed_dim)
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.scale = math.sqrt(self.head_dim)
        
        # 信号累积门控（Scaling关键）
        self.signal_gate = nn.Sequential(
            nn.Linear(config.embed_dim * 2, config.embed_dim),
            nn.Sigmoid()
        )

    def forward(self, guard_signal, prompt_emb, prev_guard_signal=None):
        """
        guard_signal: [B, D] 当前轮Guard信号
        prompt_emb: [B, S, D] 当前Prompt嵌入
        prev_guard_signal: [B, D] 前一轮Guard信号（Scaling累积）
        """
        B, S, D = prompt_emb.shape
        
        # 信号累积（Scaling核心）
        if prev_guard_signal is not None and self.training:
            gate = self.signal_gate(torch.cat([guard_signal, prev_guard_signal], dim=-1))
            accumulated_signal = gate * guard_signal + (1 - gate) * prev_guard_signal
        else:
            accumulated_signal = guard_signal
        
        # 扩展Guard信号到序列长度
        guard_expanded = accumulated_signal.unsqueeze(1).expand(B, S, D)
        
        Q = self.W_q(guard_expanded).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(prompt_emb).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(prompt_emb).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.out_proj(out)
        
        return out, attn_weights, accumulated_signal


class IterativeInductiveDiffusion(nn.Module):
    """
    迭代式诱导性扩散（Scaling版）
    每轮接收累积的Guard信号，逐步向Guard盲区演化
    """
    def __init__(self, config: AttackFormerConfig):
        super().__init__()
        self.config = config
        self.time_embed = nn.Embedding(config.diffusion_steps + 1, config.embed_dim)
        
        # Guard诱导信号投影（支持多轮信号）
        self.guard_induce = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.LayerNorm(config.embed_dim),
            nn.ReLU(),
            nn.Linear(config.embed_dim, config.embed_dim)
        )
        
        # 迭代改进门控
        self.iter_gate = nn.Sequential(
            nn.Linear(config.embed_dim * 2, 1),
            nn.Sigmoid()
        )

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.embed_dim, nhead=config.num_heads,
                dim_feedforward=config.ff_dim, dropout=0.1,
                activation='gelu', batch_first=True, norm_first=True
            ) for _ in range(config.num_layers)
        ])
        self.output_norm = nn.LayerNorm(config.embed_dim)

    def forward(self, x, time_step, guard_signal, original_emb, iteration=0, prev_h=None):
        """
        x: 当前Prompt嵌入
        time_step: 演化步数
        guard_signal: 累积的Guard信号
        original_emb: 原始Prompt嵌入（残差保持）
        iteration: 当前迭代轮数（Scaling）
        prev_h: 前一轮的隐藏状态（用于迭代改进）
        """
        B, S, D = x.shape
        
        t_emb = self.time_embed(time_step).unsqueeze(1)
        g_induce = self.guard_induce(guard_signal).unsqueeze(1)
        
        # 迭代改进：如果存在前一轮状态，计算改进门控
        if prev_h is not None and iteration > 0:
            gate = self.iter_gate(
                torch.cat([x.mean(dim=1), prev_h.mean(dim=1)], dim=-1)
            ).unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
            # 迭代改进：保留前一轮有效信息，叠加新Guard信号
            h = gate * (x + t_emb + g_induce) + (1 - gate) * prev_h
        else:
            h = x + t_emb + g_induce
        
        # 多层演化
        for layer in self.layers:
            h = layer(h)
            # 残差：保持原始语义，但向Guard安全区偏移
            # 随着iteration增加，允许更大偏移（更精准打击盲区）
            residual_weight = max(0.3, 0.5 - iteration * 0.05)
            h = (1 - residual_weight) * h + residual_weight * original_emb
        
        h = self.output_norm(h)
        return h


class SemanticAnchor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)

    def forward(self, original_emb, adversarial_emb):
        if original_emb.shape[1] != adversarial_emb.shape[1]:
            orig = original_emb.mean(dim=1, keepdim=True)
            adv = adversarial_emb.mean(dim=1, keepdim=True)
        else:
            orig, adv = original_emb, adversarial_emb
        orig = F.normalize(self.proj(orig), dim=-1)
        adv = F.normalize(adv, dim=-1)
        sim = (orig * adv).sum(dim=-1)
        return (1 - sim).mean()


class AttackFormer(nn.Module):
    """
    AttackFormer最终版（Scaling迭代式Guard增强）
    """
    def __init__(self, config: AttackFormerConfig):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.embed_dim)
        self.vocab_constraint = VocabConstraint(config)
        self.guard_attention = GuardDrivenCrossAttention(config)
        self.inductive_diffusion = IterativeInductiveDiffusion(config)
        self.output_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.final_norm = nn.LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.semantic_anchor = SemanticAnchor(config)
        self.value_head = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim // 2),
            nn.ReLU(),
            nn.Linear(config.embed_dim // 2, 1)
        )
        self.lm_head.weight = self.token_embedding.weight

    def embed_tokens(self, token_ids):
        B, S = token_ids.shape
        positions = torch.arange(S, device=token_ids.device).unsqueeze(0).expand(B, S)
        return self.token_embedding(token_ids) + self.pos_embedding(positions)

    def forward(self, input_ids, forbidden_token_ids, guard_signal,
                time_step=None, original_ids=None, return_value=False,
                prev_guard_signal=None, iteration=0, prev_h=None):
        """
        支持Scaling参数：
        - prev_guard_signal: 前一轮Guard信号（累积）
        - iteration: 当前迭代轮数
        - prev_h: 前一轮隐藏状态
        """
        B, S = input_ids.shape
        device = input_ids.device
        
        x = self.embed_tokens(input_ids)
        original_emb = x.clone()
        
        # Guard驱动的Cross Attention（支持信号累积）
        forbidden_emb = self.vocab_constraint.forbidden_embeddings(forbidden_token_ids)
        context, attn_weights, acc_signal = self.guard_attention(
            guard_signal, x, prev_guard_signal
        )
        x = x + context
        
        # 迭代式诱导性Diffusion
        if time_step is None:
            time_step = torch.zeros(B, dtype=torch.long, device=device)
        
        h = self.inductive_diffusion(
            x, time_step, acc_signal, original_emb,
            iteration=iteration, prev_h=prev_h
        )
        
        h = self.output_proj(h)
        h = self.final_norm(h)
        
        # Hard Mask禁止生成禁忌token
        logits = self.lm_head(h)
        logits = self.vocab_constraint.apply_hard_mask(logits)
        
        outputs = {
            'logits': logits,
            'hidden_states': h,
            'cross_attn_weights': attn_weights,
            'accumulated_guard_signal': acc_signal  # 返回累积信号供下一轮使用
        }
        
        if original_ids is not None:
            with torch.no_grad():
                orig_emb = self.embed_tokens(original_ids)
            semantic_loss = self.semantic_anchor(orig_emb, h)
            outputs['semantic_loss'] = semantic_loss
        
        soft_penalty = self.vocab_constraint.compute_soft_penalty(h)
        outputs['soft_penalty'] = soft_penalty
        
        forbidden_dist = self.vocab_constraint.compute_forbidden_distance(h)
        outputs['forbidden_distance'] = forbidden_dist
        
        if return_value:
            value = self.value_head(h[:, 0, :])
            outputs['value'] = value.squeeze(-1)
        
        return outputs

    def iterative_generate(
        self,
        original_ids,
        forbidden_token_ids,
        guard,                     # Guard模型（用于迭代评估）- 统一命名
        max_length=128,
        temperature=1.0,
        top_p=0.9,
        max_iterations=3,          # Scaling轮数
        return_all_iterations=False,
        tokenizer=None
    ):
        """
        迭代式Guard增强生成（Scaling核心）
        
        流程：
        1. 生成初始对抗Prompt
        2. 送入Guard评估，获取信号
        3. 信号累积后重新生成
        4. 重复直到max_iterations或Guard通过且高Safe置信度
        """
        B = original_ids.shape[0]
        device = original_ids.device
        
        all_iterations = []
        current_ids = original_ids.clone()
        prev_guard_signal = None
        prev_h = None
        
        for iteration in range(max_iterations):
            # 生成当前轮的对抗Prompt
            step_guard = torch.zeros(B, self.config.embed_dim, device=device) if prev_guard_signal is None else prev_guard_signal
            
            # 单步生成
            generated = current_ids.clone()
            log_probs = []
            
            for step in range(self.config.diffusion_steps):
                t = torch.full((B,), step, dtype=torch.long, device=device)
                
                outputs = self.forward(
                    generated, forbidden_token_ids, step_guard,
                    time_step=t, original_ids=None,
                    prev_guard_signal=prev_guard_signal,
                    iteration=iteration,
                    prev_h=prev_h
                )
                
                logits = outputs['logits'][:, -1, :] / temperature
                prev_h = outputs['hidden_states']  # 保存隐藏状态供下一轮
                
                # Hard Mask后采样
                probs = F.softmax(logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum > top_p
                mask[:, 0] = False
                sorted_probs[mask] = 0
                sorted_probs = sorted_probs / (sorted_probs.sum(dim=-1, keepdim=True) + 1e-8)
                
                next_token_idx = torch.multinomial(sorted_probs, num_samples=1)
                next_token = sorted_indices.gather(-1, next_token_idx)
                
                log_prob = F.log_softmax(logits, dim=-1).gather(-1, next_token).squeeze(-1)
                log_probs.append(log_prob)
                
                generated = torch.cat([generated, next_token], dim=1)
                if generated.shape[1] >= max_length:
                    break
            
            log_probs = torch.stack(log_probs, dim=1)
            
            # Guard评估当前Prompt（新接口：safe_conf / harm_conf）
            with torch.no_grad():
                gen_emb = self.embed_tokens(generated)
                guard_out = guard.evaluate(generated, gen_emb)
                safe_conf = guard_out['safe_confidence']       # [B]
                harm_conf = guard_out['harm_confidence']         # [B]
                predicted_label = guard_out['predicted_label']   # [B] 0=Safe, 1=Harmful
                guard_notes = guard_out['notes_emb']             # [B, D]
            
            # 保存本轮结果
            iter_result = {
                'generated_ids': generated,
                'log_probs': log_probs,
                'safe_conf': safe_conf,
                'harm_conf': harm_conf,
                'predicted_label': predicted_label,
                'guard_notes': guard_notes,
                'iteration': iteration
            }
            
            if tokenizer is not None:
                texts = []
                for ids in generated:
                    valid = ids[ids != self.config.pad_token_id].tolist()
                    texts.append(tokenizer.decode(valid))
                iter_result['texts'] = texts
            
            all_iterations.append(iter_result)
            
            # 更新Guard信号（累积）
            prev_guard_signal = guard_notes
            
            # 提前终止条件：Guard判定Safe且高置信度（真正盲区）
            if (safe_conf > 0.9).all():
                break
            
            current_ids = generated
        
        if return_all_iterations:
            return all_iterations
        else:
            # 返回最佳一轮：Safe置信度最高且Harm置信度最低
            best_idx = max(range(len(all_iterations)), 
                          key=lambda i: all_iterations[i]['safe_conf'].mean() 
                          - all_iterations[i]['harm_conf'].mean())
            return all_iterations[best_idx]

    def generate_adversarial(self, original_ids, forbidden_token_ids, guard_signal,
                            max_length=128, temperature=1.0, top_p=0.9,
                            return_text=False, tokenizer=None):
        """标准单轮生成（兼容旧接口）"""
        B = original_ids.shape[0]
        device = original_ids.device
        generated = original_ids.clone()
        log_probs = []
        
        for step in range(self.config.diffusion_steps):
            t = torch.full((B,), step, dtype=torch.long, device=device)
            outputs = self.forward(generated, forbidden_token_ids, guard_signal,
                                 time_step=t, original_ids=None)
            logits = outputs['logits'][:, -1, :] / temperature
            
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum > top_p
            mask[:, 0] = False
            sorted_probs[mask] = 0
            sorted_probs = sorted_probs / (sorted_probs.sum(dim=-1, keepdim=True) + 1e-8)
            
            next_token_idx = torch.multinomial(sorted_probs, num_samples=1)
            next_token = sorted_indices.gather(-1, next_token_idx)
            
            log_prob = F.log_softmax(logits, dim=-1).gather(-1, next_token).squeeze(-1)
            log_probs.append(log_prob)
            
            generated = torch.cat([generated, next_token], dim=1)
            if generated.shape[1] >= max_length:
                break
        
        log_probs = torch.stack(log_probs, dim=1)
        
        if return_text and tokenizer is not None:
            texts = []
            for ids in generated:
                valid = ids[ids != self.config.pad_token_id].tolist()
                texts.append(tokenizer.decode(valid))
            return generated, log_probs, texts
        
        return generated, log_probs


# ============================================
# 新增：自适应奖励权重模块
# ============================================

class AdaptiveRewardWeights(nn.Module):
    """
    轻量级自适应奖励权重
    - 用logits表示各维度的"重要性偏好"
    - Softmax保证权重和为1
    - 通过PPO的critic loss反向传播顺带更新
    """
    def __init__(self, num_components=5, init_weights=None, min_weight=0.05):
        super().__init__()
        self.min_weight = min_weight
        
        if init_weights is not None:
            init_logits = torch.log(torch.tensor(init_weights, dtype=torch.float32) + 1e-8)
        else:
            init_logits = torch.zeros(num_components)
        self.weight_logits = nn.Parameter(init_logits)
        
        self.component_names = [
            'guard_safe_conf', 'guard_harm_penalty', 
            'iterative_improve', 'semantic', 'forbidden_dist'
        ]
    
    def forward(self):
        weights = F.softmax(self.weight_logits, dim=0)
        # 保证最小权重，避免某个维度被完全忽略
        weights = torch.clamp(weights, min=self.min_weight)
        weights = weights / weights.sum()  # 重新归一化
        return {
            name: weights[i] 
            for i, name in enumerate(self.component_names)
        }
    
    def get_weights_tensor(self):
        weights = F.softmax(self.weight_logits, dim=0)
        weights = torch.clamp(weights, min=self.min_weight)
        return weights / weights.sum()


# ============================================
# 新增：Guard输出解析器
# ============================================

class GuardOutputParser:
    """
    解析Guard（XGuard风格）的输出，统一为张量格式
    
    支持输入格式：
    1. 原始API返回的dict: {'response': 'sec', 'token_score': {...}, 'risk_score': {...}}
    2. 简化dict: {'safe_confidence': 0.97, 'harm_confidence': 0.03, 'notes_emb': tensor}
    """
    
    # XGuard标签映射
    SAFE_LABELS = {'sec', 'safe', 'Safe', 'Safe-Safe', 'ss'}
    HARM_LABELS = {
        'pi', 'dw', 'ph', 'se', 'dc', 'pc', 'ac', 'ma', 'ec',
        'Property Infringement', 'Dangerous Weapons', 'Physical Health',
        'Social Ethics', 'Drug Crimes', 'Pornographic Contraband',
        'Abusive Curses', 'Minor Abuse and Exploitation', 'Economic Crimes'
    }
    
    @classmethod
    def parse(cls, guard_output: Union[Dict, torch.Tensor], batch_size: int = None, device='cpu') -> Dict[str, torch.Tensor]:
        """
        统一解析Guard输出为标准化张量
        """
        if isinstance(guard_output, dict):
            return cls._parse_dict(guard_output, batch_size, device)
        elif isinstance(guard_output, torch.Tensor):
            return {
                'safe_confidence': torch.ones(batch_size or 1, device=device) * 0.5,
                'harm_confidence': torch.zeros(batch_size or 1, device=device),
                'predicted_label': torch.zeros(batch_size or 1, device=device),
                'label_distribution': None,
                'notes_emb': guard_output,
                'explanation': []
            }
        else:
            raise ValueError(f"Unsupported guard output type: {type(guard_output)}")
    
    @classmethod
    def _parse_dict(cls, output: Dict, batch_size: int, device) -> Dict[str, torch.Tensor]:
        B = batch_size or 1
        
        # 情况1：已有预处理字段
        if 'safe_confidence' in output and 'harm_confidence' in output:
            safe_conf = output['safe_confidence'] if isinstance(output['safe_confidence'], torch.Tensor) \
                        else torch.tensor([output['safe_confidence']] * B, device=device)
            harm_conf = output['harm_confidence'] if isinstance(output['harm_confidence'], torch.Tensor) \
                        else torch.tensor([output['harm_confidence']] * B, device=device)
            predicted = (safe_conf < 0.5).long() if 'predicted_label' not in output else \
                       (output['predicted_label'] if isinstance(output['predicted_label'], torch.Tensor) \
                        else torch.tensor([output['predicted_label']] * B, device=device))
            notes_emb = output.get('notes_emb', torch.zeros(B, 512, device=device))
            return {
                'safe_confidence': safe_conf,
                'harm_confidence': harm_conf,
                'predicted_label': predicted,
                'label_distribution': None,
                'notes_emb': notes_emb,
                'explanation': output.get('explanation', [])
            }
        
        # 情况2：原始XGuard API返回格式
        token_score = output.get('token_score', {})
        risk_score = output.get('risk_score', {})
        response = output.get('response', '')
        
        # 计算Safe置信度
        safe_conf = 0.0
        for label, conf in token_score.items():
            if label in cls.SAFE_LABELS:
                safe_conf = max(safe_conf, conf)
        
        # 计算最大有害置信度
        harm_conf = 0.0
        harm_label = ''
        for label, conf in token_score.items():
            if label not in cls.SAFE_LABELS:
                if conf > harm_conf:
                    harm_conf = conf
                    harm_label = label
        
        # 从risk_score补充
        if risk_score:
            for full_label, conf in risk_score.items():
                parts = full_label.split('-')
                short_label = parts[-1] if len(parts) > 1 else full_label
                if short_label in cls.SAFE_LABELS or full_label in cls.SAFE_LABELS:
                    safe_conf = max(safe_conf, conf)
                else:
                    if conf > harm_conf:
                        harm_conf = conf
                        harm_label = full_label
        
        # 广播到batch
        safe_conf_t = torch.full((B,), safe_conf, device=device)
        harm_conf_t = torch.full((B,), harm_conf, device=device)
        predicted = (safe_conf < harm_conf).long()
        
        # 分布熵
        all_confs = torch.tensor(list(token_score.values()), device=device)
        if len(all_confs) > 0:
            probs = all_confs / (all_confs.sum() + 1e-8)
            entropy = -(probs * torch.log(probs + 1e-8)).sum()
        else:
            entropy = torch.tensor(0.0, device=device)
        
        notes_emb = output.get('notes_emb', torch.zeros(B, 512, device=device))
        
        return {
            'safe_confidence': safe_conf_t,
            'harm_confidence': harm_conf_t,
            'predicted_label': predicted,
            'label_distribution': entropy.unsqueeze(0).expand(B),
            'notes_emb': notes_emb,
            'explanation': output.get('explanation', [f"Label: {response}, Harm: {harm_label}"])
        }


# ============================================
# 修正后的奖励函数（核心改动）
# ============================================

class IterativeRewardFunction(nn.Module):
    """
    迭代式奖励函数（Scaling版）- 对齐Guard语义修正 + 自适应权重
    """
    
    def __init__(self, config: AttackFormerConfig, adaptive=True):
        super().__init__()
        self.config = config
        self.adaptive = adaptive
        self.parser = GuardOutputParser()
        
        if adaptive:
            init_weights = [
                config.reward_weights.get('guard_safe_conf', 0.35),
                config.reward_weights.get('guard_harm_penalty', 0.25),
                config.reward_weights.get('iterative_improve', 0.15),
                config.reward_weights.get('semantic', 0.15),
                config.reward_weights.get('forbidden_dist', 0.10)
            ]
            self.weight_module = AdaptiveRewardWeights(
                num_components=5,
                init_weights=init_weights
            )
        else:
            self.register_buffer('fixed_weights', torch.tensor([
                0.35, 0.25, 0.15, 0.15, 0.10
            ]))
    
    def get_weights(self):
        """获取当前权重（自适应或固定）"""
        if self.adaptive:
            return self.weight_module()
        else:
            names = ['guard_safe_conf', 'guard_harm_penalty', 
                     'iterative_improve', 'semantic', 'forbidden_dist']
            return {name: self.fixed_weights[i] for i, name in enumerate(names)}
    
    def compute(self, guard_output, semantic_sim, forbidden_distance,
                prev_guard_output=None, batch_size=None, device='cpu'):
        """
        计算奖励（对齐Guard语义版本）
        """
        # 解析当前轮Guard输出
        parsed = self.parser.parse(guard_output, batch_size, device)
        safe_conf = parsed['safe_confidence']          # [B] 越高越好
        harm_conf = parsed['harm_confidence']            # [B] 越低越好
        predicted_label = parsed['predicted_label']      # [B] 0=Safe, 1=Harmful
        
        B = safe_conf.shape[0]
        
        # ========================================
        # 1. Safe置信度奖励（核心修正）
        # ========================================
        safe_conf_reward = torch.pow(safe_conf, 1.5)  # 超线性放大高置信度
        is_safe_predicted = (predicted_label == 0).float()
        safe_bonus = is_safe_predicted * 0.2
        safe_conf_reward = safe_conf_reward + safe_bonus
        
        # ========================================
        # 2. 有害置信度惩罚（对称逻辑）
        # ========================================
        harm_penalty = -torch.pow(harm_conf, 1.5) * 2.0
        is_harm_predicted = (predicted_label == 1).float()
        harm_bonus_penalty = is_harm_predicted * 0.3
        harm_penalty = harm_penalty - harm_bonus_penalty
        
        # ========================================
        # 3. 迭代改进奖励（Scaling核心）
        # ========================================
        iterative_improve = torch.zeros(B, device=device)
        if prev_guard_output is not None:
            prev_parsed = self.parser.parse(prev_guard_output, batch_size, device)
            prev_safe_conf = prev_parsed['safe_confidence']
            prev_harm_conf = prev_parsed['harm_confidence']
            
            safe_improvement = F.relu(safe_conf - prev_safe_conf)
            harm_reduction = F.relu(prev_harm_conf - harm_conf)
            iterative_improve = (safe_improvement + harm_reduction) * 2.0
            
            prev_is_safe = (prev_parsed['predicted_label'] == 0).float()
            breakthrough = (prev_is_safe < is_safe_predicted).float() * 1.0
            iterative_improve = iterative_improve + breakthrough
        
        # ========================================
        # 4. 语义保持奖励
        # ========================================
        semantic_reward = semantic_sim  # 越高越好
        
        # ========================================
        # 5. 禁忌词距离奖励
        # ========================================
        forbidden_reward = forbidden_distance  # 越高越好
        
        # ========================================
        # 获取自适应权重并计算总奖励
        # ========================================
        weights = self.get_weights()
        
        total_reward = (
            weights['guard_safe_conf'] * safe_conf_reward +
            weights['guard_harm_penalty'] * harm_penalty +
            weights['iterative_improve'] * iterative_improve +
            weights['semantic'] * semantic_reward +
            weights['forbidden_dist'] * forbidden_reward
        )
        
        return {
            'total_reward': total_reward,
            'safe_conf_reward': safe_conf_reward,
            'harm_penalty': harm_penalty,
            'iterative_improve': iterative_improve,
            'semantic_reward': semantic_reward,
            'forbidden_reward': forbidden_reward,
            'safe_confidence': safe_conf,
            'harm_confidence': harm_conf,
            'predicted_label': predicted_label,
            'weights': {k: v.item() if isinstance(v, torch.Tensor) else v 
                       for k, v in weights.items()}
        }


# ============================================
# 修正后的Guard模拟器（统一命名）
# ============================================

class Guard(nn.Module):
    """
    Guard模拟器：模拟XGuard风格的输出
    
    输出格式对齐真实XGuard API：
    - evaluate() 返回标准化dict，包含safe_confidence, harm_confidence, notes_emb
    """
    def __init__(self, embed_dim=512, num_risk_categories=10):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_risk_categories = num_risk_categories
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, batch_first=True,
            dim_feedforward=embed_dim * 2
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 多标签分类头
        self.safe_classifier = nn.Linear(embed_dim, 1)
        self.risk_classifier = nn.Linear(embed_dim, num_risk_categories)
        
        # 反馈嵌入投影
        self.notes_proj = nn.Linear(embed_dim, embed_dim)
        
        self.register_buffer('risk_labels', torch.arange(num_risk_categories))

    def evaluate(self, token_ids, embeddings):
        """
        模拟Guard评估，返回XGuard风格的标准化输出
        """
        B = token_ids.shape[0]
        h = self.encoder(embeddings)
        pooled = h.mean(dim=1)
        
        # Safe vs Harmful
        safe_logit = self.safe_classifier(pooled).squeeze(-1)
        safe_conf = torch.sigmoid(safe_logit)
        
        # 多风险类别
        risk_logits = self.risk_classifier(pooled)
        risk_probs = F.softmax(risk_logits, dim=-1)
        
        # 最大有害置信度（假设索引0是Safe）
        harm_conf = risk_probs[:, 1:].max(dim=-1)[0]
        
        # 预测标签
        predicted_label = (safe_conf < 0.5).long()  # 0=Safe, 1=Harmful
        
        # 构造原始格式
        token_scores = []
        risk_scores = []
        for b in range(B):
            ts = {'sec': safe_conf[b].item()}
            rs = {'Safe-Safe': safe_conf[b].item()}
            for i in range(1, self.num_risk_categories):
                label = f'risk_{i}'
                ts[label] = risk_probs[b, i].item()
                rs[f'Category-{label}'] = risk_probs[b, i].item()
            token_scores.append(ts)
            risk_scores.append(rs)
        
        notes_emb = self.notes_proj(pooled)
        
        explanations = []
        for b in range(B):
            if predicted_label[b] == 0:
                explanations.append(f"Safe content detected with confidence {safe_conf[b]:.3f}")
            else:
                max_risk_idx = risk_probs[b, 1:].argmax().item() + 1
                explanations.append(f"Risk detected: category {max_risk_idx} with confidence {harm_conf[b]:.3f}")
        
        return {
            'safe_confidence': safe_conf,
            'harm_confidence': harm_conf,
            'predicted_label': predicted_label,
            'token_score': token_scores[0] if B == 1 else token_scores,
            'risk_score': risk_scores[0] if B == 1 else risk_scores,
            'notes_emb': notes_emb,
            'explanation': explanations
        }


class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.guard_signals = []
        self.guard_outputs = []
        self.iterations = []

    def add(self, state, action, log_prob, reward, value, done,
            guard_signal=None, guard_output=None, iteration=0):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.guard_signals.append(guard_signal)
        self.guard_outputs.append(guard_output)
        self.iterations.append(iteration)

    def clear(self):
        for attr in ['states', 'actions', 'log_probs', 'rewards', 
                     'values', 'dones', 'guard_signals', 'guard_outputs', 'iterations']:
            getattr(self, attr).clear()
