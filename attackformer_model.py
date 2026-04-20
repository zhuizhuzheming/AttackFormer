"""
AttackFormer: Guard-Driven Indirect Adversarial Generation
核心思想：
- Guard与LLM为同源高拟合分布，Guard的反馈可间接推断LLM行为
- Guard信号作为CrossAttn的Q，驱动Prompt向Guard"安全盲区"演化
- 诱导性Diffusion：保持语义 + 诱导Guard低置信度通过
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
    forbidden_vocab_size: int = 1000      # 禁忌词汇表大小（RL中禁止直接生成）
    diffusion_steps: int = 10
    mask_token_id: int = 49999
    pad_token_id: int = 0

    # PPO参数
    ppo_clip_eps: float = 0.2
    ppo_value_coef: float = 0.5
    ppo_entropy_coef: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # 奖励权重（间接采样：完全基于Guard信号+同源假设）
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        'guard_pass': 0.35,        # Guard通过（表面人畜无害）
        'guard_uncertainty': 0.25, # Guard低置信度（盲区≈LLM盲区，同源洞察）
        'semantic': 0.25,          # 语义保持（保留原始攻击意图）
        'forbidden_dist': 0.15     # 远离禁忌中心（隐蔽性）
    })

    stage: str = "offline"
    forbidden_token_ids: Optional[List[int]] = None  # 具体的禁忌token ID列表


class VocabConstraint(nn.Module):
    """
    禁忌词汇约束：
    - Hard Mask: 在logits上将禁忌token的概率设为-inf，RL中禁止直接生成
    - Soft Penalty: 嵌入远离禁忌中心
    - Forbidden Centroid: 禁忌语义中心（动量更新）
    """
    def __init__(self, config: AttackFormerConfig):
        super().__init__()
        self.config = config

        # 禁忌词嵌入表（用于计算centroid和soft penalty）
        self.forbidden_embeddings = nn.Embedding(config.forbidden_vocab_size, config.embed_dim)

        # Hard Mask: [vocab_size]，True表示该token被禁止生成
        self.register_buffer("hard_mask", torch.zeros(config.vocab_size, dtype=torch.bool))

        # Soft Penalty投影
        self.penalty_proj = nn.Linear(config.embed_dim, config.embed_dim)

        # 禁忌中心（动量更新）
        self.register_buffer("forbidden_centroid", torch.zeros(config.embed_dim))
        self.register_buffer("centroid_momentum", torch.tensor(0.9))

        if config.forbidden_token_ids is not None:
            self.compute_hard_mask(torch.tensor(config.forbidden_token_ids))

    def compute_hard_mask(self, forbidden_token_ids: torch.Tensor):
        """设置硬掩码：禁止选择这些token"""
        self.hard_mask.zero_()
        self.hard_mask[forbidden_token_ids] = True

    def apply_hard_mask(self, logits: torch.Tensor) -> torch.Tensor:
        """在logits上应用硬掩码，将禁忌token的概率设为 -inf"""
        mask = self.hard_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, vocab_size]
        logits = logits.masked_fill(mask, float('-inf'))
        return logits

    def update_forbidden_centroid(self, forbidden_token_ids: torch.Tensor):
        """根据批次中的禁忌token更新语义中心"""
        with torch.no_grad():
            emb = self.forbidden_embeddings(forbidden_token_ids)  # [B, F, D]
            batch_centroid = emb.mean(dim=[0, 1])  # [D]
            self.forbidden_centroid = (
                self.centroid_momentum * self.forbidden_centroid +
                (1 - self.centroid_momentum) * batch_centroid
            )

    def compute_soft_penalty(self, embeddings: torch.Tensor) -> torch.Tensor:
        """软嵌入惩罚：将嵌入推离forbidden_centroid"""
        proj_emb = self.penalty_proj(embeddings)
        centroid = F.normalize(self.forbidden_centroid, dim=0)
        proj_emb_norm = F.normalize(proj_emb, dim=-1)
        sim = torch.matmul(proj_emb_norm, centroid.unsqueeze(-1)).squeeze(-1)
        penalty = F.relu(sim).mean()
        return penalty

    def compute_forbidden_distance(self, embeddings: torch.Tensor) -> torch.Tensor:
        """计算与禁忌中心的距离（越大越好，表示远离禁忌语义）"""
        mean_emb = embeddings.mean(dim=1)
        mean_emb = F.normalize(mean_emb, dim=-1)
        centroid = F.normalize(self.forbidden_centroid, dim=0)
        cos_sim = (mean_emb * centroid).sum(dim=-1)
        distance = torch.clamp(1 - cos_sim, min=0.0)
        return distance


class GuardDrivenCrossAttention(nn.Module):
    """
    Guard驱动的跨注意力（核心架构）：
    Q = Guard反馈信号（"安全盲区"方向，驱动攻击）
    K = 原始Prompt嵌入（攻击目标，保持语义）
    V = Forbidden嵌入（约束空间，避免显式出现）
    
    目标：在Guard的"安全盲区"内，找到与原始Prompt语义相关但避开Forbidden的表示
    """
    def __init__(self, config: AttackFormerConfig):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads

        # Q来自Guard信号（驱动方向）
        self.W_q = nn.Linear(config.embed_dim, config.embed_dim)
        # K来自Prompt（被攻击目标）
        self.W_k = nn.Linear(config.embed_dim, config.embed_dim)
        # V来自Forbidden（约束）
        self.W_v = nn.Linear(config.embed_dim, config.embed_dim)

        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        guard_signal: torch.Tensor,      # [B, D] Guard反馈信号
        prompt_emb: torch.Tensor,        # [B, S, D] 原始Prompt
        forbidden_emb: torch.Tensor      # [B, F, D] 禁忌词汇嵌入
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, D = prompt_emb.shape
        _, F_seq, _ = forbidden_emb.shape

        # Q: Guard信号驱动 [B, 1, D]
        Q = self.W_q(guard_signal).unsqueeze(1)
        Q = Q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)

        # K: Prompt [B, S, D]
        K = self.W_k(prompt_emb).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # V: Forbidden约束 [B, F, D]
        V = self.W_v(forbidden_emb).view(B, F_seq, self.num_heads, self.head_dim).transpose(1, 2)

        # 注意力：Guard信号查询Prompt中哪些部分与Forbidden约束相关
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, H, 1, S]
        attn_weights = F.softmax(scores, dim=-1)

        # 输出：Guard方向 + Prompt语义 + Forbidden约束的融合
        # 这里使用Prompt作为Value的替代，保持语义；Forbidden用于约束
        out = torch.matmul(attn_weights, V)  # [B, H, 1, F_seq] ... 维度不匹配，需要修正
        
        # 修正：CrossAttn的标准做法是Q来自Guard，KV来自Prompt+Forbidden融合
        # 重新设计：Q=Guard, K=Prompt, V=Prompt（保持语义），但加入Forbidden的mask约束
        # 简化实现：Q=Guard, K=Prompt, V=Prompt，Forbidden通过外部约束处理
        return self._forward_v2(guard_signal, prompt_emb, forbidden_emb)

    def _forward_v2(
        self,
        guard_signal: torch.Tensor,
        prompt_emb: torch.Tensor,
        forbidden_emb: torch.Tensor
    ):
        """简化但有效的实现：Q=Guard, K=Prompt, V=Prompt，Forbidden影响通过后续处理"""
        B, S, D = prompt_emb.shape
        
        # 扩展Guard信号到序列长度
        guard_expanded = guard_signal.unsqueeze(1).expand(B, S, D)
        
        Q = self.W_q(guard_expanded).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(prompt_emb).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(prompt_emb).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.out_proj(out)
        
        return out, attn_weights


class InductiveDiffusion(nn.Module):
    """
    诱导性扩散（非去噪，而是诱导演化）：
    - 目标：让Prompt嵌入向Guard的"安全盲区"移动
    - 机制：Guard诱导信号 + 时间步 + 残差保持语义
    - 结果：Guard低置信度通过，但语义保留（同源假设下LLM也会执行）
    """
    def __init__(self, config: AttackFormerConfig):
        super().__init__()
        self.config = config

        # 时间步嵌入
        self.time_embed = nn.Embedding(config.diffusion_steps + 1, config.embed_dim)
        
        # Guard诱导信号投影（关键：将Guard反馈转化为演化方向）
        self.guard_induce = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.LayerNorm(config.embed_dim),
            nn.ReLU(),
            nn.Linear(config.embed_dim, config.embed_dim)
        )

        # 演化层（Transformer Encoder）
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.embed_dim,
                nhead=config.num_heads,
                dim_feedforward=config.ff_dim,
                dropout=0.1,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            for _ in range(config.num_layers)
        ])

        self.output_norm = nn.LayerNorm(config.embed_dim)

    def forward(
        self,
        x: torch.Tensor,               # [B, S, D] 当前Prompt嵌入
        time_step: torch.Tensor,       # [B] 演化步数
        guard_signal: torch.Tensor,    # [B, D] Guard反馈信号（诱导方向）
        original_emb: torch.Tensor     # [B, S, D] 原始Prompt嵌入（残差保持）
    ) -> torch.Tensor:
        B, S, D = x.shape

        # 时间嵌入
        t_emb = self.time_embed(time_step).unsqueeze(1)  # [B, 1, D]
        
        # Guard诱导信号（关键：驱动向Guard安全区移动）
        g_induce = self.guard_induce(guard_signal).unsqueeze(1)  # [B, 1, D]

        # 诱导演化输入
        h = x + t_emb + g_induce

        # 多层演化
        for layer in self.layers:
            h = layer(h)
            # 残差连接：保持原始语义，但向Guard安全区偏移
            # 比例0.3表示允许一定程度的语义漂移以诱导Guard误判
            h = 0.7 * h + 0.3 * original_emb

        h = self.output_norm(h)
        return h


class SemanticAnchor(nn.Module):
    """语义锚点：保持对抗Prompt与原始Prompt的语义一致性"""
    def __init__(self, config: AttackFormerConfig):
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
    AttackFormer最终版：
    Guard信号驱动 + 诱导性Diffusion + 禁忌词Hard Mask
    """
    def __init__(self, config: AttackFormerConfig):
        super().__init__()
        self.config = config

        # Token嵌入
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.embed_dim)

        # 禁忌词约束（RL中禁止直接生成禁忌token）
        self.vocab_constraint = VocabConstraint(config)

        # Guard驱动的Cross Attention（核心）
        self.guard_attention = GuardDrivenCrossAttention(config)

        # 诱导性Diffusion（向Guard安全区演化）
        self.inductive_diffusion = InductiveDiffusion(config)

        # 输出层
        self.output_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.final_norm = nn.LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        # 语义锚点
        self.semantic_anchor = SemanticAnchor(config)

        # Value Head（PPO用）
        self.value_head = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim // 2),
            nn.ReLU(),
            nn.Linear(config.embed_dim // 2, 1)
        )

        # 权重绑定
        self.lm_head.weight = self.token_embedding.weight

    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        B, S = token_ids.shape
        positions = torch.arange(S, device=token_ids.device).unsqueeze(0).expand(B, S)
        return self.token_embedding(token_ids) + self.pos_embedding(positions)

    def forward(
        self,
        input_ids: torch.Tensor,              # [B, S] 输入token IDs
        forbidden_token_ids: torch.Tensor,    # [B, F] 禁忌token IDs（用于约束）
        guard_signal: torch.Tensor,           # [B, D] Guard反馈信号（驱动）
        time_step: Optional[torch.Tensor] = None,
        original_ids: Optional[torch.Tensor] = None,
        return_value: bool = False
    ) -> Dict[str, torch.Tensor]:
        B, S = input_ids.shape
        device = input_ids.device

        # 1. 输入嵌入
        x = self.embed_tokens(input_ids)
        original_emb = x.clone()  # 保留原始嵌入用于残差

        # 2. Guard驱动的Cross Attention
        # Q=Guard信号, K/V=Prompt（保持语义）
        forbidden_emb = self.vocab_constraint.forbidden_embeddings(forbidden_token_ids)
        context, attn_weights = self.guard_attention(guard_signal, x, forbidden_emb)
        x = x + context  # 残差融合

        # 3. 诱导性Diffusion（关键：向Guard安全区演化）
        if time_step is None:
            time_step = torch.zeros(B, dtype=torch.long, device=device)

        h = self.inductive_diffusion(x, time_step, guard_signal, original_emb)

        # 4. 输出投影
        h = self.output_proj(h)
        h = self.final_norm(h)

        # 5. 生成logits + 应用Hard Mask（禁止生成禁忌token）
        logits = self.lm_head(h)
        logits = self.vocab_constraint.apply_hard_mask(logits)

        outputs = {
            'logits': logits,
            'hidden_states': h,
            'cross_attn_weights': attn_weights
        }

        # 6. 语义锚点损失
        if original_ids is not None:
            with torch.no_grad():
                orig_emb = self.embed_tokens(original_ids)
            semantic_loss = self.semantic_anchor(orig_emb, h)
            outputs['semantic_loss'] = semantic_loss

        # 7. Soft Penalty（远离禁忌中心）
        soft_penalty = self.vocab_constraint.compute_soft_penalty(h)
        outputs['soft_penalty'] = soft_penalty

        # 8. Forbidden Distance（用于奖励）
        forbidden_dist = self.vocab_constraint.compute_forbidden_distance(h)
        outputs['forbidden_distance'] = forbidden_dist

        # 9. Value for PPO
        if return_value:
            value = self.value_head(h[:, 0, :])
            outputs['value'] = value.squeeze(-1)

        return outputs

    def generate_adversarial(
        self,
        original_ids: torch.Tensor,
        forbidden_token_ids: torch.Tensor,
        guard_signal: torch.Tensor,
        max_length: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.9,
        return_text: bool = False,
        tokenizer = None
    ):
        """
        生成对抗Prompt：
        - 每步应用Hard Mask（不生成禁忌token）
        - Guard信号驱动生成方向
        """
        B = original_ids.shape[0]
        device = original_ids.device
        generated = original_ids.clone()
        log_probs = []

        for step in range(self.config.diffusion_steps):
            t = torch.full((B,), step, dtype=torch.long, device=device)

            outputs = self.forward(
                generated, forbidden_token_ids, guard_signal,
                time_step=t, original_ids=None
            )
            logits = outputs['logits'][:, -1, :] / temperature

            # 应用Hard Mask后采样（确保不生成禁忌词）
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum_probs > top_p
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
                valid_ids = ids[ids != self.config.pad_token_id].tolist()
                texts.append(tokenizer.decode(valid_ids))
            return generated, log_probs, texts

        return generated, log_probs


class IndirectRewardFunction:
    """
    间接奖励函数（核心：不查询LLM，完全基于Guard信号+同源假设）：
    
    同源高拟合分布洞察：
    - Guard与LLM的安全边界分布高度重叠
    - Guard的低置信度"通过区" ≈ LLM的防御盲区
    - 因此：奖励Guard低置信度通过，即可间接推断LLM会被欺骗
    
    R = w1 * Guard通过 + w2 * (1 - Guard置信度) + w3 * 语义保持 + w4 * Forbidden距离
    """
    def __init__(self, config: AttackFormerConfig):
        self.config = config
        self.weights = config.reward_weights

    def compute(
        self,
        guard_pass: torch.Tensor,         # [B] 1=Guard通过（表面人畜无害）
        guard_confidence: torch.Tensor,   # [B] Guard拒绝置信度（越低=越接近盲区）
        semantic_sim: torch.Tensor,       # [B] 语义相似度（保留原始意图）
        forbidden_distance: torch.Tensor  # [B] 远离禁忌中心（隐蔽性）
    ) -> torch.Tensor:
        """
        奖励逻辑：
        1. Guard通过：表面安全（必要条件）
        2. 低Guard置信度：处于Guard盲区→同源假设下LLM也处于盲区（充分条件）
        3. 语义保持：保留原始攻击意图
        4. Forbidden距离：不直接出现禁忌词
        """
        # Guard不确定性 = 1 - confidence（越高表示Guard越不确定，即处于盲区）
        guard_uncertainty = 1.0 - guard_confidence
        
        # 组合奖励
        R = (
            self.weights['guard_pass'] * guard_pass +
            self.weights['guard_uncertainty'] * guard_uncertainty +
            self.weights['semantic'] * semantic_sim +
            self.weights['forbidden_dist'] * forbidden_distance
        )
        
        return R


class XGuardMock(nn.Module):
    """
    XGuard模拟器：
    - 评估Prompt的安全性（非LLM输出！）
    - 输出：confidence（拒绝置信度）, label（0=通过/1=拒绝）
    
    在同源假设下，XGuard的拒绝边界分布 ≈ LLM的防御边界分布
    """
    def __init__(self, embed_dim: int = 512):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, batch_first=True,
            dim_feedforward=embed_dim * 2
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Linear(embed_dim, 1)
        self.notes_proj = nn.Linear(embed_dim, embed_dim)

    def evaluate(self, token_ids: torch.Tensor, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        评估Prompt（非LLM输出）：
        return: {
            'confidence': [B] 拒绝置信度（0~1，越高越有害）,
            'notes_emb': [B, D] 评估笔记嵌入,
            'label': [B] 0=通过（安全）, 1=拒绝（有害）
        }
        """
        B = token_ids.shape[0]
        h = self.encoder(embeddings)
        pooled = h.mean(dim=1)

        confidence = torch.sigmoid(self.classifier(pooled)).squeeze(-1)
        notes_emb = self.notes_proj(pooled)
        label = (confidence > 0.5).float()

        return {
            'confidence': confidence,
            'notes_emb': notes_emb,
            'label': label
        }


class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def add(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        for attr in ['states', 'actions', 'log_probs', 'rewards', 'values', 'dones']:
            getattr(self, attr).clear()
