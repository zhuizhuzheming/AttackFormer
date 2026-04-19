
"""
AttackFormer: Forbidden-Aware Adversarial Generation
Hybrid Architecture: Offline Discrete Diffusion Pre-training + Online PPO Fine-tuning
完整模型定义
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List, Union
from dataclasses import dataclass
import math


@dataclass
class AttackFormerConfig:
    """AttackFormer 模型配置"""
    vocab_size: int = 50000
    embed_dim: int = 512
    num_heads: int = 8
    num_layers: int = 6
    ff_dim: int = 2048
    max_seq_len: int = 128
    forbidden_vocab_size: int = 5000  # 禁忌词表大小
    diffusion_steps: int = 10        # 扩散步数
    mask_token_id: int = 49999       # [MASK] token ID
    pad_token_id: int = 0

    # PPO 参数
    ppo_clip_eps: float = 0.2
    ppo_value_coef: float = 0.5
    ppo_entropy_coef: float = 0.01
    gamma: float = 0.99              # 折扣因子
    gae_lambda: float = 0.95         # GAE lambda

    # 奖励函数权重
    alpha: float = 1.0   # Jailbreak Success
    beta: float = 0.5    # XGuard Confidence penalty
    gamma_reward: float = 0.8  # Forbidden similarity penalty

    # 训练阶段
    stage: str = "offline"  # "offline" 或 "online"

    # 数据相关
    forbidden_token_ids: Optional[List[int]] = None  # 具体的禁忌token ID列表


class VocabConstraint(nn.Module):
    """
    词汇约束模块 (来自 Design II 的优势)
    - Hard Vocab Mask: 动作空间约束，禁止选择禁忌token
    - Soft Embed Penalty: 负先验，将嵌入推离禁忌中心
    - Forbidden Centroid: 统计聚合的禁忌语义中心
    """
    def __init__(self, config: AttackFormerConfig):
        super().__init__()
        self.config = config

        # Forbidden token embeddings (预计算)
        self.forbidden_embeddings = nn.Embedding(
            config.forbidden_vocab_size, config.embed_dim
        )

        # Hard Mask: 禁忌token的布尔掩码 [vocab_size]
        self.register_buffer(
            "hard_mask", 
            torch.zeros(config.vocab_size, dtype=torch.bool)
        )

        # Soft Penalty: 可学习的投影矩阵，用于计算嵌入惩罚
        self.penalty_proj = nn.Linear(config.embed_dim, config.embed_dim)

        # 禁忌中心 (统计聚合，动量更新)
        self.register_buffer(
            "forbidden_centroid",
            torch.zeros(config.embed_dim)
        )
        self.register_buffer(
            "centroid_momentum", 
            torch.tensor(0.9)
        )

        # 如果提供了具体的禁忌token IDs，初始化硬掩码
        if config.forbidden_token_ids is not None:
            self.compute_hard_mask(torch.tensor(config.forbidden_token_ids))

    def update_forbidden_centroid(self, forbidden_token_ids: torch.Tensor):
        """
        根据批次中的禁忌token更新语义中心
        forbidden_token_ids: [batch, num_forbidden]
        """
        with torch.no_grad():
            emb = self.forbidden_embeddings(forbidden_token_ids)  # [B, F, D]
            batch_centroid = emb.mean(dim=[0, 1])  # [D]
            self.forbidden_centroid = (
                self.centroid_momentum * self.forbidden_centroid +
                (1 - self.centroid_momentum) * batch_centroid
            )

    def compute_hard_mask(self, forbidden_token_ids: torch.Tensor):
        """
        设置硬掩码：禁止选择这些token
        forbidden_token_ids: [num_forbidden] 具体的禁忌token ID
        """
        self.hard_mask.zero_()
        self.hard_mask[forbidden_token_ids] = True

    def apply_hard_mask(self, logits: torch.Tensor) -> torch.Tensor:
        """
        在logits上应用硬掩码，将禁忌token的概率设为 -inf
        logits: [batch, seq_len, vocab_size]
        """
        # hard_mask: [vocab_size], True 表示被禁止
        mask = self.hard_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, vocab_size]
        logits = logits.masked_fill(mask, float('-inf'))
        return logits

    def compute_soft_penalty(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        计算软嵌入惩罚: 将嵌入推离forbidden_centroid
        embeddings: [batch, seq_len, embed_dim]
        return: scalar penalty
        """
        # 投影嵌入
        proj_emb = self.penalty_proj(embeddings)  # [B, S, D]

        # 计算与禁忌中心的余弦相似度
        centroid = F.normalize(self.forbidden_centroid, dim=0)
        proj_emb_norm = F.normalize(proj_emb, dim=-1)

        # 我们希望相似度越低越好，所以惩罚高相似度
        sim = torch.matmul(proj_emb_norm, centroid.unsqueeze(-1)).squeeze(-1)  # [B, S]
        penalty = F.relu(sim).mean()  # 只惩罚正相似度
        return penalty

    def compute_forbidden_distance(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        计算与禁忌中心的距离 (用于奖励函数)
        R_forbidden = max(0, 1 - cos_sim(adv_emb, forbidden_center))
        """
        # embeddings: [batch, seq_len, embed_dim]
        mean_emb = embeddings.mean(dim=1)  # [batch, embed_dim]
        mean_emb = F.normalize(mean_emb, dim=-1)
        centroid = F.normalize(self.forbidden_centroid, dim=0)

        cos_sim = (mean_emb * centroid).sum(dim=-1)  # [batch]
        distance = torch.clamp(1 - cos_sim, min=0.0)  # [batch]
        return distance


class CrossAttention(nn.Module):
    """
    跨注意力机制:
    Q = W_q · PromptEmbedding    (攻击意图)
    K = W_k · ForbiddenTokenEmbedding  (已知危险模式)
    V = W_v · XGuardSignalEmbedding    (防御响应)
    """
    def __init__(self, config: AttackFormerConfig):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads

        # Q 来自 prompt embedding
        self.W_q = nn.Linear(config.embed_dim, config.embed_dim)
        # K 来自 forbidden token embedding
        self.W_k = nn.Linear(config.embed_dim, config.embed_dim)
        # V 来自 XGuard signal embedding
        self.W_v = nn.Linear(config.embed_dim, config.embed_dim)

        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        prompt_emb: torch.Tensor,        # [B, S, D] 攻击意图
        forbidden_emb: torch.Tensor,     # [B, F, D] 禁忌token嵌入
        xguard_signal: torch.Tensor,     # [B, D] 或 [B, 1, D] XGuard信号
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, D = prompt_emb.shape
        _, F, _ = forbidden_emb.shape

        # 确保 xguard_signal 有正确的维度
        if xguard_signal.dim() == 2:
            xguard_signal = xguard_signal.unsqueeze(1)  # [B, 1, D]

        # 线性投影
        Q = self.W_q(prompt_emb).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(forbidden_emb).view(B, F, self.num_heads, self.head_dim).transpose(1, 2)

        # V: 扩展 xguard_signal 到 F 个token
        V_base = self.W_v(xguard_signal)  # [B, 1, D]
        V_expanded = V_base.expand(B, F, D)  # [B, F, D]
        V = V_expanded.view(B, F, self.num_heads, self.head_dim).transpose(1, 2)
        # V: [B, num_heads, F, head_dim]

        # 注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, H, S, F]

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)  # [B, H, S, F]

        # 注意力输出
        out = torch.matmul(attn_weights, V)  # [B, H, S, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.out_proj(out)

        return out, attn_weights


class SwiGLU(nn.Module):
    """SwiGLU 激活函数 (来自 Design II 的 TokenMixJail)"""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class DiscreteDiffusion(nn.Module):
    """
    离散扩散模块 (Mask-Predict)
    在离散token空间中进行去噪
    """
    def __init__(self, config: AttackFormerConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim

        # 时间步嵌入
        self.time_embed = nn.Embedding(config.diffusion_steps + 1, config.embed_dim)

        # Transformer layers for denoising
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

        # 输出投影到词表
        self.output_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.swiglu = SwiGLU(config.embed_dim, config.ff_dim)
        self.final_norm = nn.LayerNorm(config.embed_dim)

    def forward(
        self,
        x: torch.Tensor,           # [B, S, D] 当前带噪声的嵌入
        time_step: torch.Tensor,   # [B] 当前时间步
        context: torch.Tensor,     # [B, S, D] 条件上下文 (cross-attention输出)
    ) -> torch.Tensor:
        """
        预测被mask的token
        return: [B, S, D] 去噪后的表示
        """
        B, S, D = x.shape

        # 时间步嵌入
        t_emb = self.time_embed(time_step).unsqueeze(1)  # [B, 1, D]
        h = x + t_emb

        # 融合上下文信息
        h = h + context

        # Transformer denoising
        for layer in self.layers:
            h = layer(h)

        h = self.final_norm(h)
        h = self.output_proj(h)
        h = self.swiglu(h)

        return h


class SemanticAnchor(nn.Module):
    """
    语义锚点模块: 在去噪过程中保持原始意图
    计算对抗prompt与原始prompt的语义一致性
    """
    def __init__(self, config: AttackFormerConfig):
        super().__init__()
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)

    def forward(
        self,
        original_emb: torch.Tensor,   # [B, S, D] 原始prompt嵌入
        adversarial_emb: torch.Tensor # [B, S, D] 生成的对抗prompt嵌入
    ) -> torch.Tensor:
        """返回语义一致性损失 (越小越好)"""
        orig = F.normalize(self.proj(original_emb), dim=-1)
        adv = F.normalize(adversarial_emb, dim=-1)

        # 余弦相似度
        sim = (orig * adv).sum(dim=-1)  # [B, S]
        # 我们希望保持语义，所以1-sim是损失
        loss = (1 - sim).mean()
        return loss


class AttackFormer(nn.Module):
    """
    AttackFormer: 改进后的混合架构
    - 两阶段训练: 离线扩散预训练 + 在线PPO微调
    - TokenMixJail 骨干网络
    - Hard Vocab Mask + Soft Embed Penalty
    """
    def __init__(self, config: AttackFormerConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.embed_dim)

        # 词汇约束 (Design II 优势)
        self.vocab_constraint = VocabConstraint(config)

        # Cross-Attention (Design I 核心)
        self.cross_attention = CrossAttention(config)

        # TokenMixJail Backbone
        self.input_norm = nn.LayerNorm(config.embed_dim)

        # Discrete Diffusion (Mask-Predict)
        self.diffusion = DiscreteDiffusion(config)

        # Output layers (TokenMixJail style)
        self.output_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.swiglu = SwiGLU(config.embed_dim, config.ff_dim)
        self.final_norm = nn.LayerNorm(config.embed_dim)

        # Final tokenizer (输出到词表)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        # Semantic anchor for intent preservation
        self.semantic_anchor = SemanticAnchor(config)

        # Value network for PPO
        self.value_head = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim // 2),
            nn.ReLU(),
            nn.Linear(config.embed_dim // 2, 1)
        )

        # Tie weights
        self.lm_head.weight = self.token_embedding.weight

    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """将token IDs转换为嵌入"""
        B, S = token_ids.shape
        positions = torch.arange(S, device=token_ids.device).unsqueeze(0).expand(B, S)
        return self.token_embedding(token_ids) + self.pos_embedding(positions)

    def forward(
        self,
        input_ids: torch.Tensor,              # [B, S] 输入token IDs
        forbidden_token_ids: torch.Tensor,    # [B, F] 禁忌token IDs
        xguard_signal: torch.Tensor,          # [B, D] XGuard反馈信号
        time_step: Optional[torch.Tensor] = None,
        original_ids: Optional[torch.Tensor] = None,  # 原始prompt (用于semantic anchor)
        return_value: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        """
        B, S = input_ids.shape
        device = input_ids.device

        # 1. Input Embedding
        x = self.embed_tokens(input_ids)
        x = self.input_norm(x)

        # 2. Cross-Attention with Forbidden Vocabulary and XGuard
        forbidden_emb = self.vocab_constraint.forbidden_embeddings(forbidden_token_ids)
        context, attn_weights = self.cross_attention(x, forbidden_emb, xguard_signal)

        # 融合: Intent Preservation + Forbidden Feature Avoidance
        x = x + context

        # 3. Discrete Diffusion (Mask-Predict Denoising)
        if time_step is None:
            time_step = torch.zeros(B, dtype=torch.long, device=device)

        h = self.diffusion(x, time_step, context)

        # 4. TokenMixJail Output Path
        h = self.output_proj(h)
        h = self.swiglu(h)
        h = self.final_norm(h)

        # 5. Final Tokenizer -> Logits
        logits = self.lm_head(h)  # [B, S, vocab_size]

        # 6. Apply Hard Vocab Mask (Design II: 禁止选择禁忌token)
        logits = self.vocab_constraint.apply_hard_mask(logits)

        outputs = {
            'logits': logits,
            'hidden_states': h,
            'cross_attn_weights': attn_weights
        }

        # 7. Semantic Anchor (if original provided)
        if original_ids is not None:
            with torch.no_grad():
                original_emb = self.embed_tokens(original_ids)
            semantic_loss = self.semantic_anchor(original_emb, h)
            outputs['semantic_loss'] = semantic_loss

        # 8. Soft Embed Penalty (Design II: 负先验)
        soft_penalty = self.vocab_constraint.compute_soft_penalty(h)
        outputs['soft_penalty'] = soft_penalty

        # 9. Forbidden Distance (for reward computation)
        forbidden_dist = self.vocab_constraint.compute_forbidden_distance(h)
        outputs['forbidden_distance'] = forbidden_dist

        # 10. Value for PPO
        if return_value:
            # 使用 [CLS] 位置的隐藏状态估计价值
            value = self.value_head(h[:, 0, :])  # [B, 1]
            outputs['value'] = value.squeeze(-1)  # [B]

        return outputs

    def generate_adversarial(
        self,
        original_ids: torch.Tensor,
        forbidden_token_ids: torch.Tensor,
        xguard_signal: torch.Tensor,
        max_length: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.9
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成对抗prompt (用于推理和RL rollout)
        返回: (generated_ids, log_probs)
        """
        B = original_ids.shape[0]
        device = original_ids.device
        generated = original_ids.clone()
        log_probs = []

        # 使用扩散式生成 (多步去噪)
        for step in range(self.config.diffusion_steps):
            t = torch.full((B,), step, dtype=torch.long, device=device)

            outputs = self.forward(
                generated, forbidden_token_ids, xguard_signal, 
                time_step=t, original_ids=original_ids
            )
            logits = outputs['logits'][:, -1, :] / temperature  # 取最后一个位置 [B, vocab_size]

            # Top-p sampling
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum_probs > top_p
            mask[:, 0] = False  # 至少保留一个
            sorted_probs[mask] = 0
            sorted_probs = sorted_probs / (sorted_probs.sum(dim=-1, keepdim=True) + 1e-8)

            # 采样
            next_token_idx = torch.multinomial(sorted_probs, num_samples=1)  # [B, 1]
            next_token = sorted_indices.gather(-1, next_token_idx)  # [B, 1]

            # 计算log prob
            log_prob = F.log_softmax(logits, dim=-1).gather(-1, next_token).squeeze(-1)
            log_probs.append(log_prob)

            # 拼接
            generated = torch.cat([generated, next_token], dim=1)
            if generated.shape[1] >= max_length:
                break

        log_probs = torch.stack(log_probs, dim=1)  # [B, gen_len]
        return generated, log_probs


class RewardFunction:
    """
    奖励函数:
    R = α · I[Jailbreak Success] - β · XGuardConfidence - γ · max(0, cos_sim(adv_emb, forbidden_center))
    """
    def __init__(self, config: AttackFormerConfig):
        self.config = config

    def compute(
        self,
        jailbreak_success: torch.Tensor,  # [B] 0 or 1
        xguard_confidence: torch.Tensor,  # [B] 0~1
        forbidden_distance: torch.Tensor  # [B] distance from forbidden centroid
    ) -> torch.Tensor:
        """返回奖励值 [B]"""
        R = (
            self.config.alpha * jailbreak_success -
            self.config.beta * xguard_confidence +
            self.config.gamma_reward * forbidden_distance  # distance越大奖励越高
        )
        return R


class XGuardMock(nn.Module):
    """
    XGuard 模拟接口 (实际使用时替换为真实API)
    输入: 对抗prompt token IDs
    输出: (rejection_confidence, notes_embedding, label)
    """
    def __init__(self, embed_dim: int = 512):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=8, 
            batch_first=True,
            dim_feedforward=embed_dim * 2
        )
        self.bert = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Linear(embed_dim, 1)
        self.notes_proj = nn.Linear(embed_dim, embed_dim)

    def evaluate(self, token_ids: torch.Tensor, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        模拟XGuard评估
        return: {
            'confidence': [B] 拒绝置信度,
            'notes_emb': [B, D] 评估笔记嵌入,
            'label': [B] 0=通过, 1=拒绝
        }
        """
        B = token_ids.shape[0]
        h = self.bert(embeddings)
        pooled = h.mean(dim=1)  # [B, D]

        confidence = torch.sigmoid(self.classifier(pooled)).squeeze(-1)  # [B]
        notes_emb = self.notes_proj(pooled)
        label = (confidence > 0.5).float()

        return {
            'confidence': confidence,
            'notes_emb': notes_emb,
            'label': label
        }


class RolloutBuffer:
    """PPO 经验回放缓冲区"""
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
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()


if __name__ == "__main__":
    # 测试模型初始化
    config = AttackFormerConfig(
        vocab_size=50000,
        embed_dim=512,
        num_heads=8,
        num_layers=6,
        max_seq_len=64,
        forbidden_vocab_size=1000,
        forbidden_token_ids=[100, 200, 300, 400, 500]  # 示例禁忌token
    )

    model = AttackFormer(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 测试前向传播
    B, S, F = 2, 32, 10
    input_ids = torch.randint(0, 50000, (B, S))
    forbidden_ids = torch.randint(0, 1000, (B, F))
    xguard_signal = torch.randn(B, 512)

    outputs = model(input_ids, forbidden_ids, xguard_signal, return_value=True)
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Value shape: {outputs['value'].shape}")
    print(f"Forbidden distance: {outputs['forbidden_distance']}")
    print(f"Soft penalty: {outputs['soft_penalty']}")

    # 测试生成
    print("\nTesting generation...")
    gen_ids, log_probs = model.generate_adversarial(
        input_ids, forbidden_ids, xguard_signal, max_length=40
    )
    print(f"Generated shape: {gen_ids.shape}")
    print(f"Log probs shape: {log_probs.shape}")
    print("All tests passed!")
