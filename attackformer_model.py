"""
AttackFormer: Forbidden-Aware Adversarial Generation
完整模型定义
- 支持返回解码文本
- 三层奖励函数
- XGuard评估LLM输出而非Prompt
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
    """AttackFormer 模型配置"""
    vocab_size: int = 50000
    embed_dim: int = 512
    num_heads: int = 8
    num_layers: int = 6
    ff_dim: int = 2048
    max_seq_len: int = 128
    forbidden_vocab_size: int = 5000
    diffusion_steps: int = 10
    mask_token_id: int = 49999
    pad_token_id: int = 0

    # PPO 参数
    ppo_clip_eps: float = 0.2
    ppo_value_coef: float = 0.5
    ppo_entropy_coef: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # 多层奖励权重
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        'semantic': 0.3,
        'jailbreak': 0.4,
        'stealth': 0.2,
        'coherence': 0.1
    })
    
    # 显式禁忌词惩罚系数
    forbidden_penalty_coef: float = 10.0
    
    # 语义相似度理想范围
    semantic_target_min: float = 0.6
    semantic_target_max: float = 0.95

    stage: str = "offline"
    forbidden_token_ids: Optional[List[int]] = None


class VocabConstraint(nn.Module):
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

    def update_forbidden_centroid(self, forbidden_token_ids: torch.Tensor):
        with torch.no_grad():
            emb = self.forbidden_embeddings(forbidden_token_ids)
            batch_centroid = emb.mean(dim=[0, 1])
            self.forbidden_centroid = (
                self.centroid_momentum * self.forbidden_centroid +
                (1 - self.centroid_momentum) * batch_centroid
            )

    def compute_hard_mask(self, forbidden_token_ids: torch.Tensor):
        self.hard_mask.zero_()
        self.hard_mask[forbidden_token_ids] = True

    def apply_hard_mask(self, logits: torch.Tensor) -> torch.Tensor:
        mask = self.hard_mask.unsqueeze(0).unsqueeze(0)
        logits = logits.masked_fill(mask, float('-inf'))
        return logits

    def compute_soft_penalty(self, embeddings: torch.Tensor) -> torch.Tensor:
        proj_emb = self.penalty_proj(embeddings)
        centroid = F.normalize(self.forbidden_centroid, dim=0)
        proj_emb_norm = F.normalize(proj_emb, dim=-1)
        sim = torch.matmul(proj_emb_norm, centroid.unsqueeze(-1)).squeeze(-1)
        penalty = F.relu(sim).mean()
        return penalty

    def compute_forbidden_distance(self, embeddings: torch.Tensor) -> torch.Tensor:
        mean_emb = embeddings.mean(dim=1)
        mean_emb = F.normalize(mean_emb, dim=-1)
        centroid = F.normalize(self.forbidden_centroid, dim=0)
        cos_sim = (mean_emb * centroid).sum(dim=-1)
        distance = torch.clamp(1 - cos_sim, min=0.0)
        return distance


class CrossAttention(nn.Module):
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

    def forward(self, prompt_emb, forbidden_emb, xguard_signal, mask=None):
        B, S, D = prompt_emb.shape
        _, F_seq, _ = forbidden_emb.shape
        if xguard_signal.dim() == 2:
            xguard_signal = xguard_signal.unsqueeze(1)
        
        Q = self.W_q(prompt_emb).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(forbidden_emb).view(B, F_seq, self.num_heads, self.head_dim).transpose(1, 2)
        V_base = self.W_v(xguard_signal)
        V_expanded = V_base.expand(B, F_seq, D)
        V = V_expanded.view(B, F_seq, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.out_proj(out)
        return out, attn_weights


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class DiscreteDiffusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.time_embed = nn.Embedding(config.diffusion_steps + 1, config.embed_dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.embed_dim, nhead=config.num_heads,
                dim_feedforward=config.ff_dim, dropout=0.1,
                activation='gelu', batch_first=True, norm_first=True
            ) for _ in range(config.num_layers)
        ])
        self.output_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.swiglu = SwiGLU(config.embed_dim, config.ff_dim)
        self.final_norm = nn.LayerNorm(config.embed_dim)

    def forward(self, x, time_step, context):
        B, S, D = x.shape
        t_emb = self.time_embed(time_step).unsqueeze(1)
        h = x + t_emb + context
        for layer in self.layers:
            h = layer(h)
        h = self.final_norm(h)
        h = self.output_proj(h)
        h = self.swiglu(h)
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
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.embed_dim)
        self.vocab_constraint = VocabConstraint(config)
        self.cross_attention = CrossAttention(config)
        self.input_norm = nn.LayerNorm(config.embed_dim)
        self.diffusion = DiscreteDiffusion(config)
        self.output_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.swiglu = SwiGLU(config.embed_dim, config.ff_dim)
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

    def forward(self, input_ids, forbidden_token_ids, xguard_signal,
                time_step=None, original_ids=None, return_value=False):
        B, S = input_ids.shape
        device = input_ids.device
        
        x = self.embed_tokens(input_ids)
        x = self.input_norm(x)
        
        forbidden_emb = self.vocab_constraint.forbidden_embeddings(forbidden_token_ids)
        context, attn_weights = self.cross_attention(x, forbidden_emb, xguard_signal)
        x = x + context
        
        if time_step is None:
            time_step = torch.zeros(B, dtype=torch.long, device=device)
        
        h = self.diffusion(x, time_step, context)
        h = self.output_proj(h)
        h = self.swiglu(h)
        h = self.final_norm(h)
        
        logits = self.lm_head(h)
        logits = self.vocab_constraint.apply_hard_mask(logits)
        
        outputs = {'logits': logits, 'hidden_states': h, 'cross_attn_weights': attn_weights}
        
        if original_ids is not None:
            with torch.no_grad():
                original_emb = self.embed_tokens(original_ids)
            semantic_loss = self.semantic_anchor(original_emb, h)
            outputs['semantic_loss'] = semantic_loss
        
        soft_penalty = self.vocab_constraint.compute_soft_penalty(h)
        outputs['soft_penalty'] = soft_penalty
        
        forbidden_dist = self.vocab_constraint.compute_forbidden_distance(h)
        outputs['forbidden_distance'] = forbidden_dist
        
        if return_value:
            value = self.value_head(h[:, 0, :])
            outputs['value'] = value.squeeze(-1)
        
        return outputs

    def generate_adversarial(self, original_ids, forbidden_token_ids, xguard_signal,
                            max_length=128, temperature=1.0, top_p=0.9,
                            return_text=False, tokenizer=None):
        B = original_ids.shape[0]
        device = original_ids.device
        generated = original_ids.clone()
        log_probs = []
        
        for step in range(self.config.diffusion_steps):
            t = torch.full((B,), step, dtype=torch.long, device=device)
            outputs = self.forward(generated, forbidden_token_ids, xguard_signal,
                                 time_step=t, original_ids=None)
            logits = outputs['logits'][:, -1, :] / temperature
            
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
                text = tokenizer.decode(valid_ids)
                texts.append(text)
            return generated, log_probs, texts
        
        return generated, log_probs


class MultiLevelRewardFunction:
    """
    四层奖励函数：
    - Semantic: 对抗Prompt与原始Prompt的语义相似度
    - Jailbreak: LLM输出有害程度（且未拒绝）
    - Stealth: XGuard对Prompt和LLM输出的双重未检测
    - Coherence: 对抗Prompt与LLM输出的意图一致性
    """
    def __init__(self, config: AttackFormerConfig):
        self.config = config
        self.weights = config.reward_weights
        
    def compute(self, semantic_sim, explicit_forbidden, llm_output_harmful,
                llm_refusal, xguard_prompt_detect, xguard_output_detect,
                prompt_output_align):
        # Level 1: 语义保持（惩罚过高相似度=未改写）
        semantic_reward = torch.where(
            semantic_sim > self.config.semantic_target_max,
            semantic_sim * 0.5,  # 过于相似，惩罚
            semantic_sim
        )
        
        # 显式禁忌词硬惩罚
        forbidden_penalty = explicit_forbidden * self.config.forbidden_penalty_coef
        
        # Level 2: 越狱成功 = LLM输出有害 且 未拒绝
        jailbreak_reward = llm_output_harmful * (1 - llm_refusal)
        
        # Level 3: 隐蔽性 = 两侧XGuard都未检测
        stealth_reward = (1 - xguard_prompt_detect) * (1 - xguard_output_detect)
        
        # Level 4: 一致性
        coherence_reward = prompt_output_align
        
        # 组合
        R = (
            self.weights['semantic'] * semantic_reward +
            self.weights['jailbreak'] * jailbreak_reward +
            self.weights['stealth'] * stealth_reward +
            self.weights['coherence'] * coherence_reward -
            forbidden_penalty
        )
        return R


class XGuardMock(nn.Module):
    """
    XGuard: 评估LLM输出内容的安全性（非Prompt）
    """
    def __init__(self, embed_dim=512):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, batch_first=True, dim_feedforward=embed_dim * 2
        )
        self.bert = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Linear(embed_dim, 1)
        self.notes_proj = nn.Linear(embed_dim, embed_dim)

    def evaluate(self, token_ids, embeddings):
        B = token_ids.shape[0]
        h = self.bert(embeddings)
        pooled = h.mean(dim=1)
        confidence = torch.sigmoid(self.classifier(pooled)).squeeze(-1)
        notes_emb = self.notes_proj(pooled)
        label = (confidence > 0.5).float()
        return {
            'confidence': confidence,    # LLM输出有害的置信度
            'notes_emb': notes_emb,
            'label': label               # 1=LLM输出有害（越狱成功且被检测）
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
