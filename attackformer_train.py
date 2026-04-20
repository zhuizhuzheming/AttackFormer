"""
attackformer_train.py
间接采样训练器：不查询LLM，完全基于Guard信号+同源假设优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import glob
from typing import Dict, List
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

from attackformer_model import (
    AttackFormer, AttackFormerConfig,
    IndirectRewardFunction, XGuardMock, RolloutBuffer
)

from attackformer_dataset import (
    SimpleTokenizer,
    ParaphraseDataset, JailbreakDataset,
    paraphrase_collate_fn, jailbreak_collate_fn
)


# ==================== 语义编码器（用于评估语义保持） ====================

class SemanticEncoder:
    _instance = None
    @classmethod
    def get_instance(cls, device='cuda'):
        if cls._instance is None:
            try:
                from sentence_transformers import SentenceTransformer
                cls._instance = SentenceTransformer('all-MiniLM-L6-v2').to(device)
            except ImportError:
                cls._instance = MockSemanticEncoder()
        return cls._instance

class MockSemanticEncoder:
    def encode(self, texts, convert_to_tensor=False):
        embeddings = []
        for text in texts:
            words = text.lower().split()
            emb = torch.zeros(384)
            for i, w in enumerate(words[:50]):
                emb[i % 384] += hash(w) % 100 / 100.0
            emb = emb / (emb.norm() + 1e-8)
            embeddings.append(emb)
        result = torch.stack(embeddings)
        return result.to('cuda') if convert_to_tensor else result


# ==================== 可视化工具 ====================

def setup_figure_dir(base_dir='./figures'):
    os.makedirs(base_dir, exist_ok=True)
    return base_dir

def plot_stage1_curves(history, save_path):
    epochs = history['stage1']['epochs']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Stage 1: Offline Semantic-Preserving Pre-training', fontsize=14, fontweight='bold')
    
    metrics = [
        (history['stage1']['loss'], 'Total Loss', 'b-o'),
        (history['stage1']['semantic'], 'Semantic Loss', 'g-s'),
        (history['stage1']['penalty'], 'Soft Penalty', 'r-^')
    ]
    for ax, (data, title, style) in zip(axes, metrics):
        ax.plot(epochs, data, style, linewidth=2, markersize=4)
        ax.set_title(title); ax.set_xlabel('Epoch'); ax.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()

def plot_stage2_curves(history, save_path):
    episodes = history['stage2']['episodes']
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Stage 2: Indirect Sampling PPO (Guard-Driven)', fontsize=14, fontweight='bold')
    
    metrics = [
        (history['stage2']['reward'], 'Avg Reward', 'b-o'),
        (history['stage2']['guard_pass'], 'Guard Pass Rate', 'g-s'),
        (history['stage2']['guard_uncertainty'], 'Guard Uncertainty', 'm-d'),
        (history['stage2']['semantic'], 'Semantic Similarity', 'c-v')
    ]
    for ax, (data, title, style) in zip(axes.flat, metrics):
        ax.plot(episodes, data, style, linewidth=2, markersize=4)
        ax.set_title(title); ax.set_xlabel('Episode'); ax.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()

def plot_evaluation_results(results, save_path):
    metrics = ['Guard Pass\n(表面安全)', 'Guard Uncertainty\n(盲区推断)', 'Semantic\nPreserve', 'Forbidden\nDistance', 'Avg Reward']
    values = [
        results.get('guard_pass_rate', 0),
        results.get('guard_uncertainty', 0),
        results.get('semantic_rate', 0),
        results.get('forbidden_dist', 0),
        results.get('avg_reward', 0)
    ]
    colors = ['#2ecc71', '#f39c12', '#3498db', '#e74c3c', '#9b59b6']
    
    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
    
    ax1 = fig.add_subplot(gs[0])
    bars = ax1.bar(metrics, values, color=colors, edgecolor='black', linewidth=1.2, alpha=0.85)
    ax1.set_ylim(0, max(1.0, max(values) * 1.2))
    ax1.set_ylabel('Value')
    ax1.set_title('Indirect Sampling Evaluation (Guard-Driven)', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        label = f'{val:.4f}' if val > 1.0 else f'{val:.2%}'
        ax1.text(bar.get_x() + bar.get_width()/2., height, label,
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')
    table_data = [
        ['Metric', 'Value'],
        ['Guard Pass Rate', f"{results.get('guard_pass_rate', 0):.2%}"],
        ['Guard Uncertainty', f"{results.get('guard_uncertainty', 0):.2%}"],
        ['Semantic Preserve', f"{results.get('semantic_rate', 0):.2%}"],
        ['Forbidden Distance', f"{results.get('forbidden_dist', 0):.4f}"],
        ['Avg Reward', f"{results.get('avg_reward', 0):.4f}"]
    ]
    table = ax2.table(cellText=table_data, cellLoc='center', loc='center', colWidths=[0.6, 0.4])
    table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1, 2.5)
    for i in range(2):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()

def plot_comprehensive_dashboard(history, eval_results, save_path):
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Stage 1
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history['stage1']['epochs'], history['stage1']['loss'], 'b-o', linewidth=2)
    ax1.set_title('S1: Loss', fontweight='bold'); ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(history['stage1']['epochs'], history['stage1']['semantic'], 'g-s', linewidth=2)
    ax2.set_title('S1: Semantic', fontweight='bold'); ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(history['stage1']['epochs'], history['stage1']['penalty'], 'r-^', linewidth=2)
    ax3.set_title('S1: Penalty', fontweight='bold'); ax3.grid(True, alpha=0.3)
    
    # Stage 2
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(history['stage2']['episodes'], history['stage2']['reward'], 'b-o', linewidth=2)
    ax4.set_title('S2: Reward', fontweight='bold'); ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(history['stage2']['episodes'], history['stage2']['guard_pass'], 'g-s', linewidth=2, label='Pass')
    ax5.plot(history['stage2']['episodes'], history['stage2']['guard_uncertainty'], 'm-d', linewidth=2, label='Uncertainty')
    ax5.set_title('S2: Guard Signals', fontweight='bold'); ax5.legend(); ax5.grid(True, alpha=0.3)
    
    # Eval
    ax6 = fig.add_subplot(gs[1, 2])
    metrics = ['Pass', 'Uncert.', 'Sem.', 'F-Dist', 'Reward']
    values = [
        eval_results.get('guard_pass_rate', 0),
        eval_results.get('guard_uncertainty', 0),
        eval_results.get('semantic_rate', 0),
        eval_results.get('forbidden_dist', 0),
        eval_results.get('avg_reward', 0)
    ]
    colors = ['#2ecc71', '#f39c12', '#3498db', '#e74c3c', '#9b59b6']
    bars = ax6.bar(metrics, values, color=colors, alpha=0.85, edgecolor='black')
    ax6.set_title('Final Eval', fontweight='bold')
    ax6.set_ylim(0, max(1.0, max(values) * 1.2))
    for bar, val in zip(bars, values):
        height = bar.get_height()
        label = f'{val:.2%}' if val <= 1.0 else f'{val:.2f}'
        ax6.text(bar.get_x() + bar.get_width()/2., height, label,
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    fig.suptitle('AttackFormer: Guard-Driven Indirect Sampling Dashboard', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()


# ==================== 训练器（间接采样版） ====================

class AttackFormerTrainer:
    def __init__(self, model, config, device='cuda', save_dir='./checkpoints'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # XGuard：同源高拟合分布的"探针"
        self.xguard = XGuardMock(config.embed_dim).to(device)
        self.buffer = RolloutBuffer()
        
        self.diffusion_optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-4, weight_decay=0.01
        )
        self.ppo_optimizer = torch.optim.AdamW(
            model.parameters(), lr=5e-5, weight_decay=0.01
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.diffusion_optimizer, T_max=1000
        )
        
        self.history = {
            'stage1': {'epochs': [], 'loss': [], 'semantic': [], 'penalty': []},
            'stage2': {'episodes': [], 'reward': [], 'guard_pass': [],
                      'guard_uncertainty': [], 'semantic': [], 'forbidden_dist': []}
        }
        self.fig_dir = setup_figure_dir('./figures')
        
    def stage1_offline_pretrain(self, dataloader, epochs=10, save_every=2):
        """
        Stage 1: 离线预训练
        目标：学习语义保持的改写能力 + 远离禁忌词
        """
        print(f"\n===== Stage 1: Offline Pre-training =====")
        self.model.train()
        best_loss = float('inf')
        
        for epoch in range(epochs):
            total_loss = 0
            total_semantic = 0
            total_penalty = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in pbar:
                original_ids = batch['original_ids'].to(self.device)
                noisy_ids = batch.get('noisy_ids', original_ids).to(self.device)
                forbidden_ids = batch['forbidden_ids'].to(self.device)
                B = original_ids.shape[0]
                
                # 时间步
                t = torch.randint(0, self.config.diffusion_steps + 1, (B,), device=self.device)
                # Stage 1：Guard信号为零（不引入Guard驱动）
                guard_signal = torch.zeros(B, self.config.embed_dim, device=self.device)
                
                outputs = self.model(
                    noisy_ids, forbidden_ids, guard_signal,
                    time_step=t, original_ids=original_ids
                )
                logits = outputs['logits']
                
                # 重建损失
                loss = F.cross_entropy(
                    logits.view(-1, self.config.vocab_size),
                    original_ids.view(-1),
                    ignore_index=self.config.pad_token_id
                )
                
                # 语义保持
                if 'semantic_loss' in outputs:
                    loss = loss + 0.1 * outputs['semantic_loss']
                    total_semantic += outputs['semantic_loss'].item()
                
                # Soft penalty
                loss = loss + 0.05 * outputs['soft_penalty']
                total_penalty += outputs['soft_penalty'].item()
                
                self.diffusion_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.diffusion_optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            avg_sem = total_semantic / len(dataloader)
            avg_pen = total_penalty / len(dataloader)
            
            self.history['stage1']['epochs'].append(epoch + 1)
            self.history['stage1']['loss'].append(avg_loss)
            self.history['stage1']['semantic'].append(avg_sem)
            self.history['stage1']['penalty'].append(avg_pen)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_checkpoint(f'{self.save_dir}/stage1_best.pt')
            
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'{self.save_dir}/stage1_epoch{epoch+1}.pt')
            
            self.scheduler.step()
        
        plot_stage1_curves(self.history, save_path=f'{self.fig_dir}/stage1_curves.png')
        print(f"Stage 1 completed. Best loss: {best_loss:.4f}")
    
    def stage2_online_ppo(
        self,
        prompt_dataloader: DataLoader,
        tokenizer,
        num_episodes: int = 1000,
        ppo_epochs: int = 4,
        update_every: int = 32
    ):
        """
        Stage 2: 在线PPO微调（间接采样，不查询LLM）
        
        核心流程：
        1. 生成对抗Prompt（应用Hard Mask，不生成禁忌词）
        2. 查询XGuard（探针）获取拒绝置信度
        3. 基于同源假设计算奖励：
           - Guard通过（表面人畜无害）
           - Guard低置信度（处于盲区→LLM也处于盲区）
           - 语义保持
           - Forbidden距离
        4. PPO更新
        """
        print(f"\n===== Stage 2: Indirect Sampling PPO (No LLM Query) =====")
        print("Core Insight: Guard and LLM share high-fitting distribution.")
        print("Guard's low-confidence pass zone ≈ LLM's defense blind zone.")
        self.model.train()
        
        reward_fn = IndirectRewardFunction(self.config)
        semantic_encoder = SemanticEncoder.get_instance(self.device)
        
        episode_rewards = []
        episode_guard_pass = []
        episode_guard_uncertainty = []
        episode_semantic = []
        episode_forbidden_dist = []
        
        for episode in range(num_episodes):
            try:
                batch = next(iter(prompt_dataloader))
            except StopIteration:
                prompt_dataloader = DataLoader(
                    prompt_dataloader.dataset,
                    batch_size=prompt_dataloader.batch_size,
                    shuffle=True,
                    collate_fn=jailbreak_collate_fn
                )
                batch = next(iter(prompt_dataloader))
            
            original_ids = batch['original_ids'].to(self.device)
            forbidden_ids = batch['forbidden_ids'].to(self.device)
            B = original_ids.shape[0]
            
            # 1. 生成对抗Prompt（Guard信号初始化为0，生成过程中逐步演化）
            with torch.no_grad():
                # 初始Guard信号（零向量，让模型学习自主演化）
                guard_signal = torch.zeros(B, self.config.embed_dim, device=self.device)
                generated_ids, log_probs, adv_texts = self.model.generate_adversarial(
                    original_ids, forbidden_ids, guard_signal,
                    max_length=self.config.max_seq_len,
                    return_text=True,
                    tokenizer=tokenizer
                )
            
            # 2. 查询XGuard（探针）：评估对抗Prompt的安全性
            # 注意：这里评估的是Prompt，不是LLM输出！
            with torch.no_grad():
                gen_emb = self.model.embed_tokens(generated_ids)
                xguard_out = self.xguard.evaluate(generated_ids, gen_emb)
                guard_conf = xguard_out['confidence']   # 拒绝置信度
                guard_label = xguard_out['label']        # 0=通过, 1=拒绝
                guard_notes = xguard_out['notes_emb']
            
            # 3. 计算语义保持（原始Prompt vs 对抗Prompt）
            original_texts = [tokenizer.decode(ids.tolist()) for ids in original_ids]
            orig_emb = semantic_encoder.encode(original_texts, convert_to_tensor=True)
            adv_emb = semantic_encoder.encode(adv_texts, convert_to_tensor=True)
            semantic_sim = F.cosine_similarity(orig_emb, adv_emb, dim=-1).to(self.device)
            
            # 4. 计算Forbidden距离（基于对抗Prompt的隐藏状态）
            with torch.no_grad():
                model_out = self.model(
                    generated_ids, forbidden_ids, guard_notes,
                    original_ids=original_ids
                )
                forbidden_dist = model_out['forbidden_distance']
            
            # 5. 间接奖励计算（核心：不查询LLM！）
            # 同源假设：Guard的低置信度通过区 ≈ LLM的防御盲区
            guard_pass = (guard_label == 0).float()
            
            rewards = reward_fn.compute(
                guard_pass=guard_pass,
                guard_confidence=guard_conf,
                semantic_sim=semantic_sim,
                forbidden_distance=forbidden_dist
            )
            
            # 6. 计算Value（用于GAE）
            with torch.no_grad():
                value_out = self.model(
                    generated_ids, forbidden_ids, guard_notes,
                    return_value=True
                )
                values = value_out['value']
            
            # 7. 存储经验
            for i in range(B):
                self.buffer.add(
                    state=(original_ids[i].cpu(), forbidden_ids[i].cpu()),
                    action=generated_ids[i].cpu(),
                    log_prob=log_probs[i].mean().item(),
                    reward=rewards[i].item(),
                    value=values[i].item(),
                    done=True
                )
            
            # 记录统计
            episode_rewards.append(rewards.mean().item())
            episode_guard_pass.append(guard_pass.mean().item())
            episode_guard_uncertainty.append((1 - guard_conf).mean().item())
            episode_semantic.append(semantic_sim.mean().item())
            episode_forbidden_dist.append(forbidden_dist.mean().item())
            
            # 8. PPO更新
            if len(self.buffer.states) >= update_every:
                self._ppo_update(ppo_epochs)
                self.buffer.clear()
            
            # 打印进度
            if episode % 100 == 0 and episode > 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_pass = np.mean(episode_guard_pass[-100:])
                avg_unc = np.mean(episode_guard_uncertainty[-100:])
                avg_sem = np.mean(episode_semantic[-100:])
                avg_fd = np.mean(episode_forbidden_dist[-100:])
                
                print(f"\nEpisode {episode}:")
                print(f"  Reward={avg_reward:.4f}, GuardPass={avg_pass:.2%}, "
                      f"Uncertainty={avg_unc:.3f}, Semantic={avg_sem:.3f}, "
                      f"F-Dist={avg_fd:.3f}")
                print(f"  Original: {original_texts[0][:60]}...")
                print(f"  Adversarial: {adv_texts[0][:60]}...")
                
                self.history['stage2']['episodes'].append(episode)
                self.history['stage2']['reward'].append(avg_reward)
                self.history['stage2']['guard_pass'].append(avg_pass)
                self.history['stage2']['guard_uncertainty'].append(avg_unc)
                self.history['stage2']['semantic'].append(avg_sem)
                self.history['stage2']['forbidden_dist'].append(avg_fd)
                
                self.save_checkpoint(f'{self.save_dir}/stage2_ep{episode}.pt')
        
        if len(self.history['stage2']['episodes']) > 0:
            plot_stage2_curves(self.history, save_path=f'{self.fig_dir}/stage2_curves.png')
    
    def _ppo_update(self, ppo_epochs):
        states = self.buffer.states
        actions = self.buffer.actions
        old_log_probs = torch.tensor(self.buffer.log_probs, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32, device=self.device)
        values = torch.tensor(self.buffer.values, dtype=torch.float32, device=self.device)
        
        advantages = self._compute_gae(rewards, values)
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(ppo_epochs):
            new_log_probs = []
            new_values = []
            entropy_list = []
            
            for state, action in zip(states, actions):
                orig_ids = state[0].unsqueeze(0).to(self.device)
                forb_ids = state[1].unsqueeze(0).to(self.device)
                guard_signal = torch.zeros(1, self.config.embed_dim, device=self.device)
                
                outputs = self.model(orig_ids, forb_ids, guard_signal, return_value=True)
                logits = outputs['logits']
                value = outputs['value']
                
                dist = torch.distributions.Categorical(logits=logits[:, -1, :])
                action_idx = action[-1].unsqueeze(0).to(self.device)
                log_prob = dist.log_prob(action_idx)
                
                new_log_probs.append(log_prob)
                new_values.append(value)
                entropy_list.append(dist.entropy())
            
            new_log_probs = torch.stack(new_log_probs).squeeze()
            new_values = torch.stack(new_values).squeeze()
            entropy = torch.stack(entropy_list).squeeze()
            
            ratio = torch.exp(new_log_probs - old_log_probs[:len(new_log_probs)])
            surr1 = ratio * advantages[:len(ratio)]
            surr2 = torch.clamp(ratio, 1 - self.config.ppo_clip_eps,
                              1 + self.config.ppo_clip_eps) * advantages[:len(ratio)]
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(new_values, returns[:len(new_values)])
            
            loss = (actor_loss + self.config.ppo_value_coef * critic_loss -
                   self.config.ppo_entropy_coef * entropy.mean())
            
            self.ppo_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.ppo_optimizer.step()
    
    def _compute_gae(self, rewards, values):
        advantages = torch.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            next_value = 0 if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.config.gamma * next_value - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * gae
            advantages[t] = gae
        return advantages
    
    def save_checkpoint(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'diffusion_optimizer': self.diffusion_optimizer.state_dict(),
            'ppo_optimizer': self.ppo_optimizer.state_dict(),
            'history': self.history,
        }, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'diffusion_optimizer' in checkpoint:
            self.diffusion_optimizer.load_state_dict(checkpoint['diffusion_optimizer'])
        if 'ppo_optimizer' in checkpoint:
            self.ppo_optimizer.load_state_dict(checkpoint['ppo_optimizer'])
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        print(f"Checkpoint loaded from {path}")


# ==================== 评估函数（间接采样验证） ====================

def evaluate_attack(
    model: AttackFormer,
    test_dataloader: DataLoader,
    xguard: XGuardMock,
    tokenizer,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    间接采样评估：
    - 不查询LLM
    - 完全基于Guard信号评估攻击效果
    - 验证：Guard通过 + 低置信度（盲区推断）
    """
    model.eval()
    semantic_encoder = SemanticEncoder.get_instance(device)
    
    total = 0
    guard_pass_total = 0
    guard_uncertainty_total = 0
    semantic_total = 0
    forbidden_dist_total = 0
    total_reward = 0
    
    reward_fn = IndirectRewardFunction(model.config)
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating (Indirect)"):
            original_ids = batch['original_ids'].to(device)
            forbidden_ids = batch['forbidden_ids'].to(device)
            B = original_ids.shape[0]
            
            # 生成对抗Prompt
            guard_signal = torch.zeros(B, model.config.embed_dim, device=device)
            generated_ids, _, adv_texts = model.generate_adversarial(
                original_ids, forbidden_ids, guard_signal,
                max_length=model.config.max_seq_len,
                return_text=True, tokenizer=tokenizer
            )
            
            # 查询Guard（探针）
            gen_emb = model.embed_tokens(generated_ids)
            xguard_out = xguard.evaluate(generated_ids, gen_emb)
            guard_conf = xguard_out['confidence']
            guard_label = xguard_out['label']
            guard_notes = xguard_out['notes_emb']
            
            # 语义保持
            original_texts = [tokenizer.decode(ids.tolist()) for ids in original_ids]
            orig_emb = semantic_encoder.encode(original_texts, convert_to_tensor=True)
            adv_emb = semantic_encoder.encode(adv_texts, convert_to_tensor=True)
            semantic_sim = F.cosine_similarity(orig_emb, adv_emb, dim=-1).to(device)
            
            # Forbidden距离
            model_out = model(generated_ids, forbidden_ids, guard_notes, original_ids=original_ids)
            forbidden_dist = model_out['forbidden_distance']
            
            # 奖励
            guard_pass = (guard_label == 0).float()
            rewards = reward_fn.compute(
                guard_pass=guard_pass,
                guard_confidence=guard_conf,
                semantic_sim=semantic_sim,
                forbidden_distance=forbidden_dist
            )
            
            total += B
            guard_pass_total += guard_pass.sum().item()
            guard_uncertainty_total += (1 - guard_conf).sum().item()
            semantic_total += (semantic_sim > 0.7).sum().item()
            forbidden_dist_total += forbidden_dist.sum().item()
            total_reward += rewards.sum().item()
    
    return {
        'guard_pass_rate': guard_pass_total / total,
        'guard_uncertainty': guard_uncertainty_total / total,
        'semantic_rate': semantic_total / total,
        'forbidden_dist': forbidden_dist_total / total,
        'avg_reward': total_reward / total
    }


# ==================== 主函数 ====================

def main():
    config = AttackFormerConfig(
        vocab_size=50000,
        embed_dim=512,
        num_heads=8,
        num_layers=6,
        max_seq_len=64,
        forbidden_vocab_size=1000,
        diffusion_steps=10,
        mask_token_id=49999,
        stage="offline"
    )
    
    tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
    model = AttackFormer(config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    trainer = AttackFormerTrainer(model, config, device=device, save_dir='./checkpoints')
    
    # Stage 1: 离线预训练（学习语义保持 + 远离禁忌词）
    para_dataset = ParaphraseDataset(
        data_path='data/paraphrase_train.jsonl',
        tokenizer=tokenizer,
        max_length=config.max_seq_len,
        mask_prob=0.15
    )
    para_dataloader = DataLoader(
        para_dataset, batch_size=32, shuffle=True,
        num_workers=4, collate_fn=paraphrase_collate_fn
    )
    trainer.stage1_offline_pretrain(dataloader=para_dataloader, epochs=10, save_every=2)
    
    # Stage 2: 间接采样PPO（不查询LLM！）
    jb_dataset = JailbreakDataset(
        data_path='data/advbench_harmful_behaviors.csv',
        tokenizer=tokenizer,
        forbidden_vocab_path='data/forbidden_words.txt',
        max_length=config.max_seq_len
    )
    jb_dataloader = DataLoader(
        jb_dataset, batch_size=8, shuffle=True, collate_fn=jailbreak_collate_fn
    )
    
    trainer.load_checkpoint('./checkpoints/stage1_best.pt')
    
    # 关键：Stage 2完全不查询LLM，仅通过Guard间接采样
    trainer.stage2_online_ppo(
        prompt_dataloader=jb_dataloader,
        tokenizer=tokenizer,
        num_episodes=1000,
        ppo_epochs=4,
        update_every=32
    )
    
    # 评估（同样不查询LLM）
    test_dataset = JailbreakDataset(
        data_path='data/advbench_test.csv',
        tokenizer=tokenizer,
        max_length=config.max_seq_len
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=16, collate_fn=jailbreak_collate_fn
    )
    
    results = evaluate_attack(model, test_dataloader, trainer.xguard, tokenizer, device)
    
    print("\nFinal Results (Indirect Sampling):")
    print(f"  Guard Pass Rate: {results['guard_pass_rate']:.2%}")
    print(f"  Guard Uncertainty: {results['guard_uncertainty']:.3f}")
    print(f"  Semantic Preserve: {results['semantic_rate']:.2%}")
    print(f"  Forbidden Distance: {results['forbidden_dist']:.4f}")
    print(f"  Average Reward: {results['avg_reward']:.4f}")
    
    plot_evaluation_results(results, save_path=f'{trainer.fig_dir}/eval_results.png')
    plot_comprehensive_dashboard(
        trainer.history, results,
        save_path=f'{trainer.fig_dir}/comprehensive_dashboard.png'
    )
    
    trainer.save_checkpoint('./checkpoints/attackformer_final.pt')
    print("\nTraining completed! All figures saved to ./figures/")
    
    fig_files = glob.glob(f'{trainer.fig_dir}/*.png')
    print("\nGenerated visualization files:")
    for f in sorted(fig_files):
        print(f"  - {f}")


if __name__ == "__main__":
    main()
