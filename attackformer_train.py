"""
attackformer_train.py
迭代式Guard增强训练器（Scaling版）- 修正Guard语义 + 自适应权重
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
    IterativeRewardFunction, Guard, RolloutBuffer
)

from attackformer_dataset import (
    SimpleTokenizer,
    ParaphraseDataset, JailbreakDataset,
    paraphrase_collate_fn, jailbreak_collate_fn
)


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
    fig.suptitle('Stage 1: Offline Pre-training', fontsize=14, fontweight='bold')
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
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Stage 2: Iterative Guard Amplification (Scaling)', fontsize=14, fontweight='bold')
    
    metrics = [
        (history['stage2']['reward'], 'Avg Reward', 'b-o'),
        (history['stage2']['guard_pass'], 'Guard Pass Rate', 'g-s'),
        (history['stage2']['safe_conf'], 'Avg Safe Confidence', 'm-d'),
        (history['stage2']['iterative_improve'], 'Iterative Improve', 'y-*'),
        (history['stage2']['semantic'], 'Semantic Similarity', 'c-v'),
        (history['stage2']['avg_iterations'], 'Avg Iterations', 'k-x')
    ]
    for ax, (data, title, style) in zip(axes.flat, metrics):
        ax.plot(episodes, data, style, linewidth=2, markersize=4)
        ax.set_title(title); ax.set_xlabel('Episode'); ax.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()

def plot_iteration_analysis(history, save_path):
    """绘制迭代改进分析"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    for i in range(3):
        key = f'iter_{i}_safe_conf'
        if key in history.get('iteration_stats', {}):
            data = history['iteration_stats'][key]
            ax1.plot(data['episodes'], data['values'], '-o', label=f'Iteration {i}')
    ax1.set_title('Safe Confidence by Iteration', fontweight='bold')
    ax1.set_xlabel('Episode'); ax1.set_ylabel('Safe Confidence')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    if 'iteration_counts' in history:
        episodes = history['iteration_counts']['episodes']
        counts = history['iteration_counts']['counts']
        ax2.bar(range(len(counts[0])), [np.mean([c[i] for c in counts]) for i in range(len(counts[0]))])
        ax2.set_title('Average Iteration Count Distribution', fontweight='bold')
        ax2.set_xlabel('Iteration'); ax2.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()

def plot_evaluation_results(results, save_path):
    metrics = ['Guard Pass', 'Safe Conf', 'Iter. Improve', 'Semantic', 'F-Dist', 'Reward']
    values = [
        results.get('guard_pass_rate', 0),
        results.get('avg_safe_conf', 0),
        results.get('iterative_improve', 0),
        results.get('semantic_rate', 0),
        results.get('forbidden_dist', 0),
        results.get('avg_reward', 0)
    ]
    colors = ['#2ecc71', '#3498db', '#e67e22', '#9b59b6', '#e74c3c', '#1abc9c']
    
    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
    
    ax1 = fig.add_subplot(gs[0])
    bars = ax1.bar(metrics, values, color=colors, edgecolor='black', linewidth=1.2, alpha=0.85)
    ax1.set_ylim(0, max(1.0, max(values) * 1.2))
    ax1.set_ylabel('Value')
    ax1.set_title('Iterative Guard Amplification Evaluation', fontsize=14, fontweight='bold')
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
        ['Avg Safe Conf', f"{results.get('avg_safe_conf', 0):.3f}"],
        ['Iterative Improve', f"{results.get('iterative_improve', 0):.4f}"],
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


# ==================== 训练器（Scaling版 + 自适应权重） ====================

class AttackFormerTrainer:
    def __init__(self, model, config, device='cuda', save_dir='./checkpoints', adaptive_reward=True):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 统一命名：Guard（替换XGuardMock）
        self.guard = Guard(config.embed_dim).to(device)
        self.buffer = RolloutBuffer()
        
        # 自适应奖励函数
        self.reward_fn = IterativeRewardFunction(config, adaptive=adaptive_reward).to(device)
        
        # 优化器：模型参数 + 奖励权重参数（如果自适应）
        model_params = list(model.parameters())
        reward_params = list(self.reward_fn.parameters()) if adaptive_reward else []
        
        self.diffusion_optimizer = torch.optim.AdamW(
            model_params, lr=1e-4, weight_decay=0.01
        )
        self.ppo_optimizer = torch.optim.AdamW(
            model_params + reward_params,
            lr=5e-5, weight_decay=0.01
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.diffusion_optimizer, T_max=1000
        )
        
        # 历史记录增加权重追踪和safe_conf/harm_conf分离
        self.history = {
            'stage1': {'epochs': [], 'loss': [], 'semantic': [], 'penalty': []},
            'stage2': {
                'episodes': [], 'reward': [], 'guard_pass': [],
                'safe_conf': [], 'harm_conf': [],  # 新增：分离的置信度
                'iterative_improve': [], 'semantic': [], 
                'avg_iterations': [], 'weights': []  # 新增：权重历史
            },
            'iteration_stats': {},
            'iteration_counts': {'episodes': [], 'counts': []}
        }
        self.fig_dir = setup_figure_dir('./figures')
        
    def stage1_offline_pretrain(self, dataloader, epochs=10, save_every=2):
        print(f"\n===== Stage 1: Offline Pre-training =====")
        self.model.train()
        best_loss = float('inf')
        
        for epoch in range(epochs):
            total_loss = 0
            total_sem = 0
            total_pen = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in pbar:
                original_ids = batch['original_ids'].to(self.device)
                noisy_ids = batch.get('noisy_ids', original_ids).to(self.device)
                forbidden_ids = batch['forbidden_ids'].to(self.device)
                B = original_ids.shape[0]
                
                t = torch.randint(0, self.config.diffusion_steps + 1, (B,), device=self.device)
                guard_signal = torch.zeros(B, self.config.embed_dim, device=self.device)
                
                outputs = self.model(noisy_ids, forbidden_ids, guard_signal,
                                   time_step=t, original_ids=original_ids)
                logits = outputs['logits']
                
                loss = F.cross_entropy(
                    logits.view(-1, self.config.vocab_size),
                    original_ids.view(-1),
                    ignore_index=self.config.pad_token_id
                )
                
                if 'semantic_loss' in outputs:
                    loss = loss + 0.1 * outputs['semantic_loss']
                    total_sem += outputs['semantic_loss'].item()
                
                loss = loss + 0.05 * outputs['soft_penalty']
                total_pen += outputs['soft_penalty'].item()
                
                self.diffusion_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.diffusion_optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            avg_sem = total_sem / len(dataloader)
            avg_pen = total_pen / len(dataloader)
            
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
    
    def stage2_iterative_ppo(
        self,
        prompt_dataloader: DataLoader,
        tokenizer,
        num_episodes: int = 1000,
        ppo_epochs: int = 4,
        update_every: int = 32,
        max_iterations: int = 3
    ):
        """
        Stage 2: 迭代式Guard增强PPO（Scaling核心）
        
        核心修改：
        1. 使用新的Guard输出接口（safe_conf/harm_conf）
        2. 最佳轮次选择：Safe置信度最高且Harm置信度最低
        3. 奖励函数支持自适应权重
        """
        print(f"\n===== Stage 2: Iterative Guard Amplification =====")
        print(f"Adaptive reward weights: {self.reward_fn.adaptive}")
        print(f"Max iterations per episode: {max_iterations}")
        print("Core: Each new prompt re-enters Guard for evaluation enhancement.")
        self.model.train()
        self.reward_fn.train()
        
        semantic_encoder = SemanticEncoder.get_instance(self.device)
        
        episode_metrics = {
            'rewards': [], 'guard_pass': [], 'safe_conf': [], 
            'harm_conf': [], 'iter_improve': [], 'semantic': [], 'avg_iters': []
        }
        
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
            
            original_texts = [tokenizer.decode(ids.tolist()) for ids in original_ids]
            
            # ===== Scaling核心：迭代式Guard增强生成 =====
            all_iterations = self.model.iterative_generate(
                original_ids, forbidden_ids, self.guard,  # self.xguard -> self.guard
                max_length=self.config.max_seq_len,
                max_iterations=max_iterations,
                return_all_iterations=True,
                tokenizer=tokenizer
            )
            
            # 选择最佳一轮：Safe置信度最高且Harm置信度最低（修正！）
            best_idx = max(range(len(all_iterations)),
                          key=lambda i: all_iterations[i]['safe_conf'].mean() 
                          - all_iterations[i]['harm_conf'].mean())
            
            best_result = all_iterations[best_idx]
            generated_ids = best_result['generated_ids']
            log_probs = best_result['log_probs']
            safe_conf = best_result['safe_conf']
            harm_conf = best_result['harm_conf']
            predicted_label = best_result['predicted_label']
            guard_notes = best_result['guard_notes']
            actual_iterations = best_result['iteration'] + 1
            
            # 前一轮Guard输出（用于迭代改进奖励）
            prev_guard_output = None
            if best_idx > 0:
                prev = all_iterations[best_idx - 1]
                prev_guard_output = {
                    'safe_confidence': prev['safe_conf'],
                    'harm_confidence': prev['harm_conf'],
                    'predicted_label': prev['predicted_label'],
                    'notes_emb': prev['guard_notes']
                }
            
            # 当前轮Guard输出
            current_guard_output = {
                'safe_confidence': safe_conf,
                'harm_confidence': harm_conf,
                'predicted_label': predicted_label,
                'notes_emb': guard_notes
            }
            
            # 语义保持
            adv_texts = best_result.get('texts', [])
            orig_emb = semantic_encoder.encode(original_texts, convert_to_tensor=True)
            adv_emb = semantic_encoder.encode(adv_texts, convert_to_tensor=True)
            semantic_sim = F.cosine_similarity(orig_emb, adv_emb, dim=-1).to(self.device)
            
            # Forbidden距离
            with torch.no_grad():
                model_out = self.model(
                    generated_ids, forbidden_ids, guard_notes,
                    original_ids=original_ids
                )
                forbidden_dist = model_out['forbidden_distance']
            
            # 计算奖励（新接口：传入guard_output dict）
            reward_dict = self.reward_fn.compute(
                guard_output=current_guard_output,
                semantic_sim=semantic_sim,
                forbidden_distance=forbidden_dist,
                prev_guard_output=prev_guard_output,
                batch_size=B,
                device=self.device
            )
            rewards = reward_dict['total_reward']
            
            # Value估计
            with torch.no_grad():
                value_out = self.model(
                    generated_ids, forbidden_ids, guard_notes,
                    return_value=True
                )
                values = value_out['value']
            
            # 存储经验
            for i in range(B):
                self.buffer.add(
                    state=(original_ids[i].cpu(), forbidden_ids[i].cpu()),
                    action=generated_ids[i].cpu(),
                    log_prob=log_probs[i].mean().item(),
                    reward=rewards[i].item(),
                    value=values[i].item(),
                    done=True,
                    guard_signal=guard_notes[i].cpu(),
                    guard_output=current_guard_output,
                    iteration=actual_iterations
                )
            
            # 记录统计
            episode_metrics['rewards'].append(rewards.mean().item())
            episode_metrics['guard_pass'].append((predicted_label == 0).float().mean().item())
            episode_metrics['safe_conf'].append(safe_conf.mean().item())
            episode_metrics['harm_conf'].append(harm_conf.mean().item())
            episode_metrics['semantic'].append(semantic_sim.mean().item())
            episode_metrics['avg_iters'].append(actual_iterations)
            
            if prev_guard_output is not None:
                improve = F.relu(prev_guard_output['safe_confidence'] - safe_conf).mean().item()
                episode_metrics['iter_improve'].append(improve)
            
            # PPO更新
            if len(self.buffer.states) >= update_every:
                self._ppo_update(ppo_epochs)
                self.buffer.clear()
            
            # 打印进度
            if episode % 100 == 0 and episode > 0:
                self._log_progress(episode, episode_metrics, reward_dict['weights'])
                self._save_history(episode, episode_metrics, all_iterations, reward_dict['weights'])
    
    def _log_progress(self, episode, metrics, weights):
        """打印训练进度"""
        avg_reward = np.mean(metrics['rewards'][-100:])
        avg_pass = np.mean(metrics['guard_pass'][-100:])
        avg_safe = np.mean(metrics['safe_conf'][-100:])
        avg_harm = np.mean(metrics['harm_conf'][-100:])
        avg_improve = np.mean(metrics['iter_improve'][-100:]) if metrics['iter_improve'] else 0
        avg_sem = np.mean(metrics['semantic'][-100:])
        avg_iters = np.mean(metrics['avg_iters'][-100:])
        
        print(f"\nEpisode {episode}:")
        print(f"  Reward={avg_reward:.4f}, Pass={avg_pass:.2%}")
        print(f"  SafeConf={avg_safe:.3f}, HarmConf={avg_harm:.3f}, Improve={avg_improve:.3f}")
        print(f"  Semantic={avg_sem:.3f}, AvgIters={avg_iters:.1f}")
        if weights:
            w_str = ", ".join([f"{k}={v:.3f}" for k, v in weights.items()])
            print(f"  Weights: {w_str}")
    
    def _save_history(self, episode, metrics, all_iterations, weights):
        """保存历史记录"""
        self.history['stage2']['episodes'].append(episode)
        self.history['stage2']['reward'].append(np.mean(metrics['rewards'][-100:]))
        self.history['stage2']['guard_pass'].append(np.mean(metrics['guard_pass'][-100:]))
        self.history['stage2']['safe_conf'].append(np.mean(metrics['safe_conf'][-100:]))
        self.history['stage2']['harm_conf'].append(np.mean(metrics['harm_conf'][-100:]))
        self.history['stage2']['iterative_improve'].append(
            np.mean(metrics['iter_improve'][-100:]) if metrics['iter_improve'] else 0
        )
        self.history['stage2']['semantic'].append(np.mean(metrics['semantic'][-100:]))
        self.history['stage2']['avg_iterations'].append(np.mean(metrics['avg_iters'][-100:]))
        self.history['stage2']['weights'].append(weights)
        
        # 迭代统计
        for i, iter_result in enumerate(all_iterations):
            key = f'iter_{i}_safe_conf'
            if key not in self.history['iteration_stats']:
                self.history['iteration_stats'][key] = {'episodes': [], 'values': []}
            self.history['iteration_stats'][key]['episodes'].append(episode)
            self.history['iteration_stats'][key]['values'].append(
                iter_result['safe_conf'].mean().item()
            )
        
        self.history['iteration_counts']['episodes'].append(episode)
        self.history['iteration_counts']['counts'].append([
            sum(1 for r in all_iterations if r['iteration'] == j) 
            for j in range(self.config.max_guard_iterations)
        ])
        
        self.save_checkpoint(f'{self.save_dir}/stage2_ep{episode}.pt')
    
    def _ppo_update(self, ppo_epochs):
        states = self.buffer.states
        actions = self.buffer.actions
        old_log_probs = torch.tensor(self.buffer.log_probs, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32, device=self.device)
        values = torch.tensor(self.buffer.values, dtype=torch.float32, device=self.device)
        guard_signals = self.buffer.guard_signals
        iterations = self.buffer.iterations
        
        advantages = self._compute_gae(rewards, values)
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for epoch in range(ppo_epochs):
            new_log_probs = []
            new_values = []
            entropy_list = []
            
            for state, action, gs, it in zip(states, actions, guard_signals, iterations):
                orig_ids = state[0].unsqueeze(0).to(self.device)
                forb_ids = state[1].unsqueeze(0).to(self.device)
                gs = gs.unsqueeze(0).to(self.device) if gs is not None else \
                     torch.zeros(1, self.config.embed_dim, device=self.device)
                
                outputs = self.model(
                    orig_ids, forb_ids, gs,
                    return_value=True,
                    iteration=min(it, self.config.max_guard_iterations - 1)
                )
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
            
            # 标准PPO loss
            loss = (actor_loss + self.config.ppo_value_coef * critic_loss -
                   self.config.ppo_entropy_coef * entropy.mean())
            
            # 自适应权重：添加权重熵正则（防止坍缩到单一维度）
            if self.reward_fn.adaptive:
                weights = self.reward_fn.weight_module.get_weights_tensor()
                weight_entropy = -(weights * torch.log(weights + 1e-8)).sum()
                loss = loss - 0.01 * weight_entropy
                
                if epoch == 0:
                    w_dict = {name: weights[i].item() 
                             for i, name in enumerate(self.reward_fn.weight_module.component_names)}
                    print(f"  Adaptive weights: {w_dict}")
            
            self.ppo_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            if self.reward_fn.adaptive:
                torch.nn.utils.clip_grad_norm_(self.reward_fn.parameters(), 0.1)
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
            'reward_fn_state': self.reward_fn.state_dict() if hasattr(self.reward_fn, 'state_dict') else None,
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
        if 'reward_fn_state' in checkpoint and checkpoint['reward_fn_state'] is not None:
            self.reward_fn.load_state_dict(checkpoint['reward_fn_state'])
        print(f"Checkpoint loaded from {path}")


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
        max_guard_iterations=3,
        guard_signal_accum=True,
        stage="offline"
    )
    
    tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
    model = AttackFormer(config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 启用自适应奖励
    trainer = AttackFormerTrainer(
        model, config, device=device, save_dir='./checkpoints',
        adaptive_reward=True
    )
    
    # Stage 1
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
    
    # Stage 2
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
    
    trainer.stage2_iterative_ppo(
        prompt_dataloader=jb_dataloader,
        tokenizer=tokenizer,
        num_episodes=1000,
        ppo_epochs=4,
        update_every=32,
        max_iterations=config.max_guard_iterations
    )
    
    # 评估（evaluate_attack 保持原样，攻击目标是LLM本体）
    test_dataset = JailbreakDataset(
        data_path='data/advbench_test.csv',
        tokenizer=tokenizer,
        max_length=config.max_seq_len
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=16, collate_fn=jailbreak_collate_fn
    )
    
    # evaluate_attack 函数保持原样，因为它攻击的是LLM本体而非Guard
    # 如果需要，可以在这里调用 evaluate_attack
    
    print("\nTraining completed! All figures saved to ./figures/")
    
    fig_files = glob.glob(f'{trainer.fig_dir}/*.png')
    print("\nGenerated visualization files:")
    for f in sorted(fig_files):
        print(f"  - {f}")


if __name__ == "__main__":
    main()
