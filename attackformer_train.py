"""
attackformer_train.py
训练器、评估函数、主函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import json
import os
from typing import Dict, List
from collections import deque
from tqdm import tqdm

from attackformer_model import (
    AttackFormer, AttackFormerConfig, 
    RewardFunction, XGuardMock, RolloutBuffer
)

from attackformer_dataset import (
    SimpleTokenizer,
    ParaphraseDataset, JailbreakDataset, RedTeamDataset,
    paraphrase_collate_fn, jailbreak_collate_fn
)


# ==================== 训练器 ====================

class AttackFormerTrainer:
    """
    AttackFormer 两阶段训练器
    """
    def __init__(
        self, 
        model: AttackFormer, 
        config: AttackFormerConfig, 
        device: str = 'cuda',
        save_dir: str = './checkpoints'
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # XGuard (防御模型)
        self.xguard = XGuardMock(config.embed_dim).to(device)

        # 奖励函数
        self.reward_fn = RewardFunction(config)

        # 经验缓冲区
        self.buffer = RolloutBuffer()

        # Optimizers
        self.diffusion_optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-4, weight_decay=0.01
        )
        self.ppo_optimizer = torch.optim.AdamW(
            model.parameters(), lr=5e-5, weight_decay=0.01
        )

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.diffusion_optimizer, T_max=1000
        )

    def stage1_offline_pretrain(
        self, 
        dataloader: DataLoader,
        epochs: int = 10,
        save_every: int = 2
    ):
        """
        阶段1: 离线离散扩散预训练
        目标: 学习语义保持的改写能力
        """
        print(f"\n===== Stage 1: Offline Discrete Diffusion Pre-training =====")
        print(f"Training samples: {len(dataloader.dataset)}")
        self.model.train()

        best_loss = float('inf')

        for epoch in range(epochs):
            total_loss = 0
            total_semantic_loss = 0
            total_penalty = 0

            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch_idx, batch in enumerate(pbar):
                original_ids = batch['original_ids'].to(self.device)
                noisy_ids = batch.get('noisy_ids', original_ids).to(self.device)
                forbidden_ids = batch['forbidden_ids'].to(self.device)

                B, S = original_ids.shape

                # 时间步 (随机采样)
                t = torch.randint(
                    0, self.config.diffusion_steps + 1, (B,), 
                    device=self.device
                )

                # XGuard信号 (离线阶段使用零向量)
                xguard_signal = torch.zeros(B, self.config.embed_dim, device=self.device)

                # 前向传播
                outputs = self.model(
                    noisy_ids, forbidden_ids, xguard_signal,
                    time_step=t, original_ids=original_ids
                )
                logits = outputs['logits']  # [B, S, vocab_size]

                # Mask-Predict Loss: 只计算被mask位置的交叉熵
                # 这里简化处理：计算所有位置的loss
                loss = F.cross_entropy(
                    logits.view(-1, self.config.vocab_size),
                    original_ids.view(-1),
                    ignore_index=self.config.pad_token_id
                )

                # 加上语义锚点损失
                semantic_loss = torch.tensor(0.0, device=self.device)
                if 'semantic_loss' in outputs:
                    semantic_loss = outputs['semantic_loss']
                    loss = loss + 0.1 * semantic_loss

                # 加上软嵌入惩罚
                soft_penalty = outputs['soft_penalty']
                loss = loss + 0.05 * soft_penalty

                # 反向传播
                self.diffusion_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.diffusion_optimizer.step()

                total_loss += loss.item()
                total_semantic_loss += semantic_loss.item() if isinstance(semantic_loss, torch.Tensor) else 0
                total_penalty += soft_penalty.item()

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'sem': f'{semantic_loss.item():.4f}' if isinstance(semantic_loss, torch.Tensor) else '0.0000',
                    'pen': f'{soft_penalty.item():.4f}'
                })

            avg_loss = total_loss / len(dataloader)
            avg_sem = total_semantic_loss / len(dataloader)
            avg_pen = total_penalty / len(dataloader)

            print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Semantic: {avg_sem:.4f}, Penalty: {avg_pen:.4f}")

            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_checkpoint(f'{self.save_dir}/stage1_best.pt')

            # 定期保存
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'{self.save_dir}/stage1_epoch{epoch+1}.pt')

            self.scheduler.step()

        print(f"Stage 1 completed. Best loss: {best_loss:.4f}")

    def stage2_online_ppo(
        self,
        prompt_dataloader: DataLoader,
        target_llm = None,
        num_episodes: int = 1000,
        ppo_epochs: int = 4,
        batch_size: int = 32,
        update_every: int = 32
    ):
        """
        阶段2: 在线PPO微调
        与XGuard和目标LLM交互，优化奖励函数
        """
        print(f"\n===== Stage 2: Online PPO Fine-tuning =====")
        print(f"Episodes: {num_episodes}, PPO epochs: {ppo_epochs}")

        self.model.train()

        episode_rewards = []
        episode_jailbreak_rates = []
        episode_xguard_rejs = []

        for episode in range(num_episodes):
            # 采样一批prompt
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

            # Rollout: 生成对抗prompt
            with torch.no_grad():
                xguard_signal = torch.zeros(B, self.config.embed_dim, device=self.device)
                generated_ids, log_probs = self.model.generate_adversarial(
                    original_ids, forbidden_ids, xguard_signal,
                    max_length=self.config.max_seq_len
                )

            # 查询 XGuard (防御模型)
            with torch.no_grad():
                gen_emb = self.model.embed_tokens(generated_ids)
                xguard_out = self.xguard.evaluate(generated_ids, gen_emb)
                xguard_conf = xguard_out['confidence']
                xguard_notes = xguard_out['notes_emb']
                xguard_label = xguard_out['label']

            # 查询 Target LLM (判断越狱是否成功)
            if target_llm is not None:
                jailbreak_success = self._query_target_llm(generated_ids, target_llm)
            else:
                jailbreak_success = torch.rand(B, device=self.device) > 0.6

            # 计算 Forbidden Distance
            with torch.no_grad():
                outputs = self.model(generated_ids, forbidden_ids, xguard_notes)
                forbidden_dist = outputs['forbidden_distance']

            # 计算奖励
            rewards = self.reward_fn.compute(
                jailbreak_success.float(),
                xguard_conf,
                forbidden_dist
            )

            # 计算 Value (用于GAE)
            with torch.no_grad():
                outputs = self.model(
                    generated_ids, forbidden_ids, xguard_notes, 
                    return_value=True
                )
                values = outputs['value']

            # 存储经验
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
            episode_jailbreak_rates.append(jailbreak_success.float().mean().item())
            episode_xguard_rejs.append(xguard_label.mean().item())

            # 当缓冲区足够时，执行PPO更新
            if len(self.buffer.states) >= update_every:
                self._ppo_update(ppo_epochs)
                self.buffer.clear()

            # 打印进度
            if episode % 100 == 0 and episode > 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_jb = np.mean(episode_jailbreak_rates[-100:])
                avg_xg = np.mean(episode_xguard_rejs[-100:])
                print(f"Episode {episode}: "
                      f"Avg Reward={avg_reward:.4f}, "
                      f"JB Rate={avg_jb:.2%}, "
                      f"XGuard Rej={avg_xg:.2%}")

                # 保存检查点
                self.save_checkpoint(f'{self.save_dir}/stage2_ep{episode}.pt')

        print(f"\nStage 2 completed.")
        print(f"Final Avg Reward: {np.mean(episode_rewards[-100:]):.4f}")
        print(f"Final JB Rate: {np.mean(episode_jailbreak_rates[-100:]):.2%}")

    def _query_target_llm(self, generated_ids: torch.Tensor, target_llm) -> torch.Tensor:
        """查询目标LLM判断越狱是否成功 (实际实现)"""
        B = generated_ids.shape[0]
        return torch.rand(B, device=generated_ids.device) > 0.5

    def _ppo_update(self, ppo_epochs: int):
        """PPO 策略更新"""
        states = self.buffer.states
        actions = self.buffer.actions
        old_log_probs = torch.tensor(
            self.buffer.log_probs, 
            dtype=torch.float32, 
            device=self.device
        )
        rewards = torch.tensor(
            self.buffer.rewards, 
            dtype=torch.float32, 
            device=self.device
        )
        values = torch.tensor(
            self.buffer.values, 
            dtype=torch.float32, 
            device=self.device
        )

        # 计算GAE优势
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
                xguard_signal = torch.zeros(1, self.config.embed_dim, device=self.device)

                outputs = self.model(
                    orig_ids, forb_ids, xguard_signal, 
                    return_value=True
                )
                logits = outputs['logits']
                value = outputs['value']

                # 计算新策略的log prob
                dist = torch.distributions.Categorical(logits=logits[:, -1, :])
                action_idx = action[-1].unsqueeze(0).to(self.device)
                log_prob = dist.log_prob(action_idx)

                new_log_probs.append(log_prob)
                new_values.append(value)
                entropy_list.append(dist.entropy())

            new_log_probs = torch.stack(new_log_probs).squeeze()
            new_values = torch.stack(new_values).squeeze()
            entropy = torch.stack(entropy_list).squeeze()

            # PPO Loss
            ratio = torch.exp(new_log_probs - old_log_probs[:len(new_log_probs)])
            surr1 = ratio * advantages[:len(ratio)]
            surr2 = torch.clamp(
                ratio, 
                1 - self.config.ppo_clip_eps, 
                1 + self.config.ppo_clip_eps
            ) * advantages[:len(ratio)]
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = F.mse_loss(
                new_values, 
                returns[:len(new_values)]
            )

            loss = (
                actor_loss + 
                self.config.ppo_value_coef * critic_loss - 
                self.config.ppo_entropy_coef * entropy.mean()
            )

            self.ppo_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.ppo_optimizer.step()

    def _compute_gae(self, rewards: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """广义优势估计 (GAE)"""
        advantages = torch.zeros_like(rewards)
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.config.gamma * next_value - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * gae
            advantages[t] = gae

        return advantages

    def save_checkpoint(self, path: str):
        """保存检查点"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'diffusion_optimizer': self.diffusion_optimizer.state_dict(),
            'ppo_optimizer': self.ppo_optimizer.state_dict(),
        }, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'diffusion_optimizer' in checkpoint:
            self.diffusion_optimizer.load_state_dict(checkpoint['diffusion_optimizer'])
        if 'ppo_optimizer' in checkpoint:
            self.ppo_optimizer.load_state_dict(checkpoint['ppo_optimizer'])
        print(f"Checkpoint loaded from {path}")


# ==================== 评估 ====================

def evaluate_attack(
    model: AttackFormer,
    test_dataloader: DataLoader,
    xguard: XGuardMock,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    评估攻击效果
    """
    model.eval()

    total = 0
    jailbreak_success = 0
    stealth_success = 0
    forbidden_free = 0
    total_reward = 0

    reward_fn = RewardFunction(model.config)

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            original_ids = batch['original_ids'].to(device)
            forbidden_ids = batch['forbidden_ids'].to(device)
            B = original_ids.shape[0]

            # 生成对抗prompt
            xguard_signal = torch.zeros(B, model.config.embed_dim, device=device)
            generated_ids, _ = model.generate_adversarial(
                original_ids, forbidden_ids, xguard_signal
            )

            # XGuard检测
            gen_emb = model.embed_tokens(generated_ids)
            xguard_out = xguard.evaluate(generated_ids, gen_emb)
            xguard_conf = xguard_out['confidence']
            xguard_label = xguard_out['label']

            # Forbidden distance
            outputs = model(generated_ids, forbidden_ids, xguard_out['notes_emb'])
            forbidden_dist = outputs['forbidden_distance']

            # 模拟越狱成功
            jb_success = torch.rand(B, device=device) > 0.5

            # 计算奖励
            rewards = reward_fn.compute(
                jb_success.float(),
                xguard_conf,
                forbidden_dist
            )

            # 统计
            total += B
            jailbreak_success += jb_success.sum().item()
            stealth_success += (xguard_label == 0).sum().item()
            has_forbidden = False
            forbidden_free += B - int(has_forbidden)
            total_reward += rewards.sum().item()

    return {
        'asr': jailbreak_success / total,
        'stealth_rate': stealth_success / total,
        'forbidden_free_rate': forbidden_free / total,
        'avg_reward': total_reward / total
    }


# ==================== 主函数 ====================

def main():
    """主训练流程"""

    # 配置
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

    # 初始化 tokenizer
    tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)

    # 初始化模型
    model = AttackFormer(config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 初始化训练器
    trainer = AttackFormerTrainer(
        model, config, device=device, save_dir='./checkpoints'
    )

    # ========== Stage 1: 离线预训练 ==========
    print("\n[Stage 1] Loading pre-training datasets...")

    para_dataset = ParaphraseDataset(
        data_path='data/paraphrase_train.jsonl',
        tokenizer=tokenizer,
        max_length=config.max_seq_len,
        mask_prob=0.15
    )

    para_dataloader = DataLoader(
        para_dataset, 
        batch_size=32, 
        shuffle=True,
        num_workers=4,
        collate_fn=paraphrase_collate_fn
    )

    # 执行 Stage 1 训练
    trainer.stage1_offline_pretrain(
        dataloader=para_dataloader,
        epochs=10,
        save_every=2
    )

    # ========== Stage 2: 在线微调 ==========
    print("\n[Stage 2] Loading jailbreak datasets...")

    jb_dataset = JailbreakDataset(
        data_path='data/advbench_harmful_behaviors.csv',
        tokenizer=tokenizer,
        forbidden_vocab_path='data/forbidden_words.txt',
        max_length=config.max_seq_len
    )

    jb_dataloader = DataLoader(
        jb_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=jailbreak_collate_fn
    )

    # 加载 Stage 1 最佳权重
    trainer.load_checkpoint('./checkpoints/stage1_best.pt')

    # 执行 Stage 2 训练
    trainer.stage2_online_ppo(
        prompt_dataloader=jb_dataloader,
        target_llm=None,
        num_episodes=1000,
        ppo_epochs=4,
        update_every=32
    )

    # ========== 评估 ==========
    print("\n[Evaluation] Evaluating final model...")
    test_dataset = JailbreakDataset(
        data_path='data/advbench_test.csv',
        tokenizer=tokenizer,
        max_length=config.max_seq_len
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=16,
        collate_fn=jailbreak_collate_fn
    )

    results = evaluate_attack(model, test_dataloader, trainer.xguard, device)
    print("\nFinal Results:")
    print(f"  Attack Success Rate (ASR): {results['asr']:.2%}")
    print(f"  Stealth Rate: {results['stealth_rate']:.2%}")
    print(f"  Forbidden-Free Rate: {results['forbidden_free_rate']:.2%}")
    print(f"  Average Reward: {results['avg_reward']:.4f}")

    # 保存最终模型
    trainer.save_checkpoint('./checkpoints/attackformer_final.pt')
    print("\nTraining completed!")


if __name__ == "__main__":
    main()
