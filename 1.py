import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import pickle
from typing import Dict, Tuple, List, Optional
import json

# -------------------------------
# 1. 환경(Environment) 설계
# -------------------------------

class EnvironmentA:
    """
    비정상성 환경 A:
      - 10개 상태, 5개 행동
      - 0~99단계, 100~199단계에 큰 보상 변화가 발생
      - 중간에 경미한 변화도 있음
    """
    def __init__(self, seed=1234):
        np.random.seed(seed)
        self.n_states = 10
        self.n_actions = 5
        self.current_state = 0
        self.current_step = 0
        self.max_steps = 200
        
        # 보상 구조는 단계별로 다름. 
        # 예: phase 1 (0~99), phase 2 (100~199)
        # 중간(50, 150)에서 경미한 변화
        self.rewards_phase1 = [0.8, 0.2, 0.2, 0.2, 0.2]  # action 0이 최적
        self.rewards_phase2 = [0.2, 0.2, 0.2, 0.2, 0.9]  # action 4가 최적
        
    def reset(self):
        self.current_state = 0
        self.current_step = 0
        return self.current_state
    
    def step(self, action: int) -> Tuple[int, float, bool]:
        """
        action을 취했을 때 다음 상태, 보상, 에피소드 종료 여부 반환
        """
        reward = self._get_reward(action)
        
        next_state = np.random.randint(0, self.n_states)
        self.current_state = next_state
        self.current_step += 1
        done = (self.current_step >= self.max_steps)
        
        return next_state, reward, done
    
    def _get_reward(self, action: int) -> float:
        if self.current_step < 100:
            base_reward = self.rewards_phase1[action]
            # 경미한 변화(시행 50 즈음) 예시
            if 50 <= self.current_step < 60 and action == 2:
                base_reward += 0.1
        else:
            base_reward = self.rewards_phase2[action]
            # 경미한 변화(시행 150 즈음)
            if 150 <= self.current_step < 160 and action == 1:
                base_reward += 0.2
        
        noise = np.random.normal(0, 0.1)
        return max(0.0, min(1.0, base_reward + noise))


class EnvironmentB:
    """
    연속적으로 reward가 변하는 환경 B:
      - 10개 상태, 5개 행동
      - 75~125 사이에서 action0 -> action4로 최적이 서서히 바뀜
    """
    def __init__(self, seed=1234):
        np.random.seed(seed)
        self.n_states = 10
        self.n_actions = 5
        self.current_state = 0
        self.current_step = 0
        self.max_steps = 200
        
    def reset(self):
        self.current_state = 0
        self.current_step = 0
        return self.current_state
    
    def step(self, action: int) -> Tuple[int, float, bool]:
        reward = self._get_reward(action)
        next_state = np.random.randint(0, self.n_states)
        
        self.current_state = next_state
        self.current_step += 1
        done = (self.current_step >= self.max_steps)
        return next_state, reward, done
    
    def _get_reward(self, action: int) -> float:
        # action0 -> action4로 서서히 옮겨가는 구간 (trial 75~125)
        # action0: 0.8->0.2, action4: 0.2->0.9
        if self.current_step < 75:
            base_action0 = 0.8
            base_action4 = 0.2
        elif self.current_step > 125:
            base_action0 = 0.2
            base_action4 = 0.9
        else:
            progress = (self.current_step - 75) / 50.0
            base_action0 = 0.8 - 0.6 * progress
            base_action4 = 0.2 + 0.7 * progress
        
        base_rewards = [0.3, 0.3, 0.3, 0.3, 0.3]
        base_rewards[0] = base_action0
        base_rewards[4] = base_action4
        
        noise = np.random.normal(0, 0.15)
        return max(0.0, min(1.0, base_rewards[action] + noise))


class EnvironmentC:
    """
    다중 컨텍스트 환경 C:
      - 10개 상태, 5개 행동
      - 컨텍스트 1(0~66), 컨텍스트 2(67~133), 컨텍스트 3(134~200)
    """
    def __init__(self, seed=1234):
        np.random.seed(seed)
        self.n_states = 10
        self.n_actions = 5
        self.current_state = 0
        self.current_step = 0
        self.max_steps = 200
        
    def reset(self):
        self.current_state = 0
        self.current_step = 0
        return self.current_state
    
    def step(self, action: int) -> Tuple[int, float, bool]:
        reward = self._get_reward(action)
        next_state = np.random.randint(0, self.n_states)
        
        self.current_state = next_state
        self.current_step += 1
        done = (self.current_step >= self.max_steps)
        return next_state, reward, done
    
    def _get_reward(self, action: int) -> float:
        if self.current_step <= 66:
            base = 0.8 if action == 0 else 0.2
        elif self.current_step <= 133:
            base = 0.8 if action == 2 else 0.2
        else:
            base = 0.8 if action == 4 else 0.2
        
        noise = np.random.normal(0, 0.1)
        return max(0.0, min(1.0, base + noise))

# --------------------------------------
# 2. 다양한 에이전트 구현
# --------------------------------------

class EpsilonGreedyAgent:
    """
    Epsilon-Greedy Q-Learning 에이전트
    """
    def __init__(self, n_states=10, n_actions=5, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((n_states, n_actions))
        self.action_history = []
    
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            action = np.argmax(self.Q[state])
        self.action_history.append(action)
        return action
    
    def update(self, state, action, reward, next_state, done):
        best_next = np.argmax(self.Q[next_state])
        td_target = reward + (0 if done else self.gamma * self.Q[next_state, best_next])
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
    
    def get_action_history(self):
        return np.array(self.action_history)


class UCBAgent:
    """
    Upper Confidence Bound (UCB) 알고리즘
    """
    def __init__(self, n_states=10, n_actions=5, alpha=0.1, gamma=0.9, c=0.5):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.c = c
        self.Q = np.zeros((n_states, n_actions))
        self.action_count = np.zeros((n_states, n_actions)) + 1e-9
        self.time = 1
        self.action_history = []
    
    def select_action(self, state):
        ucb_values = self.Q[state] + self.c * np.sqrt(np.log(self.time) / self.action_count[state])
        action = np.argmax(ucb_values)
        self.action_history.append(action)
        return action
    
    def update(self, state, action, reward, next_state, done):
        self.action_count[state, action] += 1
        best_next = np.argmax(self.Q[next_state])
        td_target = reward + (0 if done else self.gamma * self.Q[next_state, best_next])
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
        self.time += 1
    
    def get_action_history(self):
        return np.array(self.action_history)


class ThompsonSamplingAgent:
    """
    Thompson Sampling 알고리즘 (베이지안 접근)
    """
    def __init__(self, n_states=10, n_actions=5, gamma=0.9, alpha=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha
        self.mu = np.zeros((n_states, n_actions))
        self.sigma2 = np.ones((n_states, n_actions))
        self.action_history = []
    
    def select_action(self, state):
        samples = np.random.normal(self.mu[state], np.sqrt(self.sigma2[state]))
        action = np.argmax(samples)
        self.action_history.append(action)
        return action
    
    def update(self, state, action, reward, next_state, done):
        best_next = np.argmax(self.mu[next_state])
        td_target = reward + (0 if done else self.gamma * self.mu[next_state, best_next])
        td_error = td_target - self.mu[state, action]
        
        precision = 1.0 / self.sigma2[state, action]
        new_precision = precision + 1.0
        new_sigma2 = 1.0 / new_precision
        
        self.mu[state, action] = (precision * self.mu[state, action] + reward) * new_sigma2
        self.sigma2[state, action] = max(0.01, new_sigma2)
    
    def get_action_history(self):
        return np.array(self.action_history)


class ContextDetectingQL:
    """
    명시적 환경 변화 감지 CUSUM 기반 Q-Learning
    """
    def __init__(self, n_states=10, n_actions=5, alpha=0.1, gamma=0.9, 
                 alpha_high=0.3, window_size=20, threshold=3.0):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.alpha_high = alpha_high
        self.gamma = gamma
        self.Q = np.zeros((n_states, n_actions))
        
        self.window_size = window_size
        self.threshold = threshold
        self.recent_rewards = []
        self.detector_stat_pos = 0.0
        self.detector_stat_neg = 0.0
        self.baseline_reward = 0.5
        self.change_detected = False
        self.high_lr_count = 0
        
        self.action_history = []
        self.change_detection_points = []
    
    def select_action(self, state):
        if np.random.rand() < 0.1:
            action = np.random.randint(self.n_actions)
        else:
            action = np.argmax(self.Q[state])
        self.action_history.append(action)
        return action
    
    def update(self, state, action, reward, next_state, done):
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > self.window_size:
            self.recent_rewards.pop(0)
        
        if len(self.recent_rewards) >= self.window_size // 2:
            cur_mean = np.mean(self.recent_rewards[-5:])
            
            self.detector_stat_pos = max(0, self.detector_stat_pos + (cur_mean - self.baseline_reward - 0.05))
            self.detector_stat_neg = max(0, self.detector_stat_neg + (self.baseline_reward - cur_mean - 0.05))
            
            if (self.detector_stat_pos > self.threshold or 
                self.detector_stat_neg > self.threshold):
                self.change_detected = True
                self.change_detection_points.append(len(self.action_history) - 1)
                self.high_lr_count = 20
                self.baseline_reward = cur_mean
                self.detector_stat_pos = 0.0
                self.detector_stat_neg = 0.0
            
            if len(self.recent_rewards) % 10 == 0:
                self.baseline_reward = 0.9 * self.baseline_reward + 0.1 * cur_mean
        
        if self.high_lr_count > 0:
            cur_alpha = self.alpha_high
            self.high_lr_count -= 1
        else:
            cur_alpha = self.alpha
        
        best_next = np.argmax(self.Q[next_state])
        td_target = reward + (0 if done else self.gamma * self.Q[next_state, best_next])
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += cur_alpha * td_error
    
    def get_action_history(self):
        return np.array(self.action_history)
    
    def get_change_points(self):
        return np.array(self.change_detection_points)


class ECIAgent:
    """
    Emotion-Cognition Integration Architecture (ECIA):
    - 8차원 감정 벡터
    - 외부 변연계(ELS) + 전두엽 결정단위(PDU)
    """
    def __init__(self, n_states=10, n_actions=5, alpha=0.1, gamma=0.9,
                 eta=0.5, xi=0.1, emotion_dim=8, multi_core=True):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.eta = eta
        self.xi = xi
        self.emotion_dim = emotion_dim
        self.multi_core = multi_core
        
        self.Q = np.zeros((n_states, n_actions))
        
        # 감정-행동 매핑 (n_actions x emotion_dim)
        self.W = np.zeros((n_actions, emotion_dim))
        for a in range(n_actions):
            self.W[a, 0] = -0.5 + 0.25 * a / (n_actions - 1)
            self.W[a, 1] = 0.3
            self.W[a, 2] = 0.1 + 0.4 * a / (n_actions - 1)
            self.W[a, 3] = -0.4 + 0.2 * a / (n_actions - 1)
            self.W[a, 4] = 0.4
            self.W[a, 5] = -0.3 + 0.3 * abs(a - n_actions / 2) / (n_actions / 2)
            self.W[a, 6] = 0.2
            self.W[a, 7] = -0.2
        
        self.emotion_states = np.zeros((n_states, emotion_dim))
        self.emotion_states[:, 4] = 0.9
        
        self.reward_history = []
        self.td_errors = []
        self.last_actions = [-1] * 10
        
        if multi_core:
            self.n_cores = 3
            self.core_eta = np.array([0.4, 0.5, 0.6])
            self.core_xi = np.array([0.08, 0.1, 0.12])
            self.core_weights = np.ones(self.n_cores) / self.n_cores
            self.core_accuracy = np.ones(self.n_cores) * 0.5
        
        self.emotion_history = []
        self.action_history = []
    
    def select_action(self, state):
        E = self.emotion_states[state]
        
        if not self.multi_core:
            action_values = np.zeros(self.n_actions)
            for a in range(self.n_actions):
                emotion_modulation = np.dot(self.W[a], E)
                noise = np.random.normal(0, 1.0) * self.xi
                action_values[a] = self.Q[state, a] + self.eta * emotion_modulation + noise
            
            exp_vals = np.exp(action_values - np.max(action_values))
            probs = exp_vals / np.sum(exp_vals)
        else:
            core_probs = np.zeros((self.n_cores, self.n_actions))
            for c in range(self.n_cores):
                core_action_values = np.zeros(self.n_actions)
                for a in range(self.n_actions):
                    emotion_modulation = np.dot(self.W[a], E)
                    noise = np.random.normal(0, 1.0) * self.core_xi[c]
                    core_action_values[a] = (self.Q[state, a]
                                             + self.core_eta[c] * emotion_modulation
                                             + noise)
                exp_vals = np.exp(core_action_values - np.max(core_action_values))
                core_probs[c] = exp_vals / np.sum(exp_vals)
            
            probs = np.zeros(self.n_actions)
            for c in range(self.n_cores):
                probs += self.core_weights[c] * core_probs[c]
            probs = probs / np.sum(probs)
        
        action = np.random.choice(self.n_actions, p=probs)
        self.last_actions.pop(0)
        self.last_actions.append(action)
        self.action_history.append(action)
        
        return action
    
    def update(self, state, action, reward, next_state, done):
        best_next = np.argmax(self.Q[next_state])
        td_target = reward + (0 if done else self.gamma * self.Q[next_state, best_next])
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
        
        self.reward_history.append(reward)
        self.td_errors.append(td_error)
        
        self._update_emotions(state, action, reward, td_error, next_state)
        
        if self.multi_core and len(self.reward_history) > 1:
            self._update_core_weights()
        
        self.emotion_history.append(self.emotion_states[state].copy())
    
    def _update_emotions(self, state, action, reward, td_error, next_state):
        if len(self.reward_history) > 10:
            self.reward_history = self.reward_history[-10:]
            self.td_errors = self.td_errors[-10:]
        
        E = self.emotion_states[state].copy()
        
        # 두려움(0)
        if td_error < 0:
            E[0] = min(1.0, E[0] + 0.1 * abs(td_error))
        else:
            E[0] = max(0.0, E[0] - 0.05 * abs(td_error))
        
        # 기쁨(1)
        if td_error > 0:
            E[1] = min(1.0, E[1] + 0.15 * td_error)
        else:
            E[1] = max(0.0, E[1] - 0.05 * abs(td_error))
        
        # 희망(2)
        if len(self.reward_history) >= 3:
            recent_trend = np.mean(np.diff(self.reward_history[-3:]))
            if recent_trend > 0:
                E[2] = min(1.0, E[2] + 0.1 * recent_trend)
            else:
                E[2] = max(0.0, E[2] - 0.05 * abs(recent_trend))
        
        # 슬픔(3)
        if len(self.reward_history) >= 5:
            avg_reward = np.mean(self.reward_history[-5:])
            if avg_reward < 0.3:
                E[3] = min(1.0, E[3] + 0.1 * (0.3 - avg_reward))
            else:
                E[3] = max(0.0, E[3] - 0.05)
        
        # 호기심(4)
        state_visit_counts = np.zeros(self.n_states)
        for i, a in enumerate(self.last_actions):
            if a != -1:
                state_visit_counts[state] += 1
        
        novelty = np.exp(-0.5 * state_visit_counts[state])
        if state_visit_counts[state] <= 3:
            E[4] = min(1.0, E[4] + 0.1 * novelty)
        else:
            E[4] = max(0.3, E[4] - 0.03)
        
        if len(self.td_errors) >= 5:
            td_std = np.std(self.td_errors[-5:])
            E[4] = min(1.0, E[4] + 0.05 * td_std)
        
        # 분노(5)
        if td_error < -0.2:
            E[5] = min(1.0, E[5] + 0.2 * abs(td_error))
        else:
            E[5] = max(0.0, E[5] - 0.05)
        
        # 자부심(6)
        if len(self.reward_history) >= 5:
            success_rate = sum(r > 0.5 for r in self.reward_history[-5:]) / 5
            if success_rate > 0.6:
                E[6] = min(1.0, E[6] + 0.1 * success_rate)
            else:
                E[6] = max(0.0, E[6] - 0.05)
        
        # 수치심(7)
        if len(self.reward_history) >= 5:
            failure_rate = sum(r < 0.3 for r in self.reward_history[-5:]) / 5
            if failure_rate > 0.6:
                E[7] = min(1.0, E[7] + 0.1 * failure_rate)
            else:
                E[7] = max(0.0, E[7] - 0.05)
        
        # 감정 간 상호작용
        if E[0] > 0.7:
            E[4] = max(0.1, E[4] - 0.1)
        if E[1] > 0.7:
            E[3] = max(0.0, E[3] - 0.1)
            E[7] = max(0.0, E[7] - 0.1)
        if E[2] > 0.7:
            E[0] = max(0.0, E[0] - 0.1)
        
        self.emotion_states[state] = np.clip(E, 0.0, 1.0)
        
        # 행동-감정 가중치 W 점진적 학습
        if reward > 0.5:
            self.W[action] += 0.01 * self.emotion_states[state]
        elif reward < 0.3:
            self.W[action] -= 0.01 * self.emotion_states[state]
    
    def _update_core_weights(self):
        reward = self.reward_history[-1]
        for c in range(self.n_cores):
            prediction_error = abs(reward - (0.5 + 0.1 * (c - 1)))
            accuracy = max(0.1, 1.0 - prediction_error)
            self.core_accuracy[c] = 0.9 * self.core_accuracy[c] + 0.1 * accuracy
        
        temperature = 5.0
        exp_weights = np.exp(self.core_accuracy * temperature)
        self.core_weights = exp_weights / np.sum(exp_weights)
    
    def get_emotion_state(self, state):
        return self.emotion_states[state]
    
    def get_emotion_history(self):
        return np.array(self.emotion_history)
    
    def get_action_history(self):
        return np.array(self.action_history)


# ------------------------------------------
# 3. 실험 설계 및 평가 프레임워크
# ------------------------------------------

class ExperimentRunner:
    """
    다양한 환경과 에이전트에 대한 실험을 실행하고 결과를 분석하는 프레임워크
    """
    def __init__(self, results_dir="results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.env_classes = {
            "A": EnvironmentA,
            "B": EnvironmentB,
            "C": EnvironmentC
        }
        
        self.agent_classes = {
            "EGreedy": EpsilonGreedyAgent,
            "UCB": UCBAgent,
            "TS": ThompsonSamplingAgent,
            "CDQL": ContextDetectingQL,
            "ECIA": ECIAgent
        }
    
    def run_single_experiment(self, run_id, env_type, agent_type, max_steps=200, 
                              seeds=None, params=None):
        if seeds is None:
            env_seed = 1000 + run_id
            agent_seed = 2000 + run_id
        else:
            env_seed, agent_seed = seeds
        
        env_class = self.env_classes[env_type]
        env = env_class(seed=env_seed)
        
        agent_class = self.agent_classes[agent_type]
        if params is None:
            agent = agent_class()
        else:
            agent = agent_class(**params)
        
        rewards = []
        states = []
        actions = []
        
        if agent_type == "ECIA":
            emotion_states = []
        
        change_detection_points = None
        
        state = env.reset()
        states.append(state)
        
        for step in range(max_steps):
            action = agent.select_action(state)
            actions.append(action)
            
            next_state, reward, done = env.step(action)
            rewards.append(reward)
            
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            states.append(state)
            
            if agent_type == "ECIA":
                emotion_states.append(agent.get_emotion_state(state).copy())
            
            if done:
                break
        
        result = {
            "run_id": run_id,
            "env_type": env_type,
            "agent_type": agent_type,
            "rewards": np.array(rewards),
            "states": np.array(states),
            "actions": np.array(actions),
            "params": params
        }
        
        if agent_type == "ECIA":
            result["emotion_history"] = np.array(emotion_states)
        
        if agent_type == "CDQL":
            result["change_detection_points"] = agent.get_change_points()
        
        return result
    
    def run_batch_experiments(self, env_types=None, agent_types=None, num_runs=100, 
                              max_steps=200, params=None, n_jobs=8):
        if env_types is None:
            env_types = ["A", "B", "C"]
        
        if agent_types is None:
            agent_types = ["EGreedy", "UCB", "TS", "CDQL", "ECIA"]
        
        if params is None:
            params = {agent_type: {} for agent_type in agent_types}
        
        experiments = []
        for env_type in env_types:
            for agent_type in agent_types:
                for run_id in range(num_runs):
                    agent_params = params.get(agent_type, {})
                    seeds = (1000 + run_id, 2000 + run_id)
                    experiments.append((run_id, env_type, agent_type, max_steps, seeds, agent_params))
        
        results = []
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(self.run_single_experiment, *exp) for exp in experiments]
            for future in futures:
                results.append(future.result())
        
        structured_results = {}
        for env_type in env_types:
            structured_results[env_type] = {}
            for agent_type in agent_types:
                structured_results[env_type][agent_type] = [
                    r for r in results if r["env_type"] == env_type and r["agent_type"] == agent_type
                ]
        
        experiment_id = f"{self.timestamp}_runs{num_runs}_steps{max_steps}"
        results_path = os.path.join(self.results_dir, experiment_id)
        os.makedirs(results_path, exist_ok=True)
        
        with open(os.path.join(results_path, "experiment_config.json"), "w") as f:
            json.dump({
                "env_types": env_types,
                "agent_types": agent_types,
                "num_runs": num_runs,
                "max_steps": max_steps,
                "params": params
            }, f, indent=4)
        
        for env_type in env_types:
            for agent_type in agent_types:
                agent_results = structured_results[env_type][agent_type]
                save_path = os.path.join(results_path, f"{env_type}_{agent_type}_results.pkl")
                with open(save_path, "wb") as f:
                    pickle.dump(agent_results, f)
        
        print(f"실험 완료. 결과 저장 경로: {results_path}")
        return structured_results
    
    def analyze_results(self, results, env_type="A", save_plots=True, plot_dir=None):
        if plot_dir is None:
            plot_dir = os.path.join(self.results_dir, f"analysis_{self.timestamp}")
        
        os.makedirs(plot_dir, exist_ok=True)
        
        agent_types = list(results[env_type].keys())
        
        # 1. 평균 보상 분석
        self._analyze_mean_rewards(results, env_type, agent_types, save_plots, plot_dir)
        
        # 2. 행동 분포 분석
        self._analyze_action_distribution(results, env_type, agent_types, save_plots, plot_dir)
        
        # 3. 회복 시간 분석
        self._analyze_recovery_time(results, env_type, agent_types, save_plots, plot_dir)
        
        # 4. ECIA 감정 동태 분석 (ECIA에만 해당)
        if "ECIA" in agent_types:
            self._analyze_emotion_dynamics(results[env_type]["ECIA"], save_plots, plot_dir)
        
        # 5. 통계 분석
        stats_results = self._perform_statistical_analysis(results, env_type, agent_types)
        
        # 6. 결과 요약
        summary = self._create_result_summary(results, env_type, agent_types, stats_results)
        
        with open(os.path.join(plot_dir, f"{env_type}_summary.json"), "w") as f:
            json.dump(summary, f, indent=4)
        
        return summary
    
    def _analyze_mean_rewards(self, results, env_type, agent_types, save_plots, plot_dir):
        plt.figure(figsize=(12, 8))
        for agent_type in agent_types:
            agent_results = results[env_type][agent_type]
            all_rewards = np.array([r["rewards"] for r in agent_results])
            mean_rewards = np.mean(all_rewards, axis=0)
            std_rewards = np.std(all_rewards, axis=0) / np.sqrt(len(agent_results))
            
            x = np.arange(len(mean_rewards))
            plt.plot(x, mean_rewards, label=agent_type)
            plt.fill_between(x, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2)
        
        if env_type == "A":
            plt.axvline(x=100, color='r', linestyle='--', alpha=0.5, label="Major Change")
            plt.axvline(x=50, color='g', linestyle=':', alpha=0.5, label="Minor Change 1")
            plt.axvline(x=150, color='g', linestyle=':', alpha=0.5, label="Minor Change 2")
        elif env_type == "B":
            plt.axvspan(75, 125, color='r', alpha=0.1, label="Gradual Change")
        elif env_type == "C":
            plt.axvline(x=67, color='r', linestyle='--', alpha=0.5, label="Context 1->2")
            plt.axvline(x=134, color='r', linestyle='--', alpha=0.5, label="Context 2->3")
        
        plt.title(f"Mean Reward per Trial - Environment {env_type}")
        plt.xlabel("Trial")
        plt.ylabel("Mean Reward")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_plots:
            plt.savefig(os.path.join(plot_dir, f"{env_type}_mean_rewards.png"), dpi=300, bbox_inches="tight")
        plt.close()
    
    def _analyze_action_distribution(self, results, env_type, agent_types, save_plots, plot_dir):
        n_actions = 5
        
        if env_type == "A":
            change_point = 100
        elif env_type == "B":
            change_point = 100
        elif env_type == "C":
            change_points = [67, 134]
            for i, cp in enumerate(change_points):
                context_start = 0 if i == 0 else change_points[i-1]
                context_end = cp
                self._create_action_dist_plot(results, env_type, agent_types, n_actions, 
                                              context_start, context_end, f"Context_{i+1}", 
                                              save_plots, plot_dir)
            self._create_action_dist_plot(results, env_type, agent_types, n_actions, 
                                          change_points[-1], None, f"Context_{len(change_points)+1}", 
                                          save_plots, plot_dir)
            return
        
        self._create_action_dist_plot(results, env_type, agent_types, n_actions, 
                                      0, change_point, "Before_Change", save_plots, plot_dir)
        self._create_action_dist_plot(results, env_type, agent_types, n_actions, 
                                      change_point, None, "After_Change", save_plots, plot_dir)
    
    def _create_action_dist_plot(self, results, env_type, agent_types, n_actions, start_idx, end_idx, 
                                 phase_name, save_plots, plot_dir):
        plt.figure(figsize=(15, 10))
        bar_width = 0.15
        index = np.arange(n_actions)
        
        for i, agent_type in enumerate(agent_types):
            agent_results = results[env_type][agent_type]
            all_actions = []
            
            for r in agent_results:
                actions = r["actions"]
                phase_actions = actions[start_idx:end_idx]
                all_actions.extend(phase_actions)
            
            action_counts = np.zeros(n_actions)
            for a in all_actions:
                action_counts[a] += 1
            action_probs = action_counts / np.sum(action_counts)
            
            plt.bar(index + i * bar_width, action_probs, bar_width, 
                    label=agent_type, alpha=0.7)
        
        plt.title(f"Action Distribution - Environment {env_type} ({phase_name})")
        plt.xlabel("Action")
        plt.ylabel("Selection Probability")
        plt.xticks(index + bar_width*(len(agent_types)-1)/2, [f"Action {i}" for i in range(n_actions)])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_plots:
            plt.savefig(os.path.join(plot_dir, f"{env_type}_action_dist_{phase_name}.png"), 
                        dpi=300, bbox_inches="tight")
        plt.close()
    
    def _analyze_recovery_time(self, results, env_type, agent_types, save_plots, plot_dir):
        if env_type == "A":
            change_point = 100
            recovery_threshold = 0.9
            opt_reward = 0.9
        elif env_type == "B":
            change_point = 125
            recovery_threshold = 0.9
            opt_reward = 0.9
        elif env_type == "C":
            change_points = [67, 134]
            recovery_data = {}
            for i, cp in enumerate(change_points):
                ctx_recovery_data = {}
                for agent_type in agent_types:
                    ctx_recovery_data[agent_type] = self._calculate_context_recovery_times(
                        results[env_type][agent_type], cp, 0.8, 0.8
                    )
                recovery_data[f"Context_{i+1}_to_{i+2}"] = ctx_recovery_data
            
            plt.figure(figsize=(12, 8))
            width = 0.8 / len(agent_types)
            
            for ctx_idx, (ctx_name, ctx_data) in enumerate(recovery_data.items()):
                positions = np.arange(len(agent_types))
                for i, (agent_type, recovery_times) in enumerate(ctx_data.items()):
                    avg_recovery = np.mean(recovery_times)
                    std_recovery = np.std(recovery_times) / np.sqrt(len(recovery_times))
                    
                    plt.bar(positions[i] + ctx_idx*width, avg_recovery, width, 
                            yerr=std_recovery,
                            label=f"{agent_type} ({ctx_name})" if ctx_idx == 0 else "",
                            alpha=0.7)
            
            plt.title(f"Recovery Time After Context Switch - Environment {env_type}")
            plt.xlabel("Agent Type")
            plt.ylabel("Average Recovery Time (Trials)")
            plt.xticks(positions + width/2, agent_types)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if save_plots:
                plt.savefig(os.path.join(plot_dir, f"{env_type}_recovery_time.png"), 
                            dpi=300, bbox_inches="tight")
            plt.close()
            
            return recovery_data
        
        recovery_data = {}
        for agent_type in agent_types:
            agent_results = results[env_type][agent_type]
            recovery_times = []
            
            for r in agent_results:
                rewards = r["rewards"]
                post_change_rewards = rewards[change_point:]
                
                recovery_target = opt_reward * recovery_threshold
                recovery_indices = np.where(post_change_rewards >= recovery_target)[0]
                
                if len(recovery_indices) > 0:
                    recovery_time = recovery_indices[0]
                else:
                    recovery_time = len(post_change_rewards)
                
                recovery_times.append(recovery_time)
            
            recovery_data[agent_type] = recovery_times
        
        plt.figure(figsize=(10, 6))
        plt.boxplot([recovery_data[agent] for agent in agent_types], labels=agent_types)
        plt.title(f"Recovery Time After Environmental Change - Environment {env_type}")
        plt.xlabel("Agent Type")
        plt.ylabel("Recovery Time (Trials)")
        plt.grid(True, alpha=0.3)
        
        if save_plots:
            plt.savefig(os.path.join(plot_dir, f"{env_type}_recovery_time_boxplot.png"), 
                        dpi=300, bbox_inches="tight")
        plt.close()
        
        plt.figure(figsize=(10, 6))
        mean_recovery = [np.mean(recovery_data[agent]) for agent in agent_types]
        std_recovery = [np.std(recovery_data[agent]) / np.sqrt(len(recovery_data[agent])) for agent in agent_types]
        
        plt.bar(agent_types, mean_recovery, yerr=std_recovery, alpha=0.7)
        plt.title(f"Average Recovery Time - Environment {env_type}")
        plt.xlabel("Agent Type")
        plt.ylabel("Average Recovery Time (Trials)")
        plt.grid(True, alpha=0.3)
        
        if save_plots:
            plt.savefig(os.path.join(plot_dir, f"{env_type}_recovery_time_bar.png"), 
                        dpi=300, bbox_inches="tight")
        plt.close()
        
        return recovery_data
    
    def _calculate_context_recovery_times(self, agent_results, change_point, 
                                         recovery_threshold, opt_reward):
        recovery_times = []
        for r in agent_results:
            rewards = r["rewards"]
            post_change_rewards = rewards[change_point:change_point+67]
            
            recovery_target = opt_reward * recovery_threshold
            recovery_indices = np.where(post_change_rewards >= recovery_target)[0]
            
            if len(recovery_indices) > 0:
                recovery_time = recovery_indices[0]
            else:
                recovery_time = len(post_change_rewards)
            
            recovery_times.append(recovery_time)
        
        return recovery_times
    
    def _analyze_emotion_dynamics(self, ecia_results, save_plots, plot_dir):
        all_emotion_history = []
        for r in ecia_results:
            if "emotion_history" in r:
                all_emotion_history.append(r["emotion_history"])
        
        if not all_emotion_history:
            return
        
        avg_emotion_history = np.mean(all_emotion_history, axis=0)
        std_emotion_history = np.std(all_emotion_history, axis=0) / np.sqrt(len(all_emotion_history))
        
        emotion_names = ["두려움", "기쁨", "희망", "슬픔", "호기심", "분노", "자부심", "수치심"]
        
        plt.figure(figsize=(15, 10))
        x = np.arange(len(avg_emotion_history))
        
        for i, name in enumerate(emotion_names):
            plt.plot(x, avg_emotion_history[:, i], label=name)
            plt.fill_between(x,
                             avg_emotion_history[:, i] - std_emotion_history[:, i],
                             avg_emotion_history[:, i] + std_emotion_history[:, i],
                             alpha=0.2)
        
        plt.axvline(x=100, color='r', linestyle='--', alpha=0.5, label="Major Change")
        plt.axvline(x=50, color='g', linestyle=':', alpha=0.5, label="Minor Change 1")
        plt.axvline(x=150, color='g', linestyle=':', alpha=0.5, label="Minor Change 2")
        
        plt.title("Emotion Dynamics in ECIA")
        plt.xlabel("Trial")
        plt.ylabel("Emotion Intensity")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_plots:
            plt.savefig(os.path.join(plot_dir, "ECIA_emotion_dynamics.png"), dpi=300, bbox_inches="tight")
        plt.close()
        
        plt.figure(figsize=(20, 15))
        for i, name in enumerate(emotion_names):
            plt.subplot(3, 3, i+1)
            plt.plot(x, avg_emotion_history[:, i])
            plt.fill_between(x,
                             avg_emotion_history[:, i] - std_emotion_history[:, i],
                             avg_emotion_history[:, i] + std_emotion_history[:, i],
                             alpha=0.2)
            
            plt.axvline(x=100, color='r', linestyle='--', alpha=0.5)
            plt.title(f"{name} Dynamics")
            plt.xlabel("Trial")
            plt.ylabel("Intensity")
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(plot_dir, "ECIA_individual_emotions.png"), dpi=300, bbox_inches="tight")
        plt.close()
    
    def _perform_statistical_analysis(self, results, env_type, agent_types):
        stats_results = {}
        
        if env_type == "A":
            change_point = 100
            post_change_start = change_point
            post_change_end = None
        elif env_type == "B":
            change_point = 125
            post_change_start = change_point
            post_change_end = None
        elif env_type == "C":
            contexts = [(0, 67), (67, 134), (134, None)]
            context_stats = {}
            
            for i, (start, end) in enumerate(contexts):
                context_name = f"Context_{i+1}"
                context_stats[context_name] = self._analyze_context_performance(
                    results, env_type, agent_types, start, end
                )
            return context_stats
        
        post_change_stats = {}
        reference_agent = "ECIA"
        for agent_type in agent_types:
            if agent_type == reference_agent:
                continue
            
            ref_agent_rewards = []
            for r in results[env_type][reference_agent]:
                rewards = r["rewards"][post_change_start:post_change_end]
                avg_reward = np.mean(rewards)
                ref_agent_rewards.append(avg_reward)
            
            comp_agent_rewards = []
            for r in results[env_type][agent_type]:
                rewards = r["rewards"][post_change_start:post_change_end]
                avg_reward = np.mean(rewards)
                comp_agent_rewards.append(avg_reward)
            
            t_stat, p_value = stats.ttest_ind(ref_agent_rewards, comp_agent_rewards)
            
            ref_mean = np.mean(ref_agent_rewards)
            ref_std = np.std(ref_agent_rewards)
            comp_mean = np.mean(comp_agent_rewards)
            comp_std = np.std(comp_agent_rewards)
            
            pooled_std = np.sqrt((ref_std**2 + comp_std**2) / 2)
            cohen_d = abs(ref_mean - comp_mean) / pooled_std
            
            post_change_stats[f"{reference_agent}_vs_{agent_type}"] = {
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "cohen_d": float(cohen_d),
                "sample_sizes": (len(ref_agent_rewards), len(comp_agent_rewards)),
                "means": (float(ref_mean), float(comp_mean)),
                "stds": (float(ref_std), float(comp_std))
            }
        
        recovery_points = [10, 50]
        recovery_stats = {}
        
        for point in recovery_points:
            point_stats = {}
            for agent_type in agent_types:
                agent_recovery_rates = []
                for r in results[env_type][agent_type]:
                    if change_point + point < len(r["rewards"]):
                        recovery_reward = r["rewards"][change_point + point]
                        optimal_reward = 0.9
                        recovery_rate = recovery_reward / optimal_reward
                        agent_recovery_rates.append(recovery_rate)
                
                point_stats[agent_type] = {
                    "mean_recovery_rate": float(np.mean(agent_recovery_rates)),
                    "std_recovery_rate": float(np.std(agent_recovery_rates)),
                    "sample_size": len(agent_recovery_rates)
                }
            
            ref_agent = "ECIA"
            for agent_type in agent_types:
                if agent_type == ref_agent:
                    continue
                ref_rates = [r["rewards"][change_point + point] / 0.9
                             for r in results[env_type][ref_agent]
                             if change_point + point < len(r["rewards"])]
                comp_rates = [r["rewards"][change_point + point] / 0.9
                              for r in results[env_type][agent_type]
                              if change_point + point < len(r["rewards"])]
                
                t_stat, p_value = stats.ttest_ind(ref_rates, comp_rates)
                
                ref_mean = np.mean(ref_rates)
                ref_std = np.std(ref_rates)
                comp_mean = np.mean(comp_rates)
                comp_std = np.std(comp_rates)
                pooled_std = np.sqrt((ref_std**2 + comp_std**2) / 2)
                cohen_d = abs(ref_mean - comp_mean) / pooled_std
                
                point_stats[f"{ref_agent}_vs_{agent_type}"] = {
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "cohen_d": float(cohen_d),
                    "means": (float(ref_mean), float(comp_mean))
                }
            
            recovery_stats[f"recovery_after_{point}"] = point_stats
        
        stats_results = {
            "post_change_performance": post_change_stats,
            "recovery_analysis": recovery_stats
        }
        
        return stats_results
    
    def _analyze_context_performance(self, results, env_type, agent_types, start_idx, end_idx):
        context_stats = {}
        
        for agent_type in agent_types:
            agent_performances = []
            for r in results[env_type][agent_type]:
                rewards = r["rewards"][start_idx:end_idx]
                avg_reward = np.mean(rewards)
                agent_performances.append(avg_reward)
            
            context_stats[agent_type] = {
                "mean_reward": float(np.mean(agent_performances)),
                "std_reward": float(np.std(agent_performances)),
                "sample_size": len(agent_performances)
            }
        
        ref_agent = "ECIA"
        for agent_type in agent_types:
            if agent_type == ref_agent:
                continue
            ref_perfs = [np.mean(r["rewards"][start_idx:end_idx]) for r in results[env_type][ref_agent]]
            comp_perfs = [np.mean(r["rewards"][start_idx:end_idx]) for r in results[env_type][agent_type]]
            
            t_stat, p_value = stats.ttest_ind(ref_perfs, comp_perfs)
            ref_mean = np.mean(ref_perfs)
            ref_std = np.std(ref_perfs)
            comp_mean = np.mean(comp_perfs)
            comp_std = np.std(comp_perfs)
            pooled_std = np.sqrt((ref_std**2 + comp_std**2) / 2)
            cohen_d = abs(ref_mean - comp_mean) / pooled_std
            
            context_stats[f"{ref_agent}_vs_{agent_type}"] = {
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "cohen_d": float(cohen_d),
                "means": (float(ref_mean), float(comp_mean))
            }
        
        return context_stats
    
    def _create_result_summary(self, results, env_type, agent_types, stats_results):
        summary = {
            "env_type": env_type,
            "agent_types": agent_types,
            "num_runs": len(results[env_type][agent_types[0]]),
            "stats_results": stats_results
        }
        
        if env_type in ["A", "B"]:
            change_point = 100 if env_type == "A" else 125
            post_change_rewards = {}
            for agent_type in agent_types:
                agent_post_rewards = []
                for r in results[env_type][agent_type]:
                    rewards = r["rewards"][change_point:]
                    avg_reward = np.mean(rewards)
                    agent_post_rewards.append(avg_reward)
                
                post_change_rewards[agent_type] = {
                    "mean": float(np.mean(agent_post_rewards)),
                    "std": float(np.std(agent_post_rewards)),
                    "sample_size": len(agent_post_rewards)
                }
            
            summary["post_change_rewards"] = post_change_rewards
            
            recovery_data = {}
            opt_reward = 0.9
            recovery_threshold = 0.9
            
            for agent_type in agent_types:
                agent_recovery_times = []
                for r in results[env_type][agent_type]:
                    rewards = r["rewards"]
                    post_change_rewards = rewards[change_point:]
                    
                    recovery_target = opt_reward * recovery_threshold
                    recovery_indices = np.where(post_change_rewards >= recovery_target)[0]
                    
                    if len(recovery_indices) > 0:
                        recovery_time = recovery_indices[0]
                    else:
                        recovery_time = len(post_change_rewards)
                    
                    agent_recovery_times.append(recovery_time)
                
                recovery_data[agent_type] = {
                    "mean": float(np.mean(agent_recovery_times)),
                    "std": float(np.std(agent_recovery_times)),
                    "median": float(np.median(agent_recovery_times)),
                    "sample_size": len(agent_recovery_times)
                }
            
            summary["recovery_times"] = recovery_data
        
        elif env_type == "C":
            context_bounds = [(0, 67), (67, 134), (134, 200)]
            context_performance = {}
            
            for i, (start, end) in enumerate(context_bounds):
                context_name = f"Context_{i+1}"
                agent_performances = {}
                
                for agent_type in agent_types:
                    agent_ctx_rewards = []
                    for r in results[env_type][agent_type]:
                        rewards = r["rewards"][start:end]
                        avg_reward = np.mean(rewards)
                        agent_ctx_rewards.append(avg_reward)
                    
                    agent_performances[agent_type] = {
                        "mean": float(np.mean(agent_ctx_rewards)),
                        "std": float(np.std(agent_ctx_rewards)),
                        "sample_size": len(agent_ctx_rewards)
                    }
                
                context_performance[context_name] = agent_performances
            
            summary["context_performance"] = context_performance
            
            transition_points = [67, 134]
            transition_efficiency = {}
            
            for i, tp in enumerate(transition_points):
                trans_name = f"Transition_{i+1}_to_{i+2}"
                agent_efficiency = {}
                
                for agent_type in agent_types:
                    switch_rewards_10 = []
                    for r in results[env_type][agent_type]:
                        if tp + 10 <= len(r["rewards"]):
                            switch_reward = np.mean(r["rewards"][tp:tp+10])
                            switch_rewards_10.append(switch_reward)
                    
                    agent_efficiency[agent_type] = {
                        "mean_reward_10_trials": float(np.mean(switch_rewards_10)),
                        "std_reward_10_trials": float(np.std(switch_rewards_10)),
                        "sample_size": len(switch_rewards_10)
                    }
                
                transition_efficiency[trans_name] = agent_efficiency
            
            summary["transition_efficiency"] = transition_efficiency
        
        if "ECIA" in agent_types:
            emotion_names = ["두려움", "기쁨", "희망", "슬픔", "호기심", "분노", "자부심", "수치심"]
            
            if env_type == "A":
                change_point = 100
                emotion_dynamics = {}
                
                for r in results[env_type]["ECIA"]:
                    if "emotion_history" in r:
                        emotion_history = r["emotion_history"]
                        
                        pre_change_emotions = emotion_history[90:100].mean(axis=0)
                        post_change_emotions = emotion_history[100:110].mean(axis=0)
                        adapted_emotions = emotion_history[150:160].mean(axis=0)
                        
                        for i, name in enumerate(emotion_names):
                            if name not in emotion_dynamics:
                                emotion_dynamics[name] = {
                                    "pre_change": [],
                                    "post_change": [],
                                    "adapted": []
                                }
                            
                            emotion_dynamics[name]["pre_change"].append(pre_change_emotions[i])
                            emotion_dynamics[name]["post_change"].append(post_change_emotions[i])
                            emotion_dynamics[name]["adapted"].append(adapted_emotions[i])
                
                emotion_summary = {}
                for name in emotion_names:
                    pre_mean = np.mean(emotion_dynamics[name]["pre_change"])
                    post_mean = np.mean(emotion_dynamics[name]["post_change"])
                    adapted_mean = np.mean(emotion_dynamics[name]["adapted"])
                    
                    emotion_summary[name] = {
                        "pre_change_mean": float(pre_mean),
                        "post_change_mean": float(post_mean),
                        "adapted_mean": float(adapted_mean),
                        "change_ratio": float(post_mean / pre_mean if pre_mean > 0 else 0)
                    }
                
                summary["emotion_dynamics"] = emotion_summary
        
        return summary


def run_comprehensive_experiments(num_runs=1000, max_steps=200, save_plots=True):
    exp_runner = ExperimentRunner(results_dir="results_comprehensive")
    env_types = ["A", "B", "C"]
    agent_types = ["EGreedy", "UCB", "TS", "CDQL", "ECIA"]
    params = {
        "EGreedy": {"alpha": 0.1, "gamma": 0.9, "epsilon": 0.1},
        "UCB": {"alpha": 0.1, "gamma": 0.9, "c": 0.5},
        "TS": {"alpha": 0.1, "gamma": 0.9},
        "CDQL": {"alpha": 0.1, "gamma": 0.9, "alpha_high": 0.3, "window_size": 20, "threshold": 3.0},
        "ECIA": {"alpha": 0.1, "gamma": 0.9, "eta": 0.5, "xi": 0.1, "emotion_dim": 8, "multi_core": True}
    }
    
    print(f"실행 중: {num_runs}회 실행, {max_steps}회 시행, 모든 환경 및 에이전트")
    results = exp_runner.run_batch_experiments(
        env_types=env_types,
        agent_types=agent_types,
        num_runs=num_runs,
        max_steps=max_steps,
        params=params,
        n_jobs=8
    )
    
    analysis_results = {}
    for env_type in env_types:
        print(f"환경 {env_type} 분석 중...")
        analysis_results[env_type] = exp_runner.analyze_results(
            results, env_type=env_type, save_plots=save_plots
        )
    
    print("종합 실험 및 분석 완료!")
    return results, analysis_results


def run_single_environment_experiment(env_type="A", num_runs=1000, max_steps=200, save_plots=True):
    exp_runner = ExperimentRunner(results_dir=f"results_{env_type}")
    agent_types = ["EGreedy", "UCB", "TS", "CDQL", "ECIA"]
    params = {
        "EGreedy": {"alpha": 0.1, "gamma": 0.9, "epsilon": 0.1},
        "UCB": {"alpha": 0.1, "gamma": 0.9, "c": 0.5},
        "TS": {"alpha": 0.1, "gamma": 0.9},
        "CDQL": {"alpha": 0.1, "gamma": 0.9, "alpha_high": 0.3, "window_size": 20, "threshold": 3.0},
        "ECIA": {"alpha": 0.1, "gamma": 0.9, "eta": 0.5, "xi": 0.1, "emotion_dim": 8, "multi_core": True}
    }
    
    print(f"환경 {env_type} 실험: {num_runs}회 실행, {max_steps}회 시행")
    results = exp_runner.run_batch_experiments(
        env_types=[env_type],
        agent_types=agent_types,
        num_runs=num_runs,
        max_steps=max_steps,
        params=params,
        n_jobs=8
    )
    
    analysis = exp_runner.analyze_results(
        results, env_type=env_type, save_plots=save_plots
    )
    
    return results, analysis


if __name__ == "__main__":
    # 전체 종합 실험 (A, B, C 환경 모두)
    # results, analysis = run_comprehensive_experiments(num_runs=1000, max_steps=200)
    
    # 환경 A만 테스트
    results, analysis = run_single_environment_experiment(env_type="A", num_runs=1000, max_steps=200)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"comprehensive_analysis_{timestamp}.pkl", "wb") as f:
        pickle.dump(analysis, f)
    
    print(f"분석 결과가 comprehensive_analysis_{timestamp}.pkl 에 저장되었습니다.")
