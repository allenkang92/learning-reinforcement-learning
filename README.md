# Learning Reinforcement Learning

강화 학습(Reinforcement Learning) 개인 아카이빙..

## 목차

1. [강화학습 개요](./concept/01_reinforcement_learning_overview.md)
   - 강화학습의 기본 개념과 구조
   - 지도학습 및 비지도학습과의 차이점
   - 강화학습의 주요 적용 사례

2. [마르코프 결정 과정](./concept/02_markov_decision_process.md)
   - Markov Process와 Markov Reward Process
   - 마르코프 결정 과정(MDP)의 구성 요소
   - 벨만 방정식과 Dynamic Programming

3. [탐험과 활용 알고리즘](./concept/03_exploration_exploitation_algorithms.md)
   - 탐험(Exploration)과 활용(Exploitation)의 균형
   - Monte Carlo 예측법과 TD 알고리즘 비교
   - SARSA 알고리즘 및 특징

4. [Model-free 가치 기반 강화학습](./concept/04_model_free_value_based_rl.md)
   - Model-free vs Model-based 강화학습
   - On-policy vs Off-policy 알고리즘
   - Q-learning 알고리즘 및 구현

5. [심층 Q 네트워크(DQN)](./concept/05_deep_q_networks.md)
   - 가치 함수 근사(Value Function Approximation)
   - DQN의 핵심 기법: Experience Replay와 Target Network
   - DQN의 주요 발전 형태(Double DQN, Prioritized Experience Replay 등)

6. [Hindsight Experience Replay(HER)](./concept/06_hindsight_experience_replay.md)
   - 희소 보상 환경(Sparse Reward Environment)의 문제
   - Multi-Goal 강화학습과 Universal Value Function Approximators
   - HER 알고리즘 및 구현 방법

7. [정책 기반 강화학습](./concept/07_policy_based_reinforcement_learning.md)
   - 정책 기반 강화학습의 개념과 특징
   - 정책 그래디언트(Policy Gradient)와 목적 함수
   - REINFORCE 알고리즘 및 구현

8. [Actor-Critic 방법](./concept/08_actor_critic_methods.md)
   - Actor-Critic 알고리즘의 구조와 특징
   - A2C(Advantage Actor-Critic) 알고리즘
   - Actor-Critic 목적 함수와 구현 방법

9. [근사 정책 최적화(PPO)](./concept/09_proximal_policy_optimization.md)
   - PPO의 등장 배경과 TRPO와의 관계
   - Clipped Surrogate Objective 및 작동 원리
   - PPO 알고리즘 구현 및 다른 알고리즘과의 비교

10. [심층 확정적 정책 그래디언트(DDPG)](./concept/10_deep_deterministic_policy_gradient.md)
    - 연속적인 행동 공간에서의 강화학습 문제
    - 확정적 정책(Deterministic Policy)과 Off-policy 학습의 결합
    - Soft Target Update와 탐험 전략

11. [쌍둥이 지연 심층 확정적 정책 그래디언트(TD3)](./concept/11_twin_delayed_ddpg.md)
    - Q 값 과대평가 문제와 해결 방안
    - Clipped Double Q-Learning 및 Delayed Policy Updates
    - Target Policy Smoothing Regularization

12. [소프트 액터-크리틱(SAC)](./concept/12_soft_actor_critic.md)
    - 모델 프리 강화학습의 한계와 최대 엔트로피 강화학습
    - 확률적 정책과 자동 온도 조정(Temperature Tuning)
    - Reparameterization Trick과 구현 방법

