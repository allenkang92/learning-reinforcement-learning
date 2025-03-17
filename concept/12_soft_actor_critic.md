# 소프트 액터-크리틱(Soft Actor-Critic, SAC)

## 1. 모델 프리 강화학습의 한계

### 높은 샘플 복잡성 (Sample Complexity)
- **정의**: 모델을 성공적으로 학습하는 데 필요한 데이터(상호작용) 수
- **문제점**: 기존 모델 프리 강화학습 알고리즘은 학습에 많은 데이터가 필요함
  - 예: TD3는 HalfCheetah 환경에서 약 70만 개의 데이터가 필요
  - PPO와 DDPG는 더 많은 데이터를 필요로 함
- **영향**: 실제 물리적 시스템에서는 많은 데이터를 수집하기 어려워 현실 적용이 제한됨

### 학습 파라미터 민감도
- **문제점**: 하이퍼파라미터 값에 따라 성능의 변동이 크게 나타남
  - 학습률, 네트워크 구조, 보상 스케일링 등의 파라미터에 민감
- **영향**: 각 환경과 문제에 맞는 파라미터 튜닝이 필요하여 일반화가 어려움

### 제한된 탐험 범위
- **원인**: 
  - 기존 Actor-Critic 알고리즘은 탐험을 제한적으로만 고려 (ε-greedy 등)
  - 탐험과 활용 사이의 균형을 맞추기 어려움
- **결과**: 
  - 국소 최적해(local optimum)에 쉽게 빠질 가능성
  - 예: 내비게이션 문제에서 이미 알고 있는 경로만 사용하고 잠재적으로 더 좋은 경로를 발견하지 못함

## 2. 최대 엔트로피 강화학습 (Maximum Entropy Reinforcement Learning)

### 엔트로피 (Entropy) 개념
- **정의**: 확률 분포의 불확실성이나 무작위성을 측정하는 지표
- **수학적 정의**:
  - 정보량: h(x) = -log p(x) (p(x)는 사건 x가 발생할 확률)
  - 엔트로피: H(p) = E[-log p(x)] = -∫ p(x)log p(x)dx
- **특성**:
  - 확률 분포가 균등(uniform)할 때 엔트로피가 최대화됨
  - 확률 분포의 분산이 클수록 엔트로피가 증가함
- **예시**:
  - 주사위를 던져서 각 면이 나올 확률이 모두 1/6인 경우: 높은 엔트로피
  - 주사위가 조작되어 특정 면이 나올 확률이 높은 경우: 낮은 엔트로피

### 최대 엔트로피 원칙
- **기본 아이디어**: 주어진 제약 조건 내에서 엔트로피를 최대화하는 확률 분포를 선택
- **강화학습 적용**: 
  - 기존 목적 함수: 리턴의 기댓값 최대화 (E[∑γ^t r_t])
  - 최대 엔트로피 목적 함수: 리턴의 기댓값 + 엔트로피 최대화
    - J(π) = E[∑γ^t (r_t + αH(π(·|s_t)))]
    - α: 온도 파라미터(temperature parameter), 엔트로피의 중요도 조절

### 최대 엔트로피 강화학습의 이점
1. **효율적 탐험**:
   - 에이전트가 다양한 행동을 탐험하도록 장려
   - 불확실성이 높은 영역 탐험 촉진
2. **다중 모드 정책**(Multi-modal policy):
   - 여러 유사한 최적 행동이 있는 경우, 모든 행동을 고려
   - 한 가지 행동에만 의존하지 않고 대안적 행동도 유지
3. **견고성**(Robustness):
   - 노이즈와 불확실성에 강인한 정책 학습
   - 과적합(overfitting) 방지

## 3. Soft Actor-Critic (SAC) 알고리즘 개요

### SAC의 등장 배경
- **목표**: 기존 모델 프리 강화학습의 한계 극복
  - 샘플 복잡성 감소
  - 학습 파라미터 민감도 감소
  - 탐험 범위 확장
- **핵심 아이디어**: 최대 엔트로피 강화학습을 Off-policy Actor-Critic 구조에 결합
- **발표**: Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine - "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor" (ICML 2018)

### 이전 알고리즘들과의 관계
- **SARSA**: On-policy 알고리즘으로 샘플 복잡도가 높음
- **Off-policy 알고리즘**(DQN, DDPG): 
  - Replay Buffer 사용으로 샘플 복잡도 감소
  - 단점: 학습 안정성 및 수렴성 문제
- **Maximum Entropy 알고리즘**: 
  - 탐험 능력 향상에 초점
  - 단점: 샘플 복잡도 문제는 해결하지 못함
- **Soft Q-Learning**:
  - Off-policy + Maximum Entropy 결합
  - 단점: 연속적인 행동 공간에 적용 어려움
- **SAC**: 
  - Soft Q-Learning + Actor-Critic 결합
  - 장점: 샘플 효율성, 안정적 학습, 연속적인 행동 공간 지원

### SAC의 주요 특징
- **확률적 정책**(Stochastic Policy): 행동의 확률 분포를 출력
- **Off-policy 학습**: Experience Replay 사용
- **Maximum Entropy 최적화**: 보상 최대화와 엔트로피 최대화 동시 추구
- **Dual Q-네트워크**: TD3와 유사하게 두 개의 Q-네트워크 사용 (과대평가 방지)
- **자동 온도 조정**(Automatic Temperature Tuning): 엔트로피 가중치(α) 자동 조절

## 4. SAC 알고리즘 구성요소

### 네트워크 구조
1. **Actor 네트워크(Policy Network)**:
   - 입력: 상태 s
   - 출력: 행동 확률 분포의 파라미터 (일반적으로 정규 분포의 평균 μ와 표준편차 σ)
   - 파라미터: φ

2. **Critic 네트워크(Q-function Network)**:
   - 두 개의 독립적인 네트워크 사용 (Clipped Double Q-Learning)
   - 입력: 상태 s, 행동 a
   - 출력: Q 값 (상태-행동 쌍의 가치)
   - 파라미터: θ₁, θ₂

3. **Value 네트워크(V-function Network)** (선택적):
   - 입력: 상태 s
   - 출력: 상태 가치 V(s)
   - 파라미터: ψ
   - 참고: 최신 SAC 구현에서는 종종 생략됨

### 목표 함수(Objective Function)
- **최대 엔트로피 강화학습 목표**:
  ```
  J(π) = E_π[∑γ^t (r_t + αH(π(·|s_t)))]
  ```
  - π: 정책
  - r_t: 보상
  - H(π(·|s_t)): 상태 s_t에서의 정책 엔트로피
  - α: 온도 파라미터 (엔트로피 중요도)

### 소프트 Q-함수와 소프트 V-함수
- **소프트 Q-함수**:
  ```
  Q^π(s, a) = E_π[r(s, a) + γ(r(s₁, a₁) + αH(π(·|s₁))) + ...]
  ```
  - 기존 Q-함수에 엔트로피 항 추가

- **소프트 V-함수**:
  ```
  V^π(s) = E_a~π[Q^π(s, a) - αlogπ(a|s)]
  ```
  - 소프트 Q-함수의 기댓값에서 엔트로피 항을 뺀 값

## 5. SAC 손실 함수 및 학습 과정

### 소프트 Q-함수의 벨만 방정식
```
Q^π(s_t, a_t) = r(s_t, a_t) + γE_{s_{t+1}}[V^π(s_{t+1})]
```
- 소프트 V-함수와의 관계:
  ```
  V^π(s) = E_{a~π}[Q^π(s, a) - αlogπ(a|s)]
  ```

### 손실 함수
1. **Soft Q-함수 손실**:
   ```
   J_Q(θ_i) = E_{(s_t,a_t)~D}[(1/2)(Q_{θ_i}(s_t, a_t) - y_t)²]
   ```
   - y_t = r(s_t, a_t) + γE_{s_{t+1}~p}[V^π(s_{t+1})]
   - 실제로는 Target Network를 사용하여 y_t 계산

2. **정책 손실(Actor 손실)**:
   ```
   J_π(φ) = E_{s_t~D,ε_t~N}[αlog(π_φ(f_φ(ε_t; s_t)|s_t)) - Q(s_t, f_φ(ε_t; s_t))]
   ```
   - f_φ(ε_t; s_t): Reparameterization trick을 사용하여 행동 생성
   - α: 온도 파라미터 (자동 조정 가능)

3. **온도 파라미터 α 손실** (자동 조정 시):
   ```
   J(α) = E_{a_t~π_t}[-α(logπ_t(a_t|s_t) + H̄)]
   ```
   - H̄: 목표 엔트로피 (하이퍼파라미터)

### Reparameterization Trick
- **목적**: Policy Gradient 계산시 샘플링으로 인한 분산을 줄이고 미분 가능하게 만들기
- **방법**:
  1. 정규 분포에서 노이즈 ε ~ N(0, 1) 샘플링
  2. 평균(μ)과 표준편차(σ)를 사용하여 행동 계산: a = μ + σ * ε
- **장점**: 정책 파라미터 φ에 대한 그래디언트를 직접 계산 가능

### SAC 학습 알고리즘 (수도 코드)
```
초기화:
- 정책 파라미터 φ, Q-함수 파라미터 θ₁, θ₂
- Target Network 파라미터 θ̄₁ ← θ₁, θ̄₂ ← θ₂
- 온도 파라미터 α
- 빈 Replay Buffer D

반복 (각 에피소드마다):
    상태 s_t ← 초기 상태
    반복 (각 타임스텝 t마다):
        # 행동 선택 및 실행
        a_t ← π_φ(a_t|s_t) (정책에서 행동 샘플링)
        s_{t+1}, r_t ← 환경과 상호작용(a_t)
        D ← D ∪ {(s_t, a_t, r_t, s_{t+1})} (경험 저장)
        
        # 미니배치 샘플링 및 네트워크 업데이트
        미니배치 B = {(s_j, a_j, r_j, s_{j+1})} ← D에서 샘플링
        
        # Q-함수 업데이트
        y_j = r_j + γ(min_{i=1,2}Q_{θ̄_i}(s_{j+1}, ã_{j+1}) - αlogπ_φ(ã_{j+1}|s_{j+1}))
            (여기서 ã_{j+1} ← π_φ(·|s_{j+1})에서 샘플링)
        θ_i ← θ_i - λ_Q∇_{θ_i}J_Q(θ_i) (각각의 θ₁, θ₂에 대해 수행)
        
        # 정책 업데이트
        ã_j ← f_φ(ε; s_j) (Reparameterization trick 사용)
        φ ← φ - λ_π∇_φJ_π(φ)
        
        # 온도 파라미터 업데이트 (자동 조정 시)
        α ← α - λ_α∇_αJ(α)
        
        # Target Network 업데이트
        θ̄_i ← τθ_i + (1-τ)θ̄_i (Soft update, τ«1)
        
    s_t ← s_{t+1}
```

## 6. SAC 구현 (PyTorch)

### Actor 네트워크 구현
```python
class SACPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super(SACPolicy, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Actor 네트워크 (확률적 정책)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 평균과 로그 표준편차를 출력하는 레이어
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        x = self.actor(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        return mean, std
    
    def sample(self, state):
        mean, std = self.forward(state)
        
        # 정규 분포 생성
        normal = Normal(mean, std)
        
        # Reparameterization trick: x ~ N(μ, σ) -> z = μ + σ * ε, ε ~ N(0, 1)
        x_t = normal.rsample()  # rsample(): reparameterized sample
        
        # tanh squashing
        action = torch.tanh(x_t)
        
        # 로그 확률 계산 (tanh squashing 고려)
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        return action, log_prob
```

### Critic 네트워크 구현
```python
class SACCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(SACCritic, self).__init__()
        
        # Q1 아키텍처
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q2 아키텍처
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        
        q1 = self.q1(x)
        q2 = self.q2(x)
        
        return q1, q2
    
    def q1_forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.q1(x)
```

### SAC 손실 함수 구현
```python
def compute_critic_loss(self, batch):
    state, action, reward, next_state, done = batch
    
    # 현재 Q 값 계산
    current_q1, current_q2 = self.critic(state, action)
    
    # 다음 행동 샘플링
    with torch.no_grad():
        next_action, next_log_prob = self.actor.sample(next_state)
        
        # TD Target 계산
        target_q1, target_q2 = self.critic_target(next_state, next_action)
        target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
        target_q = reward + (1 - done) * self.gamma * target_q
    
    # Critic 손실 = MSE(현재 Q, Target Q)
    critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
    
    return critic_loss

def compute_actor_loss(self, batch):
    state, _, _, _, _ = batch
    
    # 행동 샘플링
    action, log_prob = self.actor.sample(state)
    
    # Q 값 계산 (두 Q 네트워크 중 작은 값)
    q1, q2 = self.critic(state, action)
    min_q = torch.min(q1, q2)
    
    # Actor 손실 = E[α*logπ(a|s) - Q(s,a)]
    actor_loss = (self.alpha * log_prob - min_q).mean()
    
    return actor_loss

def compute_temperature_loss(self, batch):
    state, _, _, _, _ = batch
    
    _, log_prob = self.actor.sample(state)
    
    # α 손실 = E[-α*(logπ(a|s) + target_entropy)]
    alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
    
    return alpha_loss
```

### 학습 과정 구현
```python
def train(self, replay_buffer, batch_size=256):
    # Replay Buffer에서 미니배치 샘플링
    batch = replay_buffer.sample(batch_size)
    
    # Critic 손실 계산 및 업데이트
    critic_loss = self.compute_critic_loss(batch)
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()
    
    # Actor 손실 계산 및 업데이트
    actor_loss = self.compute_actor_loss(batch)
    self.actor_optimizer.zero_grad()
    actor_loss.backward()
    self.actor_optimizer.step()
    
    # 온도 파라미터(α) 자동 조정 (학습)
    if self.automatic_entropy_tuning:
        alpha_loss = self.compute_temperature_loss(batch)
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # α 값 업데이트
        self.alpha = torch.exp(self.log_alpha.detach())
    
    # Target Network Soft Update
    for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
```

## 7. SAC의 장단점 및 응용

### 장점
1. **샘플 효율성**:
   - Off-policy 학습과 Experience Replay 사용으로 데이터 효율적
   - 기존 알고리즘보다 적은 데이터로 학습 가능

2. **안정적인 학습**:
   - Maximum Entropy 목적으로 탐험-활용 균형 자동 조절
   - Clipped Double Q-Learning으로 과대평가 문제 완화
   - 하이퍼파라미터 민감도 감소

3. **효과적인 탐험**:
   - 엔트로피 최대화로 다양한 행동 시도
   - 국소 최적해 탈출 능력 향상

4. **다중 모드 정책 학습**:
   - 여러 최적 행동이 있는 경우 다양한 전략 습득
   - 확률적 정책으로 환경의 불확실성에 대응

### 단점
1. **계산 복잡성**:
   - 여러 네트워크 학습으로 계산 비용 증가
   - DDPG, TD3보다 학습 속도가 느릴 수 있음

2. **하이퍼파라미터 최적화**:
   - 자동 엔트로피 조정 사용 시 목표 엔트로피 설정 필요
   - 네트워크 구조, 학습률 등 여전히 튜닝 필요

3. **On-policy 요소**:
   - 완전한 Off-policy가 아닌, 일부 On-policy 요소 포함
   - Replay Buffer에서 오래된 데이터 사용 시 성능 감소 가능성

### 응용 분야
1. **로보틱스**:
   - 로봇 제어, 조작 작업
   - 다양한 동작 패턴 학습 가능

2. **자율주행**:
   - 복잡한 주행 환경에서 다양한 주행 전략 학습
   - 불확실성 대처 능력 향상

3. **게임 AI**:
   - 다양한 전략을 요구하는 게임에서 활용
   - 예측 불가능한 행동으로 흥미로운 AI 구현

## 요약

1. **모델 프리 강화학습의 한계**:
   - 높은 샘플 복잡성
   - 학습 파라미터 민감도
   - 제한된 탐험 범위

2. **최대 엔트로피 강화학습**:
   - 리턴 최대화와 엔트로피 최대화를 동시에 추구
   - 다양한 행동 탐험 장려
   - 불확실성에 강인한 정책 학습

3. **SAC 알고리즘의 주요 특징**:
   - Off-policy Actor-Critic 구조
   - 확률적 정책 사용
   - 최대 엔트로피 목적 함수
   - Clipped Double Q-Learning
   - 자동 온도 조정 (α)

4. **SAC의 주요 장점**:
   - 높은 샘플 효율성
   - 안정적인 학습
   - 효과적인 탐험
   - 하이퍼파라미터 민감도 감소

5. **SAC의 구성요소**:
   - Actor 네트워크: 확률적 정책 (정규 분포 파라미터)
   - Critic 네트워크: 두 개의 Q-함수
   - 소프트 벨만 방정식과 최대 엔트로피 목적 함수
   - Reparameterization Trick을 통한 효율적인 정책 학습
