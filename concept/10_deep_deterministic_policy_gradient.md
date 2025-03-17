# 심층 확정적 정책 그래디언트(Deep Deterministic Policy Gradient, DDPG)

## 1. DDPG 알고리즘 등장 배경

### 기존 알고리즘의 한계
- **DQN (Deep Q-Network)의 특징**:
  - **장점**: 
    - Off-policy 알고리즘으로 데이터 효율성 높음 (Experience Replay 사용)
    - 안정적인 학습 (Target Network 사용)
  - **단점**: 
    - 연속적인 행동 공간(Continuous Action Space)에 적용 어려움
    - 이유: DQN은 Q 값을 최대화하는 행동을 선택하기 위해 max 연산을 수행해야 하는데, 연속적인 행동 공간에서는 이 max 연산을 실행하기 어려움

- **Actor-Critic 알고리즘의 특징**:
  - **장점**: 
    - 연속적인 행동 공간에 적용 가능 (Policy Gradient 사용)
    - 분산이 낮아 안정적인 학습
  - **단점**: 
    - 대부분 On-policy 알고리즘으로 데이터 효율성이 낮음
    - 같은 데이터를 한 번만 학습에 사용 가능

### DDPG의 필요성
- DQN의 장점(데이터 효율성)과 Actor-Critic의 장점(연속적인 행동 공간 대응)을 결합한 알고리즘 필요
- Off-policy Actor-Critic 알고리즘으로 설계됨
- 심층 강화학습의 성능을 향상시키면서 연속적인 행동 공간에서도 효과적으로 작동

## 2. DDPG 알고리즘의 특징

### Deterministic Policy (확정적 정책)
- **정의**: 상태가 주어지면 확정적인(deterministic) 행동을 선택하는 정책
- **수식**: a_t = π_θ(s_t)
- **기존 Stochastic Policy와의 비교**:
  - **확률적 정책(Stochastic Policy)**: 상태에 대해 행동의 확률 분포를 출력 (예: 평균, 표준편차)
  - **확정적 정책(Deterministic Policy)**: 상태에 대해 직접 행동 값을 출력
- **장점**: 
  - 행동 공간이 큰 경우 효율적
  - 정책 그래디언트의 분산이 작아 학습이 안정적

### Off-policy 알고리즘
- 확정적 정책을 사용하므로 Off-policy 학습이 자연스러움
  - 이유: 확정적 정책은 행동을 명확히 선택하므로, 다른 정책으로 생성된 데이터도 활용 가능
- Experience Replay 사용 가능 (데이터 효율성 높음)
  - 같은 데이터를 여러 번 학습에 재사용

### Soft Target Update 기법
- **문제**: 
  - Critic 네트워크 업데이트 시 TD Target (y_i = r_i + γQ_ω'(s_(i+1), π_θ'(s_(i+1))))도 함께 변동하여 학습 불안정
  - 학습 목표가 계속 변하는 문제 발생
- **해결**: Target Network 사용
  - **Target Network**: TD Target 계산에 사용되는 별도의 네트워크
  - **Soft Target Update**: Target Network의 파라미터를 주기적으로, 천천히 업데이트
  - **수식**:
    - ω' = τω + (1-τ)ω' (Critic Target Network)
    - θ' = τθ + (1-τ)θ' (Actor Target Network)
    - τ: 작은 상수 (예: 0.001)
- **장점**: 학습 목표가 안정적으로 유지되어 학습 안정성 향상

### 탐험 전략 (Exploration)
- **확정적 정책의 탐험 문제**: 
  - 확정적 정책은 항상 같은 상태에서 같은 행동을 선택하므로 탐험이 어려움
- **해결**: 행동에 노이즈 추가
  - **수식**: a_t = π_θ(s_t) + ε_t
  - ε_t: 노이즈 (예: 정규 분포에서 샘플링, Ornstein-Uhlenbeck 노이즈)
- **Ornstein-Uhlenbeck 노이즈**:
  - 시간적으로 상관관계가 있는 노이즈 (temporal correlation)
  - 물리적 제어 문제에서 효과적인 탐험 제공
  - 이전 노이즈 상태에 의존하여 새 노이즈를 생성 (관성이 있는 움직임)

## 3. DDPG 목적 함수 유도

### 기본 목적 함수
- **DDPG 최종 목적 함수**: J(θ) = E[Q_ω(s_i, π_θ(s_i))]
  - θ: Actor 네트워크 파라미터
  - ω: Critic 네트워크 파라미터
  - π_θ(s_i): 상태 s_i에서 확정적 정책 π_θ에 의해 결정되는 행동
  - Q_ω(s_i, π_θ(s_i)): 상태 s_i에서 확정적 정책에 의해 결정된 행동의 Q 값
  - 목표: 이 Q 값의 기댓값을 최대화하는 정책 파라미터 θ를 찾는 것

### 목적 함수 유도 과정
1. **시작점**: 리턴의 기댓값으로 표현된 목적 함수
   - J(θ) = E[G₀]
   - = ∫_(s₀, a₀, s₁, a₁, ..., s_T, a_T) G₀ p_θ(s₀, a₀, s₁, a₁, ..., s_T, a_T) ds₀da₀ds₁da₁...ds_Tda_T
   - = ∫_(τ_(s₀:a_T)) G₀ p_θ(τ_(s₀:a_T)) dτ_(s₀:a_T)
   - τ_(s₀:a_T): 궤적 (trajectory)

2. **조건부 확률의 연쇄 법칙 적용**
   - p(x, y) = p(y|x)p(x)
   - J(θ) = ∫_(s₀) ∫_(τ_(a₀:a_T)) G₀ p_θ(τ_(a₀:a_T)|s₀) p(s₀) dτ_(a₀:a_T) ds₀

3. **상태 가치 함수 도입**
   - ∫_(τ_(a₀:a_T)) G₀ p_θ(τ_(a₀:a_T)|s₀) dτ_(a₀:a_T) = V(s₀) (상태 가치 함수)
   - J(θ) = ∫_(s₀) V(s₀)p(s₀)ds₀

4. **확정적 정책 적용**
   - a_t = π_θ(s_t) (상태가 주어지면 행동은 확정적)
   - V(s₀) = Q(s₀, a₀) (상태 가치 함수는 Q 값으로 표현)
   - J(θ) = ∫_(s₀) Q(s₀, a₀)p(s₀)ds₀

5. **Q 함수 전개**
   - Q(s₀, a₀) = ∫_(τ_(s₁:a_T)) G₀ p_θ(τ_(s₁:a_T)|s₀, a₀) dτ_(s₁:a_T)
   - = ∫_(s₁, a₁) ∫_(τ_(s₂:a_T)) (r₀ + γG₁) p_θ(τ_(s₂:a_T)|s₁, a₁) dτ_(s₂:a_T) p_θ(s₁, a₁|s₀, a₀) dτ_(s₁:a₁)
   - = ∫_(s₁, a₁) (r₀ + γQ(s₁, a₁)) p_θ(s₁, a₁|s₀, a₀) dτ_(s₁:a₁)
   - = ∫_(s₁) (r₀ + γQ(s₁, π_θ(s₁))) p(s₁|s₀, a₀) dτ_(s₁:a₁) (MDP 특징, 확정적 정책)

6. **목적 함수 미분 (∇_θ J(θ))**
   - ∇_θ J(θ) = ∫_(s) ∇_θ Q(s, π_θ(s)) p(s) ds
   - = ∫_(s) ∇_a Q(s, a)|_(a=π_θ(s)) ∇_θ π_θ(s) p(s) ds
   - = E_s[∇_a Q(s, a)|_(a=π_θ(s)) ∇_θ π_θ(s)]

7. **Deterministic Policy Gradient 정리**
   - ∇_θ J(θ) = E_s[∇_a Q_ω(s, a)|_(a=π_θ(s)) ∇_θ π_θ(s)]
   - 실제 구현에서는 간단히: ∇_θ J(θ) ≈ E[∇_θ Q_ω(s, π_θ(s))]

8. **최종 목적 함수**
   - J(θ) = E[Q_ω(s, π_θ(s))]
   - 정책 파라미터 θ에 대한 미분: ∇_θ J(θ) = E[∇_a Q_ω(s, a)|_(a=π_θ(s)) ∇_θ π_θ(s)]

## 4. DDPG 알고리즘 학습 과정

### 네트워크 구조
- **Actor Network**: 상태를 입력받아 행동을 출력
  - 상태 → 행동 (s → a = π_θ(s))
- **Critic Network**: 상태와 행동을 입력받아 Q 값을 출력
  - (상태, 행동) → Q 값 ((s, a) → Q_ω(s, a))
- **Target Networks**: Actor와 Critic 각각의 Target Network
  - Actor Target Network: π_θ'
  - Critic Target Network: Q_ω'

### 학습 단계
1. **초기화**:
   - Actor Network 파라미터 θ, Critic Network 파라미터 ω 초기화
   - Target Networks 파라미터 θ', ω' 초기화 (θ' ← θ, ω' ← ω)
   - Replay Buffer 초기화

2. **데이터 수집**:
   - 현재 정책 π_θ에 노이즈를 추가하여 행동 선택: a_t = π_θ(s_t) + ε_t
   - 환경과 상호작용하여 경험 (s_t, a_t, r_t, s_(t+1)) 생성
   - Replay Buffer에 경험 저장

3. **미니배치 샘플링**:
   - Replay Buffer에서 미니배치 크기만큼 경험 샘플링

4. **Critic Network 업데이트**:
   - TD Target 계산:
     - y_i = r_i + γQ_ω'(s_(i+1), π_θ'(s_(i+1)))
   - 손실 함수 계산:
     - L(ω) = (1/N) ∑_i (y_i - Q_ω(s_i, a_i))²
   - 경사 하강법으로 Critic Network 파라미터 ω 업데이트:
     - ω ← ω - α_ω∇_ω L(ω)

5. **Actor Network 업데이트**:
   - 손실 함수 계산:
     - L(θ) = -(1/N) ∑_i Q_ω(s_i, π_θ(s_i))
   - 경사 하강법으로 Actor Network 파라미터 θ 업데이트:
     - θ ← θ - α_θ∇_θ L(θ)

6. **Target Networks 업데이트** (Soft Target Update):
   - Critic Target Network: ω' ← τω + (1-τ)ω'
   - Actor Target Network: θ' ← τθ + (1-τ)θ'
   - τ: Soft Update 비율 (작은 값, 예: 0.001)

7. **반복**: 2~6 단계를 충분한 시간 동안 반복

## 5. DDPG 알고리즘 구현

### Actor-Critic 네트워크 구조
```python
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, action_dim)
        self.max_action = max_action
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        # tanh 함수로 -1~1 사이의 값 출력, max_action을 곱해서 범위 조정
        return self.max_action * torch.tanh(self.layer3(x))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # 상태 처리 레이어
        self.layer1 = nn.Linear(state_dim, 400)
        # 상태+행동 처리 레이어
        self.layer2 = nn.Linear(400 + action_dim, 300)
        self.layer3 = nn.Linear(300, 1)
        
    def forward(self, x, u):
        # 상태를 처리
        x = F.relu(self.layer1(x))
        # 상태와 행동을 연결
        x = F.relu(self.layer2(torch.cat([x, u], 1)))
        # Q 값 출력
        return self.layer3(x)
```

### Ornstein-Uhlenbeck 노이즈 구현
```python
class OUNoise:
    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu
        
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
```

### DDPG 알고리즘 구현
```python
class DDPG:
    def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.001):
        # 네트워크 초기화
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
        
        self.discount = discount
        self.tau = tau
        self.noise = OUNoise(action_dim)
    
    def select_action(self, state, add_noise=True):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]
        if add_noise:
            action += self.noise.sample()
        return action
    
    def train(self, replay_buffer, batch_size=100):
        # Replay Buffer에서 배치 샘플링
        state, action, next_state, reward, done = replay_buffer.sample(batch_size)
        
        # Critic 업데이트
        # Target Q 값 계산
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_Q = self.critic_target(next_state, next_action)
            target_Q = reward + (1 - done) * self.discount * target_Q
        
        # 현재 Q 값 계산
        current_Q = self.critic(state, action)
        
        # Critic 손실 계산 및 업데이트
        critic_loss = F.mse_loss(current_Q, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor 업데이트
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Target Networks 업데이트 (Soft Target Update)
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
```

### 학습 루프 구현
```python
def main():
    env = gym.make("Pendulum-v0")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # DDPG 에이전트 초기화
    agent = DDPG(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer()
    
    # 학습 파라미터
    max_episodes = 1000
    max_steps = 500
    
    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # 행동 선택 및 실행
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            # 경험 저장
            replay_buffer.add((state, action, next_state, reward, float(done)))
            
            # 모델 학습
            if len(replay_buffer) > 100:  # 충분한 데이터가 모이면 학습 시작
                agent.train(replay_buffer)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        print(f"Episode {episode}: reward = {episode_reward}")
```

## 6. DDPG의 이슈와 확장

### 주요 이슈
- **학습 불안정성**: 
  - Soft Target Update와 Experience Replay를 사용해도 여전히 학습이 불안정할 수 있음
  - 하이퍼파라미터에 민감함
- **샘플 효율성**: 
  - Off-policy이지만 여전히 많은 샘플을 필요로 함
- **탐험-활용 균형**: 
  - 적절한 탐험 전략 선택이 어려움
  - 노이즈 파라미터 튜닝이 까다로움

### DDPG 확장
- **TD3 (Twin Delayed DDPG)**: 
  - 두 개의 Critic 네트워크를 사용하여 Q 값 과대추정 문제 해결
  - Actor 업데이트 지연 (Delayed Policy Updates)으로 학습 안정성 확보
- **SAC (Soft Actor-Critic)**: 
  - 정책 엔트로피를 고려하여 탐험-활용 균형 개선
  - 더욱 안정적인 학습과 샘플 효율성 향상
- **D4PG (Distributed Distributional DDPG)**: 
  - 분산 학습과 분포 강화학습을 결합하여 성능 향상
  - 병렬 처리를 통한 학습 속도 개선

## 요약

1. **DDPG 등장 배경**:
   - DQN의 데이터 효율성과 Actor-Critic의 연속 행동 공간 처리 능력을 결합
   - 연속적인 행동 공간에서 효과적으로 작동하는 Off-policy 알고리즘

2. **주요 특징**:
   - 확정적 정책(Deterministic Policy) 사용
   - Off-policy 학습으로 데이터 효율성 향상
   - Soft Target Update를 통한 학습 안정성 확보
   - 노이즈 추가를 통한 효과적인 탐험 전략

3. **목적 함수**:
   - J(θ) = E[Q_ω(s, π_θ(s))]
   - Actor와 Critic을 번갈아 업데이트하여 학습

4. **학습 과정**:
   - Actor Network: 상태를 입력받아 행동 출력
   - Critic Network: 상태와 행동을 입력받아 Q 값 출력
   - Experience Replay: 과거 경험 재사용
   - Soft Target Update: 안정적인 학습 목표 유지

5. **DDPG의 의의**:
   - 연속적인 행동 공간 문제에서 효과적인 성능
   - DQN과 Actor-Critic의 장점을 결합한 효율적인 알고리즘
   - 후속 알고리즘(TD3, SAC 등)의 기반이 됨
