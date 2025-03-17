# 쌍둥이 지연 심층 확정적 정책 그래디언트(Twin Delayed DDPG, TD3)

## 1. Q 값 과대평가 문제

### 과대평가 문제의 정의
- **정의**: Critic 네트워크가 상태 및 행동에 대한 가치(Q 값)를 실제보다 높게 평가하는 경향
- **문제점**:
  - 나쁜 행동을 선택하도록 유도 (가치가 낮은 행동의 Q 값을 과대평가)
  - 강화 학습 모델의 성능 저하
  - 특히 함수 근사(Function Approximation)를 사용하는 딥러닝 기반 강화학습에서 더 두드러짐

### 발생 원인
- **부트스트래핑 에러(Bootstrapping Error)**:
  - **정의**: 예측값을 사용하여 또 다른 값을 예측하면서 발생하는 에러
  - Q-learning의 업데이트 식: Q(S_t, A_t) ← Q(S_t, A_t) + α(R_t+1 + γ⋅max_a Q(S_t+1, a) - Q(S_t, A_t))
  - max 연산자: 다음 상태의 여러 행동 중 가장 큰 Q 값 선택
  - **문제**: 추정된 Q 값에는 노이즈가 있음. max 연산자는 이 중 가장 큰 값(가장 큰 노이즈 포함)을 선택
  - 결과: 양의 추정 오차가 누적되어 Q 값의 과대평가 발생

### 수학적 증명
- **Actor-Critic에서의 과대평가**:
  1. **정책 업데이트**:
     - **근사 정책 파라미터(ϕ_approx)**: ϕ_approx = ϕ + (α/Z₁)E_s[∇_ϕπ_ϕ(s)∇_a Q_θ(s, a)|_{a=π_ϕ(s)}]
     - **참 정책 파라미터(ϕ_true)**: ϕ_true = ϕ + (α/Z₂)E_s[∇_ϕπ_ϕ(s)∇_a Q_π(s, a)|_{a=π_ϕ(s)}]
     - α: 학습률, Z₁, Z₂: 정규화 상수
  
  2. **가정**: 충분히 작은 학습률 α 일 때:
     - E[Q_θ(s, π_approx(s))] ≥ E[Q_θ(s, π_true(s))]
     - E[Q_π(s, π_true(s))] ≥ E[Q_π(s, π_approx(s))]
  
  3. **결론**:
     - E[Q_θ(s, π_true(s))] ≥ E[Q_π(s, π_true(s))] → 과대평가
     - E[Q_θ(s, π_approx(s))] ≥ E[Q_π(s, π_approx(s))] → 과대평가
     - Q 값 과대평가 에러 = E[Q_θ(s, π_approx(s))] - E[Q_π(s, π_approx(s))] ≥ 0
  
  4. **악순환**:
     - Critic의 과대평가된 값을 기반으로 Actor 업데이트
     - Actor는 실제로는 최적이 아닌 행동 선택
     - 새로운 데이터가 Replay Buffer에 추가되고, 이를 기반으로 Critic 다시 학습
     - 과대평가 오차가 계속 누적되어 정책 성능 저하

## 2. TD3 알고리즘 개요

### TD3의 의의
- **DDPG의 확장**: DDPG 알고리즘의 문제점(특히 Q 값 과대평가)을 해결한 알고리즘
- **목적**: 가치 함수(Q 값)의 정확한 추정을 통한 정책 성능 향상
- **발표**: Scott Fujimoto, Herke van Hoof, David Meger - "Addressing Function Approximation Error in Actor-Critic Methods" (ICML 2018)

### TD3의 핵심 아이디어
1. **Clipped Double Q-Learning**: 과소평가를 유도하여 과대평가 완화
2. **Delayed Policy Updates**: 정책 업데이트 지연을 통한 학습 안정성 향상
3. **Target Policy Smoothing Regularization**: 타겟 정책에 노이즈 추가로 분산 감소

### DDPG vs TD3 비교
- **공통점**:
  - Off-policy Actor-Critic 알고리즘
  - 확정적 정책(Deterministic Policy) 사용
  - Experience Replay와 Target Networks 사용
- **차이점**:
  - DDPG: 하나의 Critic 네트워크
  - TD3: 두 개의 Critic 네트워크 + 세 가지 핵심 개선 기법 적용

## 3. Clipped Double Q-Learning

### Double Q-Learning 배경
- **Double DQN**의 아이디어: 행동 선택과 평가에 사용되는 가치 함수를 분리하여 과대평가 감소
- **원리**: 하나의 네트워크로 행동 선택, 다른 네트워크로 해당 행동의 Q 값 평가
- **Double DQN의 Target**:
  ```
  y = r + γ⋅Q(s', argmax_a Q(s', a; θ); θ')
  ```
  - θ: 행동 선택에 사용되는 파라미터
  - θ': Q 값 평가에 사용되는 파라미터

### TD3의 Clipped Double Q-Learning
- **구조**: 두 개의 독립적인 Critic 네트워크(Q_θ₁, Q_θ₂)와 Target Networks(Q_θ₁', Q_θ₂')
- **TD Target 계산**: 두 Target Q 값 중 작은 값을 사용
  ```
  y = r + γ⋅min(Q_θ₁'(s', a'), Q_θ₂'(s', a'))
  ```
  - a': Target Actor Network에서 선택한 행동
  - min(): 두 값 중 작은 값 선택 → 의도적인 과소평가 유도

### 효과
- **과소평가 경향**: 의도적으로 Q 값을 과소평가하여 과대평가로 인한 문제 상쇄
- **분산 감소**: 여러 Q 함수의 평균을 취하는 것처럼 작용하여 분산 감소
- **안정성 증가**: 정책이 실제 성능보다 높게 평가된 행동을 선택할 가능성 감소

### 구현
```python
# TD3에서의 Clipped Double Q-Learning 구현
def calculate_target(self, next_states, rewards, done):
    # Target Actor를 사용하여 다음 행동 선택
    next_actions = self.actor_target(next_states)
    
    # 두 Target Critic의 Q 값 계산
    target_Q1, target_Q2 = self.critic_target(next_states, next_actions)
    
    # 두 Q 값 중 작은 값 선택
    target_Q = torch.min(target_Q1, target_Q2)
    
    # TD Target 계산
    target_Q = rewards + (1 - done) * self.gamma * target_Q
    
    return target_Q
```

## 4. Delayed Policy Updates

### 배경
- **문제**: Actor-Critic에서 Actor와 Critic을 동시에 업데이트하면 학습 불안정
- **이유**: 
  - Critic이 정확한 Q 값을 추정하기 전에 Actor가 업데이트되면 잘못된 방향으로 학습
  - Actor 업데이트로 정책이 변하면 Critic의 학습 목표가 자주 변동

### TD3의 Delayed Policy Updates
- **아이디어**: Critic 네트워크를 여러 번 업데이트한 후 Actor 네트워크를 한 번만 업데이트
- **매개변수**: 업데이트 지연 파라미터 d (보통 d=2)
- **과정**:
  1. 매 타임스텝: 두 Critic 네트워크 업데이트
  2. d 타임스텝마다: Actor 네트워크와 Target Networks 업데이트

### 효과
- **학습 안정성**: Actor 네트워크가 좀 더 정확한 Q 값에 기반하여 업데이트됨
- **수렴 속도**: 정책이 안정적으로 유지되어 Critic 학습이 더 효율적
- **오버피팅 방지**: 과도한 정책 업데이트 방지

### 구현
```python
# TD3에서의 Delayed Policy Updates 구현
def train(self, replay_buffer, batch_size=100):
    # Replay Buffer에서 미니배치 샘플링
    state, action, next_state, reward, done = replay_buffer.sample(batch_size)
    
    # Target Q 값 계산 (Clipped Double Q-Learning)
    target_Q = self.calculate_target(next_state, reward, done)
    
    # Critic 네트워크 업데이트
    current_Q1, current_Q2 = self.critic(state, action)
    critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()
    
    # 정책 업데이트 지연 (매 d 스텝마다)
    if self.total_it % self.policy_freq == 0:
        # Actor 네트워크 업데이트
        actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Target Networks 업데이트
        self.update_target_networks()
    
    self.total_it += 1
```

## 5. Target Policy Smoothing Regularization

### 배경
- **문제**: 확정적 정책(Deterministic Policy)은 하나의 상태에 대해 하나의 행동만 할당하여 과적합 위험
- **결과**: Q 함수가 특정 행동에 대해 과대평가되면, 정책이 그 행동에만 의존하게 됨

### TD3의 Target Policy Smoothing
- **아이디어**: Target 정책에 노이즈를 추가하여 비슷한 행동이 비슷한 가치를 갖도록 정규화
- **직관**: 서로 가까운 행동들이 유사한 Q 값을 가져야 함 → 함수 근사기의 일반화 능력 향상
- **구현**: 
  ```
  a' = clip(π'(s') + ε, a_min, a_max)
  ε ~ clip(N(0, σ), -c, c)
  ```
  - π'(s'): Target Actor의 행동
  - ε: 노이즈 (정규 분포에서 샘플링 후 -c와 c 사이로 클리핑)
  - a_min, a_max: 행동의 최소/최대 값

### 효과
- **평활화(Smoothing)**: Q 함수의 극단적인 피크 완화
- **분산 감소**: 유사한 행동에 대해 일관된 Q 값 유지
- **일반화**: 과적합 방지 및 전이 학습(Transfer Learning) 능력 향상

### 구현
```python
# TD3에서의 Target Policy Smoothing 구현
def calculate_target(self, next_states, rewards, done):
    # Target Actor를 사용하여 다음 행동 선택
    next_actions = self.actor_target(next_states)
    
    # 노이즈 추가 (Target Policy Smoothing)
    noise = torch.randn_like(next_actions) * self.policy_noise
    noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
    next_actions = torch.clamp(next_actions + noise, -self.max_action, self.max_action)
    
    # 두 Target Critic의 Q 값 계산 및 최소값 선택
    target_Q1, target_Q2 = self.critic_target(next_states, next_actions)
    target_Q = torch.min(target_Q1, target_Q2)
    
    # TD Target 계산
    target_Q = rewards + (1 - done) * self.gamma * target_Q
    
    return target_Q
```

## 6. TD3 알고리즘 전체 구조

### 알고리즘 수도 코드
```
Initialize critic networks Q_θ₁, Q_θ₂ and actor network π_ϕ with random parameters
Initialize target networks θ₁' ← θ₁, θ₂' ← θ₂, ϕ' ← ϕ
Initialize replay buffer R

For episode = 1, M do:
    Initialize state s₁
    For t = 1, T do:
        # 행동 선택 및 실행
        Select action with exploration noise: a_t = π_ϕ(s_t) + ε,  ε ~ N(0, σ)
        Execute a_t, observe reward r_t and next state s_t+1
        Store transition (s_t, a_t, r_t, s_t+1) in R

        # 미니배치 샘플링
        Sample a random minibatch of N transitions (s_i, a_i, r_i, s_i+1) from R

        # Target Policy Smoothing
        ã_i+1 = π_ϕ'(s_i+1) + ε
        ε ~ clip(N(0, σ̃), -c, c)
        ã_i+1 = clip(ã_i+1, a_min, a_max)

        # Clipped Double Q-Learning
        y_i = r_i + γ min(Q_θ₁'(s_i+1, ã_i+1), Q_θ₂'(s_i+1, ã_i+1))

        # Critic 업데이트
        L(θ_j) = (1/N) ∑_i (y_i - Q_θ_j(s_i, a_i))²  for j = 1, 2
        Update critics using gradient descent: θ_j ← θ_j - η_Q ∇_θ_j L(θ_j)

        # Delayed Policy Updates (매 d 스텝마다)
        If t mod d == 0:
            # Actor 업데이트
            J(ϕ) = (1/N) ∑_i Q_θ₁(s_i, π_ϕ(s_i))
            Update actor using gradient ascent: ϕ ← ϕ + η_π ∇_ϕ J(ϕ)

            # Target Networks 업데이트
            θ_j' ← τθ_j + (1-τ)θ_j'  for j = 1, 2
            ϕ' ← τϕ + (1-τ)ϕ'
    End For
End For
```

### 네트워크 아키텍처
- **Actor Network**: 상태를 입력받아 확정적인 행동 출력
  - 입력: 상태 s
  - 출력: 행동 a = π_ϕ(s)
  - 활성화 함수: ReLU(은닉층), tanh(출력층)

- **Critic Networks**: 상태와 행동을 입력받아 Q 값 출력
  - 입력: 상태 s, 행동 a
  - 출력: Q 값 Q(s, a)
  - 두 개의 독립적인 네트워크(Q_θ₁, Q_θ₂)

### 하이퍼파라미터
- **일반**:
  - 감가율(γ): 0.99
  - Replay Buffer 크기: 10⁶
  - 배치 크기: 100
  - 최대 에피소드: 1000
  
- **TD3 특화**:
  - 정책 업데이트 지연(d): 2
  - 정책 노이즈(σ): 0.1 ~ 0.2 (환경에 따라 다름)
  - 노이즈 클리핑(c): 0.5
  - Target 네트워크 업데이트 비율(τ): 0.005

## 7. TD3 구현

### Actor-Critic 네트워크 구조
```python
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action
        
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = torch.tanh(self.l3(a)) * self.max_action
        return a

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        # Q1 아키텍처
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        
        # Q2 아키텍처
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)
        
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        
        return q1, q2
    
    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1
```

### TD3 클래스 구현
```python
class TD3:
    def __init__(
        self, 
        state_dim, 
        action_dim, 
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2
    ):
        # 네트워크 초기화
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        
        # 하이퍼파라미터
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise * max_action
        self.noise_clip = noise_clip * max_action
        self.policy_freq = policy_freq
        
        # 카운터
        self.total_it = 0
    
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))
        return self.actor(state).cpu().data.numpy().flatten()
    
    def train(self, replay_buffer, batch_size=100):
        self.total_it += 1
        
        # Replay Buffer에서 미니배치 샘플링
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        
        # ---------- Critic 업데이트 ----------
        # Target Policy Smoothing
        noise = torch.randn_like(action) * self.policy_noise
        noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
        
        next_action = self.actor_target(next_state) + noise
        next_action = torch.clamp(next_action, -self.max_action, self.max_action)
        
        # Clipped Double Q-Learning
        target_Q1, target_Q2 = self.critic_target(next_state, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + not_done * self.discount * target_Q
        
        # 현재 Q 값 계산
        current_Q1, current_Q2 = self.critic(state, action)
        
        # Critic 손실 함수 계산 및 업데이트
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ---------- Delayed Policy Updates ----------
        if self.total_it % self.policy_freq == 0:
            # Actor 업데이트
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Target Networks 업데이트
            with torch.no_grad():
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
```

## 8. TD3 성능 평가 및 비교

### TD3와 DDPG 성능 비교
- **평가 지표**: 에피소드 평균 보상, 학습 안정성, 데이터 효율성
- **성능 우위**: 대부분의 벤치마크에서 TD3가 DDPG보다 우수한 성능
- **학습 안정성**: TD3의 학습 곡선이 더 안정적이고 표준 편차가 낮음
- **데이터 효율성**: 같은 데이터로 TD3가 더 높은 성능 달성

### TD3 성능 개선의 주요 요인
1. **Clipped Double Q-Learning**: 과대평가 문제 완화로 더 정확한 Q 값 추정
2. **Delayed Policy Updates**: 안정적인 정책 학습으로 수렴 가능성 증가
3. **Target Policy Smoothing**: Q 함수의 평활화로 분산 감소 및 일반화 성능 향상

### 벤치마크 결과 (논문 기준)
- **MuJoCo 테스크**:
  - HalfCheetah-v1: TD3 > DDPG
  - Hopper-v1: TD3 > DDPG
  - Walker2d-v1: TD3 > DDPG
  - Ant-v1: TD3 > DDPG
  - Reacher-v1: TD3 > DDPG
  - InvertedPendulum-v1: 비슷함 (둘 다 최대 성능 도달)

## 요약

1. **TD3의 배경**:
   - DQN과 DDPG에서 발견된 Q 값 과대평가 문제 해결 목적
   - Q 값 과대평가는 강화학습의 성능을 저하시키는 중요한 요인

2. **TD3의 세 가지 핵심 아이디어**:
   - **Clipped Double Q-Learning**: 두 개의 Critic 네트워크를 사용하고 작은 Q 값을 선택하여 과대평가 완화
   - **Delayed Policy Updates**: Critic을 여러 번 업데이트한 후 Actor를 업데이트하여 학습 안정성 향상
   - **Target Policy Smoothing**: Target 정책에 노이즈를 추가하여 Q 함수의 평활화 및 분산 감소

3. **TD3의 구현**:
   - 두 개의 Critic 네트워크와 하나의 Actor 네트워크 구성
   - Target Networks의 Soft Update 유지
   - 각 핵심 기법을 코드 레벨에서 구현

4. **성능 비교**:
   - DDPG보다 대부분의 환경에서 우수한 성능
   - 특히 복잡하고 어려운 환경에서 성능 차이가 두드러짐
   - 학습 안정성과 데이터 효율성 측면에서 장점
