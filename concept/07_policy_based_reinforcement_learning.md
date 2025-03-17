# 정책 기반 강화학습(Policy-based Reinforcement Learning)

## 1. 정책 기반 강화학습 소개

### 정책 기반 강화학습의 개념
- **정의**: 가치 함수(Value Function) 대신 정책(Policy)을 직접 학습하는 방식
- **비유**: 선생님과 학생
  - **가치 기반**: 선생님(가치 함수)이 학생(정책)에게 어떻게 공부해야 할지(행동) 평가하고 지도
    - **단점**: 학생의 공부 방법에 대한 자유도가 낮음 (선생님의 지도 방식에 의존)
  - **정책 기반**: 학생이 스스로 학습 기준(정책)을 세우고 독학
    - **장점**: 학생의 자유도가 높음

### 가치 기반 강화학습과 비교
- **가치 기반 강화학습 (예: DQN)**:
  - 가치 함수(Q 함수)를 학습하고, 간접적으로 정책을 결정 (ε-greedy 등)
  - 단계: 가치 함수 학습 → 간접적 정책 결정
- **정책 기반 강화학습**:
  - 정책 자체를 직접 최적화
  - 단계: 직접 정책 학습 → 최적 정책

### 가치 기반 강화학습의 단점
- **연속적인 행동 공간(Continuous Action Space) 문제**:
  - **가치 기반**: 행동의 수가 많아지면 Q-table 또는 Q-network의 출력 차원이 커져 학습이 어려움
  - **정책 기반**: 정책을 확률 분포로 표현하므로 연속적인 행동 공간에 적용 가능
- **탐험 자유도**:
  - **가치 기반**: 결정적 정책(Deterministic Policy) 또는 ε-greedy 정책 사용. 탐험의 자유도에 한계
  - **정책 기반**: 확률적 정책(Stochastic Policy) 사용. 모든 행동에 대해 확률 값을 출력하므로 매 스텝마다 탐험 가능

## 2. 정책 기반 강화학습의 학습 기준: 리턴(Return)

### 리턴의 개념
- **정의**: 현재 시점부터 미래에 받을 보상의 총합
- **수식**: G_t = R_t + γR_(t+1) + γ²R_(t+2) + ... + γ^nR_(t+n)
- **의미**: 정책의 좋고 나쁨을 판단하는 기준

### 정책 표현 방식
- **파라미터화된 정책**: 정책을 인공 신경망과 같은 모델로 파라미터화 (파라미터: θ)
- **입력**: 상태(State)
- **출력**: 
  - **이산 행동 공간**: 각 행동에 대한 확률 분포
  - **연속 행동 공간**: 행동의 확률 분포 파라미터(예: 정규 분포의 평균과 표준편차)

### 정책의 확률적 특성
- **확률적 정책(Stochastic Policy)**:
  - 상태 s에서 행동 a를 선택할 확률을 출력: π_θ(a|s)
  - 무작위성을 통해 탐험 가능
- **결정적 정책(Deterministic Policy)**:
  - 상태 s에 대해 하나의 행동만을 출력: a = π_θ(s)
  - 가치 기반 방법에서 주로 사용

## 3. 정책 기반 강화학습의 최적 정책 학습 과정

### 데이터 수집
- **궤적(Trajectory, τ)**: 한 에피소드 내 모든 스텝의 상태, 행동, 보상의 sequence
  - τ = s₀, a₀, s₁, a₁, s₂, a₂, ..., s_T, a_T
- **가치 기반 vs. 정책 기반**:
  - **가치 기반**: 매 스텝마다 데이터 수집 (s_t, a_t, r_t, s_(t+1))
  - **정책 기반**: 에피소드가 종료된 후 리턴(G_t)을 계산하여 데이터로 사용(Monte Carlo 방식과 유사)

### 정책 가치 판단
- **목적 함수(Objective Function)**: 정책의 가치를 리턴의 기댓값으로 평가
- **수식**: J(θ) = E_[τ~p_θ(τ)][G₀]
  - θ: 정책 파라미터
  - τ: 궤적
  - p_θ(τ): 정책 π_θ에 의해 궤적 τ가 발생할 확률
  - G₀: 시작 시점부터의 리턴

### 정책 파라미터 업데이트
- **경사 상승법(Gradient Ascent)**: 목적 함수 J(θ)를 최대화하는 방향으로 파라미터 업데이트
- **수식**: θ_(k+1) ← θ_k + α ∇_θ J(θ)
  - α: 학습률(learning rate)
  - ∇_θ J(θ): 정책 그래디언트(Policy Gradient)

### 정책 기반 강화학습의 단점
- 정책의 분산이 큼
- 한 에피소드가 끝난 뒤에 정책의 학습을 할 수 있음
- 에피소드마다 return값이 매우 다르게 나오면, 학습의 변동이 매우 심해짐
- 대안: Actor-Critic 기반 강화학습

## 4. 목적 함수와 정책 그래디언트

### 목적 함수
- **정의**: 정책의 가치를 수치화한 함수, 최대화하려는 대상
- **수식**: J(θ) = E_[τ~p_θ(τ)][G₀] = ∫ p_θ(τ) ∑_(t=0)^T γ^t r(s_t, a_t) dτ
  - τ: 궤적
  - p_θ(τ): 정책 π_θ에 의해 궤적 τ가 발생할 확률
  - G₀: 시작 시점부터의 리턴
  - γ: 감가율
  - r(s_t, a_t): 상태 s_t에서 행동 a_t를 했을 때의 보상

### 정책 그래디언트
- **정의**: 목적 함수 J(θ)를 θ에 대해 미분한 값
- **의미**: 정책 파라미터 θ를 업데이트하는 방향을 제시
- **수식**: 
  - ∇_θ J(θ) = E_[τ~p_θ(τ)][∑_(t=0)^T (∇_θ logπ_θ(a_t|s_t) (∑_(k=t)^T γ^(k-t)r(s_k, a_k)))]
  - π_θ(a_t|s_t): 상태 s_t에서 정책 π_θ가 행동 a_t를 선택할 확률
  - ∇_θ logπ_θ(a_t|s_t): 로그 확률의 기울기(gradient of log probability)
  - (∑_(k=t)^T γ^(k-t)r(s_k, a_k)): 시간 t부터 에피소드 끝까지의 감가된 보상의 합(Return)

### 정책 그래디언트 수식 유도
- **목적 함수**: J(θ) = E_[τ~p_θ(τ)][G₀] = ∫ p_θ(τ) ∑_(t=0)^T γ^t r(s_t, a_t) dτ
- **미분**: ∇_θ J(θ) = ∇_θ ∫ p_θ(τ) ∑_(t=0)^T γ^t r(s_t, a_t) dτ
- **로그 트릭**: 
  - ∇_θ logp_θ(τ) = (∇_θ p_θ(τ)) / p_θ(τ) => ∇_θ p_θ(τ) = p_θ(τ) ∇_θ logp_θ(τ)
- **로그 확률 전개**: 
  - ∇_θ logp_θ(τ) = ∇_θ log(p(x₀) ∏_(t=0)^T π_θ(a_t|s_t)p(s_(t+1)|s_t, a_t)) = ∑_(t=0)^T ∇_θ logπ_θ(a_t|s_t)
- **최종 유도**:
  - ∇_θ J(θ) = ∫ p_θ(τ) (∑_(t=0)^T ∇_θ logπ_θ(a_t|s_t)) (∑_(t=0)^T γ^t r(s_t, a_t)) dτ
  - ∇_θ J(θ) = E_[τ~p_θ(τ)][(∑_(t=0)^T ∇_θ logπ_θ(a_t|s_t)) (∑_(t=0)^T γ^t r(s_t, a_t))]
  - **인과성 고려**: ∇_θ J(θ) = E_[τ~p_θ(τ)][∑_(t=0)^T (∇_θ logπ_θ(a_t|s_t) (∑_(k=t)^T γ^(k-t)r(s_k, a_k)))]

## 5. REINFORCE 알고리즘

### 개요
- **정의**: 정책 기반 강화학습 알고리즘 중 하나로, Monte Carlo 방법을 사용하여 정책 그래디언트를 추정하고, 경사 상승법을 통해 정책을 업데이트
- **동작 과정**:
  1. 정책 π_θ로부터 에피소드 데이터 생성(궤적 τ)
  2. 각 타임스텝 t에 대한 Return G_t 계산
  3. 정책 그래디언트 추정 및 정책 파라미터 업데이트

### 정책 그래디언트 추정 (Monte Carlo)
- **이론적 정책 그래디언트**:
  - ∇_θ J(θ) = E_[τ~p_θ(τ)][∑_(t=0)^T (∇_θ logπ_θ(a_t|s_t) (∑_(k=t)^T γ^(k-t)r(s_k, a_k)))]
  - 궤적 τ에 대한 기댓값이므로 직접 계산 불가능
- **Monte Carlo 추정**:
  1. 현재 정책 π_θ로 M개의 에피소드 샘플링
  2. 샘플링된 에피소드로 정책 그래디언트 추정:
     - ∇_θ J(θ) ≈ (1/M) ∑_(m=1)^M [∑_(t=0)^T (∇_θ logπ_θ(a_t^(m)|s_t^(m)) G_t^(m))]
     - G_t^(m): m번째 에피소드의 t 시점부터의 Return

### 손실 함수 (Loss Function)
- **목적 함수**: J(θ) = E_[τ~p_θ(τ)][G₀] (리턴의 기댓값)
- **손실 함수 변환**: 목적 함수에 음수 부호(-)를 붙여 최소화 문제로 변환
  - L(θ) = -J(θ) = - E_[τ~p_θ(τ)][G₀]
  - L(θ) = - ∑_(t=0)^T (logπ_θ(a_t^(m)|s_t^(m)) G_t^(m))

### 알고리즘 순서
1. **정책 실행**: 정책 π_θ로부터 에피소드 데이터 생성 (s₀, a₀, s₁, a₁, ..., s_T, a_T)
2. **Return 계산**: 각 타임스텝 t에 대한 Return G_t 계산 (G_t = ∑_(k=t)^T γ^(k-t)r(s_k, a_k))
3. **손실 함수 계산**: L(θ) = - ∑_(t=0)^T (logπ_θ(a_t|s_t) G_t)
4. **정책 파라미터 업데이트**: 경사 하강법을 사용하여 정책 파라미터 θ 업데이트 (θ ← θ - α ∇_θ L(θ))

### REINFORCE 알고리즘의 한계점
1. **긴 에피소드**: 에피소드가 끝나야 정책 업데이트 가능 (Monte Carlo 방식)
2. **높은 분산**: 정책 그래디언트의 분산이 큼 (Return 값의 변동이 심함)
3. **On-policy**: 데이터를 재사용할 수 없음 (정책 업데이트 시 해당 정책으로 생성된 샘플만 사용)

## 6. REINFORCE 알고리즘 PyTorch 구현

### 정책 모델 구현 (Policy 클래스)

```python
class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        # 공용 레이어
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        
        # 연속적인 행동 공간을 위한 정규 분포 파라미터
        self.mean_layer = nn.Linear(64, action_dim)
        self.log_std_layer = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean_layer(x)
        std = torch.exp(self.log_std_layer)
        return mean, std
```

### 행동 선택 구현

```python
def select_action(self, state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    mean, std = self.policy(state)
    
    # 정규 분포 생성
    m = Normal(mean, std)
    
    # 행동 샘플링
    action = m.sample()
    
    # 행동의 로그 확률 계산 및 저장
    self.save_log_probs.append(m.log_prob(action).sum())
    
    return action.detach().numpy()
```

### 손실 함수 및 파라미터 업데이트 구현

```python
def train(self):
    # Return 계산
    returns = []
    G = 0
    # 역순으로 Return 계산
    for r in self.rewards[::-1]:
        G = r + self.gamma * G
        returns.insert(0, G)
    
    # Return 정규화
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + self.eps)
    
    # 손실 함수 계산
    policy_loss = []
    for log_prob, G in zip(self.save_log_probs, returns):
        policy_loss.append(-log_prob * G)
    
    policy_loss = torch.cat(policy_loss).sum()
    
    # 정책 파라미터 업데이트
    self.optimizer.zero_grad()
    policy_loss.backward()
    self.optimizer.step()
    
    # 다음 에피소드를 위한 초기화
    del self.rewards[:]
    del self.save_log_probs[:]
```

### REINFORCE 알고리즘 실행 과정

```python
# 환경 및 에이전트 설정
env = gym.make("LunarLanderContinuous-v2")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
reinforce = REINFORCE(state_dim, action_dim, args.lr, args.gamma)

# 학습 루프
for episode in range(args.num_episodes):
    state = env.reset()
    for t in range(args.max_step):
        # 행동 선택 및 실행
        action = reinforce.select_action(state)
        next_state, reward, done, _ = env.step(action)
        
        # 보상 저장
        reinforce.rewards.append(reward)
        
        state = next_state
        if done:
            break
            
    # 정책 파라미터 업데이트
    reinforce.train()
```

## 요약

1. **정책 기반 강화학습**:
   - 가치 함수 대신 정책을 직접 학습하는 방식
   - 연속적인 행동 공간에 적합하며, 높은 탐험 자유도 제공

2. **리턴 개념**:
   - 현재부터 미래까지 받을 보상의 감가된 합
   - 정책의 가치를 평가하는 기준

3. **정책 그래디언트**:
   - 목적 함수(리턴의 기댓값)를 정책 파라미터에 대해 미분한 값
   - 경사 상승법을 통해 정책 파라미터 업데이트

4. **REINFORCE 알고리즘**:
   - Monte Carlo 방식으로 정책 그래디언트 추정
   - 에피소드 완료 후 리턴 계산 및 정책 업데이트
   - 정책 신경망 구조는 행동의 확률 분포(연속 행동의 경우 정규 분포의 평균과 표준편차)를 출력
