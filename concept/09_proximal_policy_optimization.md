# 근사 정책 최적화(Proximal Policy Optimization, PPO)

## 1. PPO 알고리즘 등장 배경

### 기존 알고리즘의 문제점
- **Advantage Actor-Critic (A2C)의 문제점**:
  - 데이터 효율성이 낮음: 한 번 업데이트에 사용한 데이터를 재사용할 수 없음 (On-policy 계열 알고리즘의 공통적인 문제)
  - 정책 업데이트가 불안정할 수 있음: 업데이트 크기가 너무 큰 경우 성능 저하 가능
- **개선 방향**:
  - 데이터 효율성 증가: 같은 데이터를 여러 번 재사용하여 학습 효율 향상
  - Step size 증가: 더 큰 업데이트 스텝으로 학습 속도 향상
  - 안정성 확보: 정책 업데이트 크기를 제한하여 성능 저하 방지

### TRPO (Trust Region Policy Optimization)
- **Trust Region**: 성능이 떨어지지 않을 것이라고 신뢰할 수 있는 업데이트 범위
- **아이디어**: 정책 업데이트 범위를 Trust Region으로 제한하여 안정적인 학습 보장
- **제약 조건**: KL Divergence를 통해 정책 변화 제한
  - KL(π_old || π_new) ≤ δ (δ는 작은 값)
- **단점**: 수학적으로 복잡하고 구현이 어려움, 다른 딥러닝 기법과 결합하기 어려움

### PPO의 필요성
- TRPO의 복잡성 해결: 더 간단한 방법으로 비슷한 성능 달성
- 데이터 효율성 향상: On-policy 알고리즘이면서도 데이터를 재사용 가능
- 안정적인 학습: 정책 업데이트 제한을 통한 안정성 확보

## 2. PPO 알고리즘의 특징

### 개요
- **정의**: Proximal Policy Optimization, 정책 그래디언트 기반의 강화학습 알고리즘
- **핵심 아이디어**: Surrogate Objective (대리 목적 함수)와 Clip 연산을 통한 안정적 학습
- **특징**:
  - On-policy 알고리즘이지만 데이터 재사용 가능
  - 구현이 간단하면서도 좋은 성능
  - 하이퍼파라미터에 덜 민감함

### Surrogate Objective
- **기존 목적 함수 (Actor-Critic)**: J(θ) = E[logπ_θ(a_t|s_t) A_t]
- **Surrogate Objective**: 
  - J^SURR(θ) = E[(π_θ(a_t|s_t) / π_θ_old(a_t|s_t)) A_t]
  - Importance Sampling 기법을 활용하여 과거 정책으로 생성된 데이터 활용
- **문제점**: 정책 비율(π_θ/π_θ_old)이 제한 없이 커질 수 있어 과도한 업데이트 위험

### Clipped Surrogate Objective
- **해결 방법**: Clip 연산을 통해 정책 비율 제한
- **Clipped Surrogate Objective**:
  - L^CLIP(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
  - r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
  - ε: 하이퍼파라미터 (보통 0.2)

### Clip 연산과 min 연산 분석
- **Clip 연산**:
  - clip(x, min, max): x 값을 min과 max 사이로 제한
  - r_t(θ)를 1-ε과 1+ε 사이로 제한하여 정책 변화 폭 제한
- **min 연산**:
  - min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t): 두 항 중 작은 값 선택
  - Advantage가 양수일 때(A_t > 0): 정책이 너무 많이 변하는 것을 제한
  - Advantage가 음수일 때(A_t < 0): 정책이 너무 적게 변하는 것을 제한

## 3. PPO 목적 함수 유도 (A2C → PPO)

### A2C 목적 함수 (시작점)
- **A2C 목적 함수**: ∇_θ J(θ) = E[∇_θ logπ_θ(a_t|s_t) A_t]
  - A_t: Advantage (현재 행동이 평균보다 얼마나 더 좋은지)

### Surrogate Objective 유도 과정
1. **기댓값을 적분 형태로 표현**:
   - ∇_θ J(θ) = E[∇_θ logπ_θ(a_t|s_t) A_t]
   - = ∫_(s_(t+1), a_t, s_t) ∇_θ logπ_θ(a_t|s_t) A_t p_θ(s_t, a_t) p(s_(t+1)|s_t, a_t) ds_t da_t ds_(t+1)

2. **Importance Sampling 적용**:
   - 목적: 현재 정책 π_θ가 아닌 과거 정책 π_θ_old로 생성된 데이터 사용
   - p_θ(s_t, a_t) / p_θ_old(s_t, a_t)를 곱하고 나눔
   - ∇_θ J(θ) = ∫ ∇_θ logπ_θ(a_t|s_t) A_t (p_θ(s_t, a_t) / p_θ_old(s_t, a_t)) p_θ_old(s_t, a_t) p(s_(t+1)|s_t, a_t) ds_t da_t ds_(t+1)

3. **로그 미분 트릭 적용**:
   - ∇_θ logπ_θ(a_t|s_t) = (∇_θ π_θ(a_t|s_t)) / π_θ(a_t|s_t)
   - ∇_θ J(θ) = ∫ (∇_θ π_θ(a_t|s_t) / π_θ(a_t|s_t)) A_t (p_θ(s_t, a_t) / p_θ_old(s_t, a_t)) p_θ_old(s_t, a_t) p(s_(t+1)|s_t, a_t) ds_t da_t ds_(t+1)

4. **결합 확률 분포 분리**:
   - p_θ(s_t, a_t) = p_θ(s_t)π_θ(a_t|s_t)
   - p_θ_old(s_t, a_t) = p_θ_old(s_t)π_θ_old(a_t|s_t)
   - ∇_θ J(θ) = ∫ (∇_θ π_θ(a_t|s_t) / π_θ(a_t|s_t)) A_t (p_θ(s_t)π_θ(a_t|s_t) / (p_θ_old(s_t)π_θ_old(a_t|s_t))) p_θ_old(s_t, a_t) p(s_(t+1)|s_t, a_t) ds_t da_t ds_(t+1)

5. **상태 분포 근사**: p_θ(s_t)와 p_θ_old(s_t)가 충분히 비슷하다고 가정하고 약분
   - ∇_θ J(θ) = ∫ (∇_θ π_θ(a_t|s_t) / π_θ(a_t|s_t)) A_t (π_θ(a_t|s_t) / π_θ_old(a_t|s_t)) p_θ_old(s_t, a_t) p(s_(t+1)|s_t, a_t) ds_t da_t ds_(t+1)

6. **정리**:
   - ∇_θ J(θ) = ∫ ∇_θ π_θ(a_t|s_t) A_t (1 / π_θ_old(a_t|s_t)) p_θ_old(s_t, a_t) p(s_(t+1)|s_t, a_t) ds_t da_t ds_(t+1)
   - ∇_θ J(θ) = E[(∇_θ π_θ(a_t|s_t) / π_θ_old(a_t|s_t)) A_t]

7. **최종 Surrogate Objective**:
   - ∇_θ J(θ) ≈ E[(π_θ(a_t|s_t) / π_θ_old(a_t|s_t)) ∇_θ logπ_θ(a_t|s_t) A_t]
   - ∇_θ J(θ) ≈ E[r_t(θ) ∇_θ logπ_θ(a_t|s_t) A_t] (여기서 r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t))

### 제약 조건
- **필요 조건**: p_θ(s_t)와 p_θ_old(s_t)가 비슷해야 함 (정책 변화가 크지 않아야 함)
- **KL Divergence**: 두 확률 분포의 차이를 측정하는 함수
  - D_KL(π_θ_old(·|s_t) || π_θ(·|s_t)) ≤ δ (δ는 작은 값)
- **PPO**: Clip 연산을 통해 정책 변화 제한

## 4. Advantage 계산 - GAE

### Advantage 계산의 중요성
- **역할**: 행동의 상대적인 가치 평가 (평균보다 얼마나 더 좋은지)
- **분산 감소**: 정책 그래디언트의 분산을 줄여 안정적인 학습 가능

### 기존 Advantage 계산 방법들
- **1-step TD**:
  - A_t ≈ r_t + γV(s_(t+1)) - V(s_t) = δ_t (TD Error)
  - 특징: 편향(bias)이 큼, 분산(variance)이 작음
- **Monte Carlo**:
  - A_t = R_t - V(s_t) = (r_t + γr_(t+1) + ... + γ^(T-t)r_T) - V(s_t)
  - 특징: 편향이 작음, 분산이 큼

### GAE (Generalized Advantage Estimation)
- **정의**: 1-step TD와 Monte Carlo 방법의 장점을 결합한 Advantage 추정 방법
- **수식**:
  - A_t^GAE(γ,λ) = ∑_(l=0)^∞ (γλ)^l δ_(t+l)
  - δ_t = r_t + γV(s_(t+1)) - V(s_t) (TD Error)
- **파라미터**:
  - γ: 감가율 (할인율)
  - λ: GAE 파라미터 (0과 1 사이의 값)
    - λ = 0: 1-step TD (편향↑, 분산↓)
    - λ = 1: Monte Carlo (편향↓, 분산↑)
    - 일반적으로 λ = 0.95 정도 사용
- **구현 방법**:
  ```python
  # Advantage 계산 (GAE)
  advantages = np.zeros_like(rewards)
  last_gae_lam = 0

  for step in reversed(range(len(rewards))):
      if step == len(rewards) - 1:
          next_non_terminal = 1.0 - dones[step]
          next_value = last_values
      else:
          next_non_terminal = 1.0 - dones[step]
          next_value = values[step + 1]
          
      delta = rewards[step] + gamma * next_value * next_non_terminal - values[step]
      advantages[step] = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
      last_gae_lam = advantages[step]
  
  # Returns = Advantage + Value
  returns = advantages + values
  ```

## 5. PPO 알고리즘 학습 과정

### 전체 학습 과정 개요
1. **초기화**: Actor 네트워크(Policy)와 Critic 네트워크(Value Function) 초기화
2. **데이터 수집**: 현재 정책을 사용하여 환경과 상호작용하며 데이터 수집
3. **Advantage 계산**: GAE를 사용하여 Advantage 계산
4. **미니배치 학습**: 수집된 데이터에서 미니배치를 샘플링하여 여러 번 학습
5. **정책 업데이트**: Clipped Surrogate Objective를 사용하여 Actor 업데이트
6. **가치 함수 업데이트**: MSE를 사용하여 Critic 업데이트
7. **반복**: 2-6 과정 반복

### 상세 학습 단계
1. **초기화**:
   - Actor 네트워크 파라미터 θ 초기화
   - Critic 네트워크 파라미터 ω 초기화

2. **데이터 수집**:
   - 현재 정책 π_θ_old로 N 타임스텝 동안 환경과 상호작용
   - 데이터 (s_i, a_i, r_i, s_(i+1), π_θ_old(a_i|s_i)) 수집, Rollout Buffer에 저장

3. **Advantage 계산**:
   - GAE를 사용하여 각 타임스텝의 Advantage A_i 계산
   - Returns = Advantage + Value Function 계산 (Critic 네트워크 학습에 사용)

4. **미니배치 학습**:
   - Rollout Buffer에서 미니배치 크기(M)만큼 데이터 샘플링
   - 같은 데이터로 K번 반복 학습 (보통 K=10)

5. **Critic 네트워크 업데이트**:
   - 손실 함수: L(ω) = (1/2M) ∑_i (Returns_i - V_ω(s_i))²
   - 경사 하강법으로 Critic 네트워크 파라미터 ω 업데이트

6. **Actor 네트워크 업데이트**:
   - 정책 비율: r_i(θ) = π_θ(a_i|s_i) / π_θ_old(a_i|s_i)
   - Clipped Surrogate Objective:
     - L^CLIP(θ) = (1/M) ∑_i min(r_i(θ)A_i, clip(r_i(θ), 1-ε, 1+ε)A_i)
   - 경사 상승법으로 Actor 네트워크 파라미터 θ 업데이트

7. **정책 갱신**:
   - 학습 후 π_θ_old ← π_θ (다음 데이터 수집에 사용할 정책 업데이트)

## 6. PPO 알고리즘 구현

### Actor-Critic 네트워크 구조
```python
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # 공유 레이어
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        
        # Actor 네트워크 (정책)
        self.actor_mean = nn.Linear(64, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        
        # Critic 네트워크 (가치 함수)
        self.critic = nn.Linear(64, 1)
        
    def forward(self, x):
        shared_features = self.shared(x)
        
        # Actor (정책) 출력
        action_mean = self.actor_mean(shared_features)
        action_std = torch.exp(self.actor_logstd)
        
        # Critic (가치 함수) 출력
        value = self.critic(shared_features)
        
        return action_mean, action_std, value
    
    def get_action(self, state, deterministic=False):
        action_mean, action_std, _ = self.forward(state)
        
        if deterministic:
            return action_mean
        else:
            distribution = Normal(action_mean, action_std)
            action = distribution.sample()
            log_prob = distribution.log_prob(action).sum(axis=-1)
            return action, log_prob
    
    def evaluate_actions(self, states, actions):
        action_mean, action_std, value = self.forward(states)
        distribution = Normal(action_mean, action_std)
        log_prob = distribution.log_prob(actions).sum(axis=-1)
        entropy = distribution.entropy().sum(axis=-1)
        return value, log_prob, entropy
```

### PPO 클립된 Surrogate Objective 구현
```python
def compute_policy_loss(self, obs, actions, old_log_prob, advantages):
    # 현재 정책에서의 로그 확률 계산
    _, log_prob, _ = self.policy.evaluate_actions(obs, actions)
    
    # 정책 비율 계산: r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
    ratio = torch.exp(log_prob - old_log_prob)
    
    # 클립된 Surrogate Objective 계산
    clip_adv = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
    policy_loss = -torch.min(ratio * advantages, clip_adv).mean()
    
    return policy_loss
```

### Critic 네트워크 손실 함수 구현
```python
def compute_value_loss(self, obs, returns):
    # Critic 네트워크를 통해 가치 함수 예측
    values = self.policy.predict_values(obs)
    
    # MSE 손실 계산
    value_loss = F.mse_loss(returns, values)
    
    return value_loss
```

### PPO 학습 루프 구현
```python
def learn(self, total_timesteps):
    # 주요 파라미터 설정
    n_steps = self.n_steps  # 데이터 수집 단계 수
    n_epochs = self.n_epochs  # 데이터 재사용 횟수
    batch_size = self.batch_size  # 미니배치 크기
    
    for iteration in range(total_timesteps // n_steps):
        # 1. 데이터 수집
        rollout_data = self.collect_rollouts()
        
        # 2. Advantage 및 Returns 계산
        advantages, returns = self.compute_gae(rollout_data)
        
        # 3. 여러 에포크 동안 미니배치로 학습
        for epoch in range(n_epochs):
            # 데이터셋에서 미니배치 샘플링
            for batch_idx in range(n_steps // batch_size):
                # 미니배치 인덱스 추출
                idx = np.random.randint(0, n_steps, batch_size)
                
                # 미니배치 데이터
                batch_obs = rollout_data.observations[idx]
                batch_actions = rollout_data.actions[idx]
                batch_old_log_prob = rollout_data.log_probs[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]
                
                # Actor 손실 계산
                policy_loss = self.compute_policy_loss(
                    batch_obs, batch_actions, batch_old_log_prob, batch_advantages)
                
                # Critic 손실 계산
                value_loss = self.compute_value_loss(batch_obs, batch_returns)
                
                # 전체 손실 계산
                loss = policy_loss + self.vf_coef * value_loss
                
                # 네트워크 업데이트
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
```

## 7. PPO와 다른 알고리즘 비교

### PPO vs TRPO
- **유사점**:
  - 정책 업데이트 제한을 통한 안정적 학습
  - On-policy 계열 알고리즘
- **차이점**:
  - TRPO: KL Divergence 제약 조건을 통한 정책 업데이트 제한 (복잡)
  - PPO: Clip 연산을 통한 정책 업데이트 제한 (간단)

### PPO vs A2C/A3C
- **유사점**:
  - Advantage를 사용한 정책 그래디언트
  - Actor-Critic 아키텍처 사용
- **차이점**:
  - A2C/A3C: 각 업데이트마다 데이터를 한 번만 사용 (데이터 효율↓)
  - PPO: 같은 데이터를 여러 번 재사용 (데이터 효율↑)

### PPO의 장단점

#### 장점
- **간단한 구현**: TRPO에 비해 구현이 훨씬 간단함
- **좋은 성능**: 많은 벤치마크에서 좋은 성능을 보임
- **안정적인 학습**: Clip 연산을 통한 학습 안정성 확보
- **데이터 효율성**: 데이터 재사용을 통한 샘플 효율성 향상
- **하이퍼파라미터 민감도 낮음**: 대부분의 환경에서 기본 하이퍼파라미터로 잘 작동

#### 단점
- **On-policy의 한계**: 완전한 Off-policy 알고리즘만큼의 데이터 효율성은 없음
- **복잡한 튜닝**: 여러 하이퍼파라미터(클립 범위, 학습률 등)를 튜닝해야 할 수 있음
- **계산 비용**: 여러 에포크에 걸쳐 데이터를 재사용하므로 계산량 증가

## 요약

1. **PPO 등장 배경**:
   - A2C의 데이터 효율성 문제
   - TRPO의 복잡성 문제
   - 안정적이면서도 간단한 알고리즘의 필요성

2. **PPO의 핵심 아이디어**:
   - Surrogate Objective: 과거 정책으로 수집한 데이터를 현재 정책 학습에 활용
   - Clipped Surrogate Objective: 정책 변화 제한을 통한 안정적 학습
   - 데이터 재사용: 같은 데이터를 여러 번 재사용하여 데이터 효율성 향상

3. **PPO 목적 함수**:
   - L^CLIP(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
   - r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
   - Clip 연산과 min 연산을 통해 정책 업데이트 제한

4. **Advantage 계산**:
   - GAE(Generalized Advantage Estimation) 사용
   - 1-step TD와 Monte Carlo의 장점 결합 (편향-분산 트레이드오프 조절)

5. **PPO의 성공 요인**:
   - 구현의 간결함
   - 안정적인 학습 성능
   - 다양한 문제에 적용 가능한 범용성
