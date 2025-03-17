# Actor-Critic 방법(Actor-Critic Methods)

## 1. Actor-Critic 알고리즘 등장 배경

### REINFORCE 알고리즘의 문제점
- **높은 분산**: 목적 함수 그래디언트의 분산이 커서 학습이 불안정
  - 목적 함수 그래디언트: ∇_θ J(θ) = E[∇_θ logπ_θ(a_t|s_t) G_t]
  - G_t: Return 값(에피소드 종료까지의 감가된 보상 합)으로 분산이 큼
- **예시**: 에피소드 길이가 100이고 각 타임스텝 보상이 +1 또는 -1인 경우
  - G_t의 범위: -100 ~ +100 (분산이 매우 큼)
  - 파라미터 θ의 업데이트가 불안정해짐

### 가치 기반 방법과 정책 기반 방법의 장단점
- **가치 기반 방법(Value-based)**:
  - **장점**: TD(Temporal Difference) 학습으로 매 타임스텝 업데이트 가능
  - **단점**: 이산적인 행동 공간에만 적용 가능
- **정책 기반 방법(Policy-based)**:
  - **장점**: 연속적인 행동 공간에도 적용 가능
  - **단점**: Monte Carlo 방법을 사용하여 에피소드 종료 후 학습(분산 큼)

### 필요성
- 연속적인 행동 공간에 적용 가능하면서(정책 기반), 분산이 작고 안정적인 학습이 가능한(가치 기반) 알고리즘 필요
- 두 방법의 장점을 결합: 정책 기반 학습 + TD 방법 = Actor-Critic 알고리즘

## 2. Actor-Critic 알고리즘의 특징

### 기본 구조
- **Actor (정책 신경망, Policy Network)**:
  - 정책(Policy)을 나타내는 신경망
  - 상태(State)를 입력받아 행동(Action)의 확률 분포를 출력
  - **목적**: 더 높은 Q 값을 받도록 행동 수정 (정책 개선)
  - **파라미터**: θ
- **Critic (가치 신경망, Value Network)**:
  - 가치 함수(Value Function, 주로 Q 함수 또는 V 함수)를 나타내는 신경망
  - 상태(State)와 행동(Action)을 입력받아 Q 값(또는 상태 가치 V)을 출력
  - **목적**: Actor의 행동을 평가 (Q 값 계산)
  - **파라미터**: ω

### 학습 방법
- **Actor**: 정책 기반 학습 (Policy Gradient)
  - **목적 함수**: J(θ) = E[logπ_θ(a_t|s_t) Q_ω(s_t, a_t)]
    - θ: Actor 네트워크의 파라미터
    - ω: Critic 네트워크의 파라미터
    - π_θ(a_t|s_t): 상태 s_t에서 정책 π_θ가 행동 a_t를 선택할 확률
    - Q_ω(s_t, a_t): 상태 s_t에서 행동 a_t를 했을 때의 Q 값 (Critic이 평가)
  - 경사 상승법으로 목적 함수 최대화: θ ← θ + α ∇_θ J(θ)
- **Critic**: 가치 기반 학습 (TD 방법)
  - **손실 함수**: L(ω) = (1/2)(y_t - Q_ω(s_t, a_t))²
    - y_t: TD Target (예: r_t + γV(s_(t+1)))
  - 경사 하강법으로 손실 함수 최소화: ω ← ω - β ∇_ω L(ω)

### REINFORCE vs. Actor-Critic 비교
- **REINFORCE**: J(θ) = E[logπ_θ(a_t|s_t) G_t] (Return 사용)
- **Actor-Critic**: J(θ) = E[logπ_θ(a_t|s_t) Q_ω(s_t, a_t)] (Q 값 사용)
- **장점**: Actor-Critic은 REINFORCE보다 분산이 작음 (Return 대신 Q 값을 사용하므로)
- **특징**: TD 방법을 통해 매 타임스텝마다 업데이트 가능 (Monte Carlo 방법보다 효율적)

## 3. Actor-Critic 목적 함수 유도 (REINFORCE → Actor-Critic)

### REINFORCE 목적 함수 (시작점)
- **REINFORCE 목적 함수 그래디언트**:
  - ∇_θ J(θ) = E[∑_(t=0)^T ∇_θ logπ_θ(a_t|s_t) G_t]
  - G_t: t 시점부터의 Return. 분산이 큼

### 유도 과정
1. **시작**: REINFORCE 목적 함수의 그래디언트에서 출발
   - ∇_θ J(θ) = E[∑_(t=0)^T ∇_θ logπ_θ(a_t|s_t) G_t]

2. **기댓값 표현**: 궤적(trajectory)에 대한 적분 형태로 표현
   - ∇_θ J(θ) = ∑_(t=0)^T ∫_(τ_(s₀:a_T)) ∇_θ logπ_θ(a_t|s_t) G_t p_θ(τ_(s₀:a_T)) dτ_(s₀:a_T)
   - τ_(s₀:a_T): s₀부터 a_T까지의 궤적

3. **조건부 확률의 연쇄 법칙 적용**:
   - 궤적을 현재 스텝까지의 궤적(τ_(s₀:a_t))과 이후 스텝의 궤적(τ_(s_(t+1):a_T))으로 분리
   - ∇_θ J(θ) = ∑_(t=0)^T ∫_(τ_(s₀:a_t)) ∫_(τ_(s_(t+1):a_T)) ∇_θ logπ_θ(a_t|s_t) G_t p_θ(τ_(s_(t+1):a_T)|τ_(s₀:a_t)) p_θ(τ_(s₀:a_t)) dτ_(s_(t+1):a_T) dτ_(s₀:a_t)

4. **MDP 특징 활용**: 미래는 과거와 독립적
   - p(s_(t+1)|s₀, a₀, ..., s_t, a_t) = p(s_(t+1)|s_t, a_t)
   - ∇_θ J(θ) = ∑_(t=0)^T ∫_(τ_(s₀:a_t)) ∫_(τ_(s_(t+1):a_T)) ∇_θ logπ_θ(a_t|s_t) G_t p_θ(τ_(s_(t+1):a_T)|s_t, a_t) p_θ(τ_(s₀:a_t)) dτ_(s₀:a_t)

5. **Q 함수 도입**:
   - ∫_(τ_(s_(t+1):a_T)) G_t p_θ(τ_(s_(t+1):a_T)|s_t, a_t) dτ_(s_(t+1):a_T) = Q(s_t, a_t)
   - Q(s_t, a_t): 상태 s_t와 행동 a_t가 주어졌을 때 기대되는 Return
   - ∇_θ J(θ) = ∑_(t=0)^T ∫_(τ_(s₀:a_t)) ∇_θ logπ_θ(a_t|s_t) Q(s_t, a_t) p_θ(τ_(s₀:a_t)) dτ_(s₀:a_t)

6. **적분 간소화**:
   - ∫_(τ_(s₀:a_t)) ... p_θ(τ_(s₀:a_t)) dτ_(s₀:a_t) = E[...]
   - ∇_θ J(θ) = ∑_(t=0)^T E[∇_θ logπ_θ(a_t|s_t) Q(s_t, a_t)]

### 최종 결과
- **Actor-Critic 목적 함수 그래디언트**:
  - ∇_θ J(θ) = ∑_(t=0)^T E[∇_θ logπ_θ(a_t|s_t) Q_ω(s_t, a_t)]
  - REINFORCE 목적 함수의 그래디언트에서 G_t를 Q_ω(s_t, a_t)로 대체

## 4. A2C (Advantage Actor-Critic) 알고리즘

### Actor-Critic의 문제점
- Q 값도 리턴의 기댓값이므로, 여전히 분산이 클 수 있음
- 추가적인 분산 감소 기법이 필요

### Advantage 함수
- **정의**: A(s, a) = Q(s, a) - V(s)
  - 특정 행동 a가 평균적인 행동보다 얼마나 더 좋은지를 나타내는 상대적인 값
  - Q(s, a): 상태 s에서 행동 a를 했을 때의 기대 보상
  - V(s): 상태 s에서 정책에 따라 행동했을 때의 기대 보상(평균적인 행동 가치)
- **특징**: Q 값에서 기저값(baseline)인 상태 가치 V를 빼서 분산을 줄임
- **예시**:
  - Q 값: 12, 23, 31
  - V 값: 10, 20, 30
  - Advantage: 2, 3, 1 (Q 값보다 범위가 작아 분산 감소)

### A2C 목적 함수 유도
- **Actor-Critic 목적 함수**: ∇_θ J(θ) = ∑_(t=0)^T E[∇_θ logπ_θ(a_t|s_t) Q_ω(s_t, a_t)]
- **아이디어**: Q_ω(s_t, a_t)에서 기저값(baseline) b(s_t)를 빼도 기댓값은 변하지 않음
  - ∑_(a) π_θ(a|s) ∇_θ logπ_θ(a|s) = ∇_θ ∑_(a) π_θ(a|s) = ∇_θ 1 = 0
  - ∴ ∑_(t=0)^T E[∇_θ logπ_θ(a_t|s_t) b(s_t)] = 0
- **A2C 목적 함수**: 
  - ∇_θ J(θ) = ∑_(t=0)^T E[∇_θ logπ_θ(a_t|s_t) A_ω(s_t, a_t)]
  - A_ω(s_t, a_t) = Q_ω(s_t, a_t) - V_ω(s_t) (일반적으로 b(s_t) = V_ω(s_t))

### A2C 알고리즘의 학습 과정
1. **초기화**: Actor 네트워크 파라미터 θ, Critic 네트워크 파라미터 ω 초기화
2. **데이터 수집**: 정책 π_θ를 사용하여 N 타임스텝 동안 데이터 수집
   - 데이터: (s_t, a_t, r_t, s_(t+1))
   - 버퍼(Rollout Buffer)에 저장
3. **데이터 추출**: 버퍼에서 데이터 추출
4. **Advantage 계산**:
   - TD Target: y_t = r_t + γV_ω(s_(t+1))
   - Advantage: A_ω(s_t, a_t) = y_t - V_ω(s_t)
5. **Actor 네트워크 목적 함수 계산**:
   - J(θ) = (1/N) ∑_i logπ_θ(a_i|s_i) A_ω(s_i, a_i)
6. **Critic 네트워크 손실 함수 계산**:
   - L(ω) = (1/2) ∑_i (y_i - V_ω(s_i))²
7. **네트워크 업데이트**:
   - Actor: 경사 상승법 (θ ← θ + α ∇_θ J(θ))
   - Critic: 경사 하강법 (ω ← ω - β ∇_ω L(ω))
8. **반복**: 2-7 과정 반복

## 5. A2C 알고리즘 구현

### 아키텍처 및 클래스 구조
- **RolloutBuffer**: N 타임스텝 동안의 데이터를 저장하는 버퍼
  - 데이터: 상태, 행동, 보상, 종료 여부, 가치, 로그 확률 등
  - 메소드: add, get, compute_returns_and_advantage 등
- **Policy**: Actor와 Critic 네트워크를 포함하는 정책 클래스
  - Actor: 상태를 입력받아 행동의 확률 분포 출력
  - Critic: 상태를 입력받아 가치(V 값) 출력
- **A2C**: 알고리즘 클래스
  - 메소드: learn, collect_rollouts, train 등

### 데이터 수집 과정
```python
# 데이터 수집
rollout_buffer = RolloutBuffer(...)  # RolloutBuffer 생성

for step in range(n_steps):  # n_steps는 rollout buffer의 크기
    with torch.no_grad():
        # policy를 사용하여 행동, 가치, 로그 확률 계산
        actions, values, log_prob = self.policy.forward(obs)
    actions = actions.cpu().numpy()

    # 환경과 상호작용
    new_obs, rewards, dones, infos = env.step(actions)

    # 데이터 저장
    rollout_buffer.add(obs, actions, rewards, dones, values, log_prob)
    obs = new_obs
```

### Actor와 Critic 네트워크 학습
```python
# Actor 네트워크 목적 함수 계산
_, log_prob, _ = self.policy.evaluate_actions(rollout_data.observations, rollout_data.actions)
advantages = rollout_data.advantages
policy_loss = -(log_prob * advantages).mean()

# Critic 네트워크 손실 함수 계산
values = self.policy.predict_values(rollout_data.observations)
value_loss = F.mse_loss(rollout_data.returns, values)

# 네트워크 업데이트
loss = policy_loss + self.vf_coef * value_loss
self.policy.optimizer.zero_grad()
loss.backward()
self.policy.optimizer.step()
```

### Advantage 계산
```python
# Advantage 계산
advantages = np.zeros_like(rewards)
last_value = values[-1]
last_advantage = 0

for step in reversed(range(len(rewards))):
    # 마지막 스텝인 경우 다음 가치로 last_value 사용
    if step == len(rewards) - 1:
        next_non_terminal = 1.0 - dones[step]
        next_value = last_value
    else:
        next_non_terminal = 1.0 - dones[step]
        next_value = values[step + 1]
    
    # TD Target 계산: r_t + γV(s_(t+1))
    delta = rewards[step] + gamma * next_value * next_non_terminal - values[step]
    
    # Advantage 계산: A(s_t, a_t) = δ_t + γλA(s_(t+1), a_(t+1))
    advantages[step] = delta + gamma * self.gae_lambda * next_non_terminal * last_advantage
    last_advantage = advantages[step]

# Return 계산: V(s_t) + A(s_t, a_t)
returns = advantages + values
```

## 6. A2C의 변형 및 확장

### N-step A2C
- **아이디어**: TD Target을 n-step으로 확장
- **Advantage 계산**:
  - TD Target: y_t = r_t + γr_(t+1) + ... + γ^(n-1)r_(t+n-1) + γ^n V_ω(s_(t+n))
  - Advantage: A_ω(s_t, a_t) = y_t - V_ω(s_t)
- **장점**: 편향과 분산 사이의 균형을 조절 가능(n이 클수록 분산 증가, 편향 감소)

### A3C (Asynchronous Advantage Actor-Critic)
- **핵심**: 여러 에이전트를 병렬로 실행하여 데이터 수집 및 학습 속도 향상
- **작동 방식**:
  - 각 에이전트는 환경의 복사본에서 독립적으로 데이터 수집
  - 수집된 데이터로 비동기적으로(asynchronously) 네트워크 업데이트
  - 글로벌 네트워크와 로컬 네트워크 간 파라미터 동기화
- **장점**: 다양한 환경에서의 경험으로 일반화 성능 향상, 병렬 처리로 학습 속도 향상

### PPO (Proximal Policy Optimization)
- **핵심**: Actor 업데이트 시 정책 변화를 제한하여 안정적인 학습
- **목적 함수**:
  - J^{PPO}(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
  - r_t(θ): 현재 정책과 이전 정책의 확률 비율(π_θ(a_t|s_t) / π_θ_old(a_t|s_t))
- **장점**: 안정적인 학습, 단순하지만 효과적인 알고리즘

## 요약

1. **Actor-Critic 알고리즘 개요**:
   - 정책 기반 방법(Actor)과 가치 기반 방법(Critic)을 결합한 알고리즘
   - REINFORCE의 높은 분산 문제를 해결하기 위해 등장

2. **구조**:
   - **Actor**: 정책을 나타내는 신경망, 상태를 입력받아 행동의 확률 분포 출력
   - **Critic**: 가치 함수를 나타내는 신경망, 상태를 입력받아 가치(V 또는 Q 값) 출력

3. **A2C (Advantage Actor-Critic)**:
   - Actor-Critic에서 Advantage 함수를 사용하여 분산을 더욱 줄인 알고리즘
   - Advantage 함수: A(s, a) = Q(s, a) - V(s)

4. **목적 함수 유도**:
   - REINFORCE: J(θ) = E[logπ_θ(a_t|s_t) G_t]
   - Actor-Critic: J(θ) = E[logπ_θ(a_t|s_t) Q_ω(s_t, a_t)]
   - A2C: J(θ) = E[logπ_θ(a_t|s_t) A_ω(s_t, a_t)]

5. **장점**:
   - REINFORCE보다 분산이 작아 안정적인 학습
   - TD 방법을 통해 매 타임스텝마다 업데이트 가능
   - 연속적인 행동 공간에도 적용 가능

6. **확장**:
   - A3C: 병렬 처리를 통한 학습 속도 향상
   - PPO: 정책 변화 제한을 통한 안정적 학습
