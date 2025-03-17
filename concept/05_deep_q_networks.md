# 심층 Q 네트워크(Deep Q-Networks, DQN)

## 1. DQN 개요

### 고차원 문제와 기존 강화학습의 한계
- **고차원 데이터 처리의 어려움**: 게임 화면(이미지)과 같은 고차원 데이터를 상태 입력으로 사용하면 연산량이 많고 학습 속도가 느림
- **Q-table의 한계**:
  - 가능한 상태(화면)의 수가 매우 많아 Q-table이 무한히 커짐
  - 새로운 상태가 나타날 때마다 Q-table을 확장해야 함
  - 각각의 Q 값이 거의 업데이트되지 않아 학습이 어려움

### DQN(Deep Q-Network)이란?
- **정의**: 기존 강화 학습 알고리즘(Q-learning)에 딥러닝을 적용한 알고리즘
- **핵심**: Q-table을 딥러닝(심층 신경망, Deep Neural Network)으로 대체
  - **Q-learning**: Q-table을 사용하여 상태-행동 쌍에 대한 Q 값을 저장하고 업데이트
  - **DQN**: 심층 신경망을 사용하여 Q 값을 추정. 상태(s)와 행동(a)을 입력으로 받아 Q(s, a)를 출력
- **장점**:
  - 고차원 데이터(이미지, 연속적인 상태 공간 등) 처리 가능
  - Q-table 방식에 비해 일반화(generalization) 성능이 뛰어남 (학습하지 않은 상태에 대해서도 적절한 Q 값 추정 가능)

## 2. 가치 함수 근사(Value Function Approximation)

### 가치 함수 근사의 필요성
- **문제**: Q-learning은 Q 값을 테이블 형태로 저장. 현실 문제에서는 상태 공간이 매우 커서 모든 Q 값을 저장하기 어려움
- **해결**: 가치 함수 근사(VFA)를 사용하여 Q 값을 근사

### 가치 함수 근사(VFA)
- **정의**: Q 값을 테이블이 아닌, 파라미터화된 함수(예: 신경망)로 표현
- **특징**:
  - 모든 데이터를 저장할 필요 없이, 파라미터만 저장하면 됨
  - 실제로 경험하지 않은 상태-행동 쌍에 대해서도 Q 값을 추정 가능 (일반화, generalization)
  - 예시: 선형 함수 y = θ₁x (x: 입력, y: Q 값, θ₁: 파라미터)
  - 가설 함수(hypothesis function): h(θ₁)x

### 선형 회귀(Linear Regression) 예시
- **목표**: 실제 데이터를 잘 표현하는 가설 함수를 찾는 것
- **손실 함수(Loss Function)**: 가설 함수가 실제 데이터를 얼마나 잘 표현하는지 나타내는 지표
  - **평균 제곱 오차(Mean Squared Error, MSE)**: 회귀 문제에서 주로 사용되는 손실 함수
    - cost(ω, b) = (1/n)∑[y_i - h(x_i)]²
      - n: 데이터 개수
      - y_i: 실제 값
      - h(x_i): 가설 함수에 의한 예측 값
      - ω, b: 가설 함수의 파라미터
  - **목표**: 손실 함수를 최소화하는 파라미터를 찾는 것

### 경사 하강법(Gradient Descent)
- **정의**: 손실 함수를 최소화하는 파라미터를 찾기 위한 알고리즘
- **과정**:
  1. 임의의 시작점에서 시작
  2. 현재 파라미터에서 손실 함수의 기울기(gradient)를 계산
  3. 기울기의 반대 방향으로 일정 크기(학습률, α)만큼 파라미터를 이동
  4. 수렴할 때까지 2-3번 과정 반복
- **수식**: ω := ω - α(∂/∂ω)cost(ω)
  - :=: 업데이트(할당) 연산자
  - α: 학습률
  - (∂/∂ω)cost(ω): 손실 함수의 기울기(미분)
- **종류**:
  - **배치 경사 하강법(Batch Gradient Descent)**: 전체 데이터를 사용하여 기울기 계산
  - **확률적 경사 하강법(Stochastic Gradient Descent, SGD)**: 데이터 일부(미니배치)를 무작위로 샘플링하여 기울기 계산 (DQN에서 사용)

## 3. 강화 학습에 딥러닝 적용 시 문제점

### 데이터 지연(Delayed Reward)
- **강화 학습**: 행동에 대한 보상이 즉시 주어지지 않고 지연될 수 있음 (예: 벽돌 깨기 게임에서 벽돌을 깰 때까지 보상 없음)
- **딥러닝**: 레이블이 지정된 데이터가 필요하며, 즉각적인 피드백을 가정
- **문제**: 지연된 보상으로 인해 학습 과정이 불안정해질 수 있음

### 데이터 상관관계(Data Correlation)
- **강화 학습**: 에이전트가 순차적으로 경험하는 데이터는 서로 상관관계가 높음 (예: Breakout 게임에서 1초 간격으로 수집되는 데이터는 매우 유사)
- **딥러닝**: 학습 데이터는 무작위로 추출(i.i.d., independent and identically distributed)되어야 함 (전체 데이터 분포를 대표해야 함)
- **문제**: 상관관계가 높은 데이터로 학습하면 가치 함수 근사기가 전체 데이터를 제대로 표현하지 못함

### 데이터 분포 변화(Non-stationary Data Distribution)
- **딥러닝**: 학습 데이터의 분포가 변하지 않는다고 가정
- **강화 학습**: 학습이 진행됨에 따라 Q 값이 변하고, 이에 따라 에이전트의 행동(정책)이 변하며, 결국 데이터 분포가 변함
- **문제**: 데이터 분포의 급격한 변화는 학습 파라미터의 진동을 유발하여 불안정성 초래

### 움직이는 타겟 값(Moving Target Value)
- **Q-learning**: Target Value = r + γmax_(a')Q(s', a')
- **DQN**: Q 값을 나타내는 파라미터(θ)가 업데이트될 때마다 Target Value도 함께 변동하여 학습 불안정
- **문제**: Target Value가 계속 변동하면 학습 목표가 불안정해짐

## 4. DQN의 해결 방안

### 경험 리플레이(Experience Replay)
- **아이디어**: 에이전트의 경험 (s_t, a_t, r_t, s_(t+1))을 리플레이 버퍼(Replay Buffer)에 저장하고, 학습 시에는 버퍼에서 무작위로 샘플링(미니배치)하여 사용
- **효과**:
  - **데이터 지연 문제 완화**: 여러 행동을 취해본 후 학습에 사용
  - **데이터 상관관계 감소**: 무작위 샘플링으로 데이터 간 상관관계 감소
  - **데이터 분포 변화 완화**: 과거 경험을 재사용하여 데이터 분포 변화를 완만하게 함

### 타겟 네트워크 분리(Target Network Separation)
- **아이디어**:
  - **Q-network(주 네트워크)**: 현재 행동에 대한 Q 값을 얻는 데 사용. 파라미터 θ
  - **Target Network(타겟 네트워크)**: Target Value를 계산하는 데 사용. 별도의 파라미터 θ⁻
  - Target Network의 파라미터 θ⁻는 주기적으로 Q-network의 파라미터 θ로 업데이트 (소프트 업데이트 또는 하드 업데이트)
- **효과**: Target Value의 변동성을 줄여 학습 안정성 향상

## 5. DQN 알고리즘

### DQN의 Q 값 업데이트
- **손실 함수(Loss Function)**: L_i(θ_i) = E_(s, a, r, s' ~ U(D))[(r + γmax_(a')Q(s', a'; θ_i⁻) - Q(s, a; θ_i))²]
  - θ_i: Q-network의 파라미터
  - θ_i⁻: Target Network의 파라미터
  - U(D): 리플레이 버퍼 D에서 균등 분포(uniform distribution)로 샘플링
- **기울기(Gradient)**: ∇θ_i L(θ_i) = E_(s, a, r, s' ~ U(D))[(r + γmax_(a')Q(s', a'; θ_i⁻) - Q(s, a; θ_i))∇θ_i Q(s, a; θ_i)]
  - SGD를 사용하여 손실 함수를 최소화하는 방향으로 파라미터 θ_i 업데이트

### DQN 수도 코드(Pseudo-code)
```
# 초기화
Initialize replay memory D to capacity N
Initialize action-value function Q with random weights θ
Initialize target action-value function Q̂ with weights θ⁻ = θ

For episode = 1, M do
    Initialize sequence s₁ = {x₁} and preprocessed sequence ∅₁ = ∅(s₁)
    For t = 1, T do
        # 행동 선택 (ε-greedy)
        With probability ε select a random action a_t
        otherwise select a_t = max_a Q*(∅(s_t), a; θ)

        # 행동 수행 및 관찰
        Execute action a_t in emulator and observe reward r_t and image x_(t+1)
        Set s_(t+1) = s_t, a_t, x_(t+1)
        Preprocess ∅_(t+1) = ∅(s_(t+1))
        
        # 경험 저장
        Store transition (∅_t, a_t, r_t, ∅_(t+1)) in D

        # 미니배치 샘플링 및 학습
        Sample random minibatch of transitions (∅_j, a_j, r_j, ∅_(j+1)) from D

        # Target Value 계산
        Set y_j = { r_j                                   if episode terminates at step j+1
                  { r_j + γmax_(a')Q̂(∅_(j+1), a'; θ⁻)  otherwise

        # 손실 계산 및 파라미터 업데이트
        Perform a gradient descent step on (y_j - Q(∅_j, a_j; θ))² with respect to the network
        parameters θ

        # Target Network 업데이트 (C 스텝마다)
        Every C steps reset Q̂ = Q
    End For
End For
```

## 6. DQN의 PyTorch 구현

### 핵심 구성 요소
1. Q-network와 Target Network
2. 경험 리플레이 버퍼
3. 학습 알고리즘 (손실 함수 계산 및 파라미터 업데이트)

### 행동 선택 (ε-greedy)
```python
# ε-greedy 알고리즘으로 행동 선택
exploration_rate = ...  # ε 값 (스케줄러를 통해 감소)

if np.random.rand() < exploration_rate:
    action = env.action_space.sample()  # 무작위 행동 (탐험)
else:
    q_values = model.policy.forward(obs)  # Q-network를 통해 Q 값 계산
    action = torch.argmax(q_values).item()  # 최대 Q 값을 갖는 행동 (활용)
```

### 경험 리플레이(Experience Replay)
```python
# 리플레이 버퍼 생성
replay_buffer = ReplayBuffer(...)

# 경험 저장
replay_buffer.add(obs, action, reward, next_obs, done)

# 미니배치 샘플링
replay_data = replay_buffer.sample(batch_size)
```

### Q 값 업데이트
```python
# Target Network를 사용하여 다음 상태의 최대 Q 값 계산
with torch.no_grad():
    next_q_values = model.policy_target.forward(replay_data.next_observations)
    next_q_values, _ = next_q_values.max(dim=1)  # max_a' Q(s', a')
    target_q_values = replay_data.rewards + (1 - replay_data.dones) * gamma * next_q_values

# 현재 Q 값 계산
current_q_values = model.policy.forward(replay_data.observations)
current_q_values = torch.gather(current_q_values, dim=1, index=replay_data.actions.long())

# 손실 계산 (MSE)
loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))

# Q-network 업데이트 (SGD)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### Target Network 업데이트
```python
# Hard update (C 스텝마다)
if total_timesteps % target_network_update_freq == 0:
    model.policy_target.load_state_dict(model.policy.state_dict())

# 또는 Soft update (매 스텝마다)
for target_param, param in zip(model.policy_target.parameters(), model.policy.parameters()):
    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
```

## 7. DQN의 주요 발전 형태

### Double DQN (DDQN)
- **문제**: 기존 DQN은 Q 값을 과대평가(overestimation)하는 경향이 있음
- **해결**: 행동 선택과 Q 값 평가를 분리
  - Q-network로 행동 선택: a' = argmax_a Q(s', a; θ)
  - Target Network로 Q 값 평가: Q(s', a'; θ⁻)

### Prioritized Experience Replay (PER)
- **아이디어**: TD 오차가 큰 경험을 더 자주 샘플링하여 학습 효율 향상
- **방법**: 각 경험에 우선순위(priority)를 부여하고, 우선순위에 비례하여 샘플링

### Dueling DQN
- **아이디어**: Q-network 구조를 변경하여 가치 함수 V(s)와 어드밴티지 함수 A(s, a)로 분리
- **효과**: 특정 상태에서 어떤 행동이 좋은지 명시적으로 학습 가능

### Rainbow DQN
- **아이디어**: 여러 DQN 개선 기법을 결합
- **포함 기법**: Double DQN, Prioritized Experience Replay, Dueling DQN, Multi-step learning, Distributional RL, Noisy Nets

## 요약

1. **DQN의 의의**:
   - Q-learning과 딥러닝을 결합하여 고차원 문제 해결
   - Q-table의 한계 극복, 게임과 같은 복잡한 환경에서 강화학습 적용 가능

2. **강화 학습에 딥러닝 적용 시 문제점**:
   - 데이터 지연(Delayed Reward)
   - 데이터 상관관계(Data Correlation)
   - 데이터 분포 변화(Non-stationary Data Distribution)
   - 움직이는 타겟 값(Moving Target Value)

3. **DQN의 핵심 기법**:
   - **경험 리플레이(Experience Replay)**: 과거 경험을 저장하고 무작위로 샘플링하여 학습
   - **타겟 네트워크 분리(Target Network Separation)**: 목표 값의 안정성을 높여 학습 성능 향상

4. **DQN 알고리즘 수식**:
   - 손실 함수(Loss Function): L_i(θ_i) = E[(r + γmax_(a')Q(s', a'; θ_i⁻) - Q(s, a; θ_i))²]
   - 경사 하강법을 통한 파라미터 업데이트

5. **DQN의 발전**:
   - Double DQN, Prioritized Experience Replay, Dueling DQN, Rainbow DQN 등 다양한 개선 기법 등장
   - 더 복잡한 문제에 대한 적용 가능성 확장
