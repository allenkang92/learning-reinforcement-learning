# Model-free 강화학습과 가치 기반 알고리즘

## 1. Model-free 강화학습이란?

### Model-free vs. Model-based
- **Model-free**:
  - 환경의 동작 방식을 모르는 상태에서 학습
  - 데이터(상태, 행동, 보상)를 통해 행동 선택
  - **예시**: 자동차 운전에서 손잡이에 힘을 주는 정도, 페달 밟는 시기 등 행동을 정의한 후 스스로 학습

- **Model-based**:
  - 에이전트가 행동하는 모델에 대한 정의가 있는 상태에서 학습
  - 환경의 역학(dynamics)을 알고 있거나 학습함

### Planning vs. Learning
- **Planning**: 환경 모델을 알고 있을 때, 이를 기반으로 문제 해결
- **Learning**: 환경에 대한 배경지식 없이, 에이전트와 환경의 상호작용을 통해 문제 해결

### Model-free 방식 장단점
- **장점**: 다양한 분야 문제 해결 가능
- **단점**: Model-based 방식보다 덜 효율적일 수 있음 (최근 연구에서는 시간 효율 측면에서 더 좋은 성과를 내기도 함)

### Model-based 방식 장단점
- **장점**: 정해진 계획(Plan)을 통해 정책 학습, 빠른 결과 도달 가능
- **단점**: 적용 범위 제한적 (환경 모델 필요)

## 2. 가치 기반 강화학습 (Value-based Reinforcement Learning)

### 학습 흐름
```
데이터(경험) -> 가치함수(v(s) or q(s,a)) 추정 -> 정책 업데이트
```

### 강화학습 접근 방식 비교
- **Value-based**:
  - 가치 함수가 완벽하다는 가정 하에 가치 함수만 학습
  - 정책은 암묵적으로 가치 함수를 따름 (예: ε-greedy)
  - **장점**: 데이터 효율적
  - **단점**: 주로 이산 행동(Discrete Action)에 사용, 출력 값이 스칼라 형태라 불안정할 수 있음

- **Policy-based**:
  - 정책을 직접 최적화 (가치 함수 업데이트 X)
  - **장점**: 안정적인 학습, 주로 연속 행동(Continuous Action)에 사용

- **Actor-Critic**:
  - 가치 함수와 정책 모두 업데이트
  - Actor(정책)와 Critic(가치 함수)이 함께 작동

## 3. On-policy vs. Off-policy

### On-policy
- **정의**: 에이전트가 선택한 행동이 정책에 *바로* 반영됨
- **특징**: Target Policy (목표 정책)와 Behavior Policy (행동 정책)가 동일
- **학습 방식**: 직접 경험을 토대로 큐 함수 업데이트
- **예시**: SARSA 알고리즘

### Off-policy
- **정의**: 행동을 선택하는 정책(Behavior Policy)과 목표 정책(Target Policy)이 *다름*
- **특징**: 간접 경험(다른 정책에 의해 생성된 데이터)을 사용하여 학습
- **학습 방식**: 최적 정책을 통해 최대 Return 값을 얻는 방향으로 학습
- **예시**: Q-learning 알고리즘

## 4. SARSA 알고리즘 (On-policy)

### 개념
- TD(Temporal Difference) 기반의 On-policy 알고리즘
- [S_t, A_t, R_t, S_(t+1), A_(t+1)]를 사용하여 큐 함수 업데이트
- 이름의 유래: State, Action, Reward, next State, next Action의 앞글자를 따서 SARSA

### 업데이트 수식
```
Q(S_t, A_t) ← Q(S_t, A_t) + α(R_t + γQ(S_(t+1), A_(t+1)) - Q(S_t, A_t))
```
- α: 학습률 (Learning Rate)
- (R_t + γQ(S_(t+1), A_(t+1)) - Q(S_t, A_t)): TD Error

### 특징
- 다음 상태의 행동까지 고려하여 큐 함수 업데이트
- On-policy: Behavior Policy의 탐험이 Target Policy 업데이트에 영향
- 탐험 중 좋지 않은 결과도 학습에 반영됨

## 5. Q-learning 알고리즘 (Off-policy)

### 개념
- TD 기반의 Off-policy 알고리즘
- 벨만 최적 방정식을 사용하여 큐 함수 업데이트
- 현재 정책과 상관없이 최적 정책을 학습

### 업데이트 수식
```
Q(S_t, A_t) ← Q(S_t, A_t) + α(R_t + γmax_(a')Q(S_(t+1), a') - Q(S_t, A_t))
```
- max_(a')Q(S_(t+1), a'): 다음 상태에서 최대 큐 함수 값

### SARSA와 Q-learning 비교

| 특징 | SARSA (On-policy) | Q-learning (Off-policy) |
| --- | --- | --- |
| 업데이트 수식 | Q(S_t, A_t) ← Q(S_t, A_t) + α(R_t + γQ(S_(t+1), A_(t+1)) - Q(S_t, A_t)) | Q(S_t, A_t) ← Q(S_t, A_t) + α(R_t + γmax_(a')Q(S_(t+1), a') - Q(S_t, A_t)) |
| 목표값 | R_t + γQ(S_(t+1), A_(t+1)) (다음 상태에서 *실제 선택한 행동*의 큐 함수) | R_t + γmax_(a')Q(S_(t+1), a') (다음 상태에서 *최대 큐 함수*) |
| 경험 | 직접 경험 (자신의 정책에 따라 행동하고 그 결과를 학습) | 간접 경험 (다른 정책-일반적으로 최적 정책-에 의해 생성된 데이터도 학습에 사용) |
| 벨만 방정식 | 벨만 기대 방정식 | 벨만 최적 방정식 |

### 특징
- 현재 정책과 상관없이 최적 정책을 학습
- Off-policy: Behavior Policy(ε-greedy)와 Target Policy(greedy) 분리
- 실제 선택한 행동이 아닌, 다음 상태에서 가능한 최대 큐 함수 값을 사용

## 6. Q-learning 수도 코드 및 구현

### Q-learning 수도 코드 (Pseudo-code)
```
# Q-table 초기화: 모든 상태-행동 쌍에 대한 Q 값을 0으로 초기화
Initialize Q(s, a) for all s ∈ S, a ∈ A(s) arbitrarily, and Q(terminal-state, ·) = 0

# 각 에피소드에 대해 반복:
Loop for each episode:
    # 초기 상태 설정:
    Initialize s

    # 에피소드의 각 스텝에 대해 반복:
    Loop for each step of episode:
        # 행동 선택: ε-greedy 정책에 따라 현재 상태 s에서 행동 a를 선택
        Choose a from s using policy derived from Q (e.g., ε-greedy)

        # 행동 수행: 선택한 행동 a를 취하고, 보상 r과 다음 상태 s'를 관찰
        Take action a, observe r, s'

        # Q-value 업데이트:
        Q(s, a) ← Q(s, a) + α[r + γmax_a' Q(s', a') - Q(s, a)]

        # 상태 업데이트:
        s ← s'

    # 에피소드 종료 조건: s가 종료 상태이면 반복 중단
    Until s is terminal
```

### Q-learning 구현 예시
```python
class QLearningAgent:
    def __init__(self, actions):
        self.actions = actions  # 가능한 행동
        self.learning_rate = 0.01  # 학습률 (α)
        self.discount_factor = 0.9  # 감가율 (γ)
        self.epsilon = 0.9  # 초기 ε 값
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])  # Q-table

    def learn(self, state, action, reward, next_state):
        # 현재 Q 값
        q_1 = self.q_table[state][action]
        # 벨만 최적 방정식을 이용한 Q 값 업데이트
        q_2 = reward + self.discount_factor * max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (q_2 - q_1)

    def get_action(self, state):
        # ε-greedy 정책
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.actions)  # 탐험
        else:
            action = self.arg_max(self.q_table[state])  # 활용
        return action

    @staticmethod
    def arg_max(state_action):
        return np.random.choice(np.where(state_action == np.max(state_action))[0])
```

### 코드 설명
- **__init__**: 에이전트 초기화
  - actions: 가능한 행동 리스트
  - learning_rate: 학습률 (α)
  - discount_factor: 감가율 (γ)
  - epsilon: ε-greedy 정책의 ε 값 (초기값은 높게 설정하여 탐험 위주)
  - q_table: Q-table (기본값이 [0.0, 0.0, 0.0, 0.0]인 defaultdict)

- **learn**: Q-learning 업데이트 규칙 구현
  - q_1: 현재 상태-행동에 대한 Q 값
  - q_2: TD Target (r + γmax_(a')Q(s', a'))
  - 업데이트: Q(s, a) ← Q(s, a) + α(TD Target - Q(s, a))

- **get_action**: ε-greedy 정책 구현
  - ε 확률로 무작위 행동 선택 (탐험)
  - 1-ε 확률로 최대 Q 값을 갖는 행동 선택 (활용)

- **arg_max**: 최대 Q 값을 가지는 action의 인덱스 반환 (여러 개일 경우 무작위 선택)

## 요약

1. **Model-free 강화학습**: 환경 모델 없이 데이터 기반으로 학습하는 방식

2. **가치 기반 강화학습**: 가치 함수 또는 큐 함수를 학습하여 정책을 결정하는 방식
   - **Value-based**: 가치 함수만 학습, 정책은 암묵적으로 따라옴
   - **Policy-based**: 정책을 직접 최적화
   - **Actor-Critic**: 가치 함수와 정책 모두 학습

3. **On-policy vs. Off-policy**:
   - **On-policy**: 행동 정책 = 목표 정책 (직접 경험)
   - **Off-policy**: 행동 정책 ≠ 목표 정책 (간접 경험)

4. **SARSA vs. Q-learning**:
   - **SARSA**: On-policy, 다음 행동까지 고려하여 큐 함수 업데이트
   - **Q-learning**: Off-policy, 다음 상태에서 최대 큐 함수를 사용하여 업데이트

5. **Q-learning의 핵심**:
   - **Off-policy**: 행동 정책과 목표 정책 분리
   - **TD Control**: 매 타임스텝마다 Q 값 업데이트
   - **벨만 최적 방정식**: Q(s, a) ← Q(s, a) + α(r + γmax_(a')Q(s', a') - Q(s, a))
