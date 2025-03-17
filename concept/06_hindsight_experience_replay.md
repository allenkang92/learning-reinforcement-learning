# Hindsight Experience Replay (HER)

## 1. 희소 보상 환경 (Sparse Reward Environment)

### 정의와 문제점
- **정의**: 대부분의 상태에서 보상이 0이고, 특정 상태에서만 드물게 보상이 주어지는 환경
- **예시**:
  - **Push a box**: 상자를 특정 위치로 밀어야 하는 작업. 대부분의 행동은 0의 보상, 목표 위치에 도달해야만 보상
  - **몬테주마의 복수 (Montezuma's Revenge)**: 복잡한 계층적 구조를 가진 게임. 0이 아닌 보상을 얻기 매우 어려움
- **문제점**:
  1. **탐험의 어려움**: 보상 신호를 찾기 위한 탐험이 매우 어려움
  2. **학습 불가**: 보상이 0인 에피소드는 학습에 사용할 수 없음 (사람은 실패로부터 배우지만, 로봇은 실패를 통해 배우기 어려움)

### Bit Flipping 문제
- **문제 설명**:
  - n개의 비트(bit)로 구성된 배열(array)이 주어짐 (각 비트는 0 또는 1)
  - **상태 (State)**: S = {0, 1}^n (n-차원 이진 벡터)
  - **행동 (Action)**: A = {0, 1, ..., n-1} (n개의 비트 중 하나를 선택하여 뒤집는(flip) 행동)
  - **목표**: 초기 상태에서 시작하여 특정 목표 상태(target state)에 도달하는 것
- **강화 학습 적용 시 문제점**:
  - **보상**: 
    - 목표 상태와 정확히 일치하면: 0
    - 그렇지 않으면: -1
    - r_g(s, a) = -[s ≠ g]
  - **희소 보상**: 대부분의 상태에서 보상이 -1 (희소 보상 환경)
  - **학습 어려움**: 상태 공간의 크기가 2^n으로 매우 큼 (n=40이면 1조 개 이상). 일반적인 강화 학습 알고리즘으로는 학습이 어려움

## 2. Cost Engineering (Reward Shaping)

### 개념과 어려움
- **정의**: 강화 학습 알고리즘이 더 잘 학습할 수 있도록 보상 함수를 수정하는 과정
- **필요 지식**: 강화 학습에 대한 깊은 이해 + 문제에 대한 도메인 지식
  - 예: 하키 게임에서 골대에 퍽을 잘 넣는 행동을 알지만, 이를 수학적으로 표현하기는 어려움
- **어려움**: 로봇 제어, 자율 주행과 같은 복잡한 문제에서는 도메인 지식이 있어도 보상 함수 설계가 어려움

### 보상 함수 설계 방법
- **단순화**: 성공/실패로만 보상 함수를 설계하는 것이 실용적일 수 있음 (성공: 1, 실패: 0)
  - **문제점**: 보상이 드물 경우 (희소 보상 환경) 학습이 어려움
- **Bit Flipping 예시 (Reward Shaping)**:
  - r_g(s, a) = -||s - g||^2 (상태와 목표 간의 비트 차이의 제곱에 음수를 취한 값)
  - 비트 차이가 적을수록 더 큰 보상
- **한계**: 복잡한 문제에서는 직관적인 Cost Engineering이 어려움

## 3. Multi-Goal 강화학습

### 개념
- **정의**: 하나의 고정된 목표(Goal)가 아닌, 여러 개의 목표를 설정하고 학습하는 방법
- **목적**: 에이전트가 다양한 목표를 달성하는 방법을 학습하도록 함
- **수학적 표현**:
  - **상태 공간**: S
  - **목표 공간**: G
  - **정책**: π(a|s, g) (상태 s와 목표 g가 주어졌을 때 행동 a를 선택할 확률)
  - **가치 함수**: v(s, g), q(s, a, g) (상태, 행동, 목표에 대한 가치 함수)

### Universal Value Function Approximators (UVFA)
- **정의**: 가치 함수 근사(VFA)를 하나 이상의 목표(goal)에 대해 사용할 수 있도록 확장한 개념
- **특징**:
  - 에이전트는 현재 상태 s뿐만 아니라 목표 g도 입력으로 받아 행동 a를 출력하고 보상 r을 얻음
  - 큐 함수: Q(s, a, g) (상태, 행동, 목표에 의존)
- **목표 달성 여부**:
  - 모든 목표 g는 상태 s에 대해 0 또는 1로 대응 (f_g(s))
  - f_g(s) = 1: 에이전트가 상태 s에서 목표 g를 달성
- **매핑 함수 (Mapping Function)**:
  - m: S → G (상태 공간에서 목표 공간으로의 매핑)
  - ∀s ∈ S, f_(m(s))(s) = 1: 주어진 상태 s가 매핑 함수 m에 의해 매핑된 목표 m(s)를 달성했음을 의미
  - 목표가 도달하고자 하는 상태인 경우: G = S, f_g(s) = [s = g], m은 항등 함수

## 4. Hindsight Experience Replay (HER)

### 핵심 아이디어
- **기본 개념**: 실패한 경험(trajectory)을 다른 목표(goal)에 대해서는 성공한 경험으로 간주하여 학습에 활용
- **목적**: 희소 보상 환경에서 학습 효율 향상
- **작동 방식**:
  - 원래 목표 g에는 도달하지 못했지만, 에피소드에서 실제로 도달한 마지막 상태 s_T를 새로운 목표 g'로 설정
  - s_1, ..., s_T trajectory는 원래 목표 g에 대해서는 실패했지만, 새로운 목표 g'에 대해서는 성공
- **강점**:
  - 희소 보상 환경에서도 학습 가능
  - 도메인 지식 없이도 사용 가능한 일반적인 방법론
  - Off-policy 강화 학습 알고리즘과 함께 사용 가능 (Experience Replay 활용)

### 데이터 저장 방식
- **Replay Buffer 저장**:
  - **Standard Experience**: 경험 (s_t, a_t, r_t, s_(t+1))과 함께 목표(goal) g도 함께 저장: (s_t || g, a_t, r_t, s_(t+1) || g)
    - ||: concatenation (연결)
  - **보상**:
    - 목표 g에 도달하면 양의 보상
    - 목표 g에 도달하지 못하면 음의 보상
  - **Hindsight Experience**: 에피소드가 끝난 후, 원래 목표 g뿐만 아니라, 에피소드에서 실제로 도달했던 상태를 새로운 목표 g'로 설정하여 Replay Buffer에 추가 저장: (s_t || g', a_t, r'_t, s_(t+1) || g')
    - r'_t: 새로운 목표 g'에 대한 보상 (예: g'에 도달했으면 1, 아니면 0)

### 알고리즘 (수도 코드)

```
Algorithm 1 Hindsight Experience Replay (HER)

Input:
  - an off-policy RL algorithm A
  - a strategy S for sampling goals for replay
  - a reward function r: S x A x G -> R

Initialize A and replay buffer R

For episode = 1, M do:
    Sample a goal g and an initial state s₀
    Observe initial state s₀

    For t = 0, T-1 do:
        Sample an action a_t using the policy of A: a_t = π_b(s_t || g)
        Execute a_t, observe next state s_(t+1) and reward r_t := r(s_t, a_t, g)
    End For

    For t = 0, T-1 do:
        Store transition (s_t || g, a_t, r_t, s_(t+1) || g) in R  // Standard experience

        // Hindsight experience
        Sample additional goals g' for replay using strategy S:  G' := S(current episode)
        For each g' ∈ G' do:
            r' := r(s_t, a_t, g')  // Recompute reward with hindsight goal
            Store transition (s_t || g', a_t, r', s_(t+1) || g') in R // Hindsight experience
        End For
    End For

    For t = 1, N do:  // N: number of training iterations
        Sample a minibatch B from R
        Perform one step of optimization of A using B
    End For
End For
```

### 알고리즘 설명
- **입력**:
  - Off-policy RL 알고리즘 (A): DQN, DDPG 등
  - 목표 샘플링 전략 (S): 예: future, final, episode
  - 보상 함수 (r): 상태, 행동, 목표에 따른 보상
- **초기화**:
  - Off-policy 알고리즘 (A) 초기화
  - Replay Buffer (R) 초기화
- **에피소드 반복**:
  - 목표 g와 초기 상태 s₀ 샘플링
  - **타임스텝 반복**:
    - 현재 정책 (π_b)에 따라 행동 a_t 샘플링
    - 행동 실행, 다음 상태 s_(t+1)와 보상 r_t 관찰
  - **Replay Buffer 저장**:
    - **Standard Experience**: (s_t || g, a_t, r_t, s_(t+1) || g) 저장
    - **Hindsight Experience**:
      - 추가 목표 g' 샘플링 (S 전략 사용)
      - 각 g'에 대해 보상 r' 재계산
      - Hindsight Experience (s_t || g', a_t, r', s_(t+1) || g') 저장
  - **학습**: Replay Buffer에서 미니배치 샘플링하여 Off-policy 알고리즘 업데이트

## 5. 목표 샘플링 전략

### 주요 전략
- **future**: 현재 에피소드에서 *미래*에 도달한 상태 중 하나를 목표로 선택 (가장 일반적인 전략)
- **final**: 에피소드의 *마지막* 상태를 목표로 선택
- **episode**: 에피소드 내에서 *임의의* 상태를 목표로 선택

### 목표 샘플링 구현
```python
def sample_goals(self, episode_transitions, transition_idx, n_sampled_goal=4):
    """
    Sample goals based on the specified strategy.
    
    :param episode_transitions: List of transitions in the current episode
    :param transition_idx: Index of the current transition
    :param n_sampled_goal: Number of goals to sample
    :return: List of sampled goals
    """
    # Original and sampled goals
    goals = []
    
    if self.goal_selection_strategy == "final":
        # Add the final state of the episode as a goal
        goals.append(episode_transitions[-1]["next_achieved_goal"])
    
    elif self.goal_selection_strategy == "future":
        # Add goals from future states in the same episode
        current_idx = transition_idx
        future_offset = np.random.randint(0, len(episode_transitions) - current_idx)
        future_idx = current_idx + future_offset
        goals.append(episode_transitions[future_idx]["next_achieved_goal"])
    
    elif self.goal_selection_strategy == "episode":
        # Add goals from any state in the same episode
        episode_idx = np.random.randint(0, len(episode_transitions))
        goals.append(episode_transitions[episode_idx]["next_achieved_goal"])
    
    return goals
```

## 6. HER 구현 및 통합

### Stable-baselines3의 HER 구현
- **HerReplayBuffer 클래스**:
  - __init__: Replay Buffer 초기화, 목표 샘플링 전략 설정
  - add: Standard Experience와 Hindsight Experience를 Replay Buffer에 추가
  - sample: Replay Buffer에서 미니배치 샘플링
  - sample_goals: 목표 샘플링 전략에 따라 추가 목표(g') 샘플링
  - _store_transition: 실제 데이터 저장

### Off-policy 알고리즘과의 통합
- **HER 클래스**:
  - __init__: Off-policy 알고리즘 (DQN, DDPG, SAC 등)과 HerReplayBuffer를 결합
  - learn: 학습 진행 (에피소드 반복, 행동 선택, Replay Buffer 저장, 학습)
  - predict: 학습된 정책을 사용하여 행동 예측

### 통합 과정
```python
# HER 래퍼와 RL 알고리즘 통합 예시
from stable_baselines3 import SAC
from stable_baselines3.her import HerReplayBuffer

# 환경 생성
env = gym.make("FetchReach-v1")

# HerReplayBuffer로 SAC 생성
model = SAC(
    "MultiInputPolicy",
    env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy="future",
        max_episode_length=100,
        online_sampling=True,
    ),
    verbose=1,
)

# 학습
model.learn(total_timesteps=10000)
```

## 7. HER의 장단점

### 장점
- **희소 보상 환경에서 효과적**: 일반 강화학습 알고리즘이 학습하기 어려운 희소 보상 환경에서 효과적으로 학습 가능
- **도메인 지식 불필요**: 보상 함수 설계(Cost Engineering)에 도메인 지식이 적게 필요함
- **실패 경험 활용**: 실패한 에피소드에서도 학습 가능 (인간처럼 실패로부터 배울 수 있음)
- **Off-policy 알고리즘과 호환**: DQN, DDPG, SAC 등 기존 Off-policy 알고리즘과 쉽게 결합 가능

### 단점
- **다중 목표 환경만 적용 가능**: 여러 목표를 설정할 수 있는 환경에서만 적용 가능
- **계산 비용**: Hindsight Experience 생성으로 인한 추가 계산 비용 발생
- **목표 샘플링 전략에 민감**: 목표 샘플링 전략에 따라 성능 차이가 발생할 수 있음
- **환경 모델링**: 목표 공간과 상태 공간의 관계를 정의해야 함 (일부 환경에서는 어려울 수 있음)

## 요약

1. **희소 보상 환경 (Sparse Reward Environment)**:
   - 대부분의 상태에서 보상이 0이고, 특정 상태에서만 보상이 주어지는 환경
   - 일반적인 강화 학습 알고리즘으로는 학습이 어려움

2. **Cost Engineering (Reward Shaping)**:
   - 강화 학습 알고리즘이 더 잘 학습할 수 있도록 보상 함수를 수정하는 과정
   - 복잡한 문제에서는 보상 함수 설계가 어려움

3. **Multi-Goal 강화학습**:
   - 여러 목표를 설정하고 학습하는 방법
   - Universal Value Function Approximators (UVFA)를 통해 구현

4. **Hindsight Experience Replay (HER)**:
   - 실패한 경험을 다른 목표에 대해서는 성공한 경험으로 간주하여 학습에 활용
   - Standard Experience와 Hindsight Experience를 모두 사용하여 학습
   - 목표 샘플링 전략(future, final, episode)을 통해 추가 목표 설정

5. **HER의 의의**:
   - 희소 보상 환경에서 학습 효율 향상
   - Cost Engineering 없이도 효과적인 학습 가능
   - 실패 경험을 활용하여 학습 가능
