# Deep Q-Network(DQN)을 적용한 2048 게임

## 개요

이 프로젝트는 강화학습 알고리즘인 Deep Q-Network(DQN)을 사용하여 2048 게임을 자동으로 플레이하는 인공지능 에이전트를 학습하는 것을 목표로 합니다. 2048 게임은 4x4 그리드에서 숫자 타일을 합쳐 2048 타일을 만드는 퍼즐 게임입니다. 이 프로젝트는 Python으로 작성되었으며, 주피터 노트북 파일(.ipynb)로 구현되었고 Keras-RL 라이브러리를 활용하여 DQN 구현을 하였습니다.

## 프로젝트 실행 환경

프로젝트의 실행에 앞서 GPU 지원을 위해 텐서플로우 버전에 맞는 파이썬/컴파일러/cuDNN/쿠다 버전을 확인합니다.

버전 확인 : [https://www.tensorflow.org/install/source?hl=ko#gpu](https://www.tensorflow.org/install/source?hl=ko#gpu)

### 테스트 실행 환경

- Windows 환경
- NVIDIA GeForce GTX 1060 3GB

```bash
Python                       3.10.10
cuDNN                        8.1
CUDA                         11.2
jupyter                      1.0.0
```

### 테스트 라이브러리 버전

```bash
gym                          0.26.2
keras                        2.10.0
keras-rl2                    1.0.5
tensorflow                   2.10.0
tensorflow-gpu               2.10.0
numpy                        1.26.4
matplotlib                   3.7.0
```

### 라이브러리 설치

프로젝트를 실행하기 위해 필요한 라이브러리는 다음과 같습니다

```bash
pip install gym keras-rl2 tensorflow numpy matplotlib
```

## 2048 게임 규칙

2048 게임은 간단한 퍼즐 게임으로, 플레이어는 격자판에서 같은 숫자를 합쳐 더 큰 숫자를 만드는 것이 목표입니다.

1. 격자판: 게임은 4x4 격자판에서 진행됩니다.
2. 초기 설정: 게임을 시작하면 두 개의 타일이 랜덤한 위치에 나타납니다. 타일에는 숫자 2 또는 4가 적혀 있습니다.
3. 타일 이동: 플레이어는 화살표 키(또는 화면 스와이프)를 사용해 타일을 상하좌우로 이동시킬 수 있습니다. 모든 타일은 선택한 방향으로 최대한 이동합니다.
4. 타일 결합: 같은 숫자가 적힌 타일 두 개가 충돌하면 하나로 합쳐집니다. 예를 들어, 두 개의 2 타일이 합쳐지면 4 타일이 됩니다. 이 과정에서 합쳐진 타일의 숫자는 합쳐진 두 타일의 숫자의 합이 됩니다.
5. 새로운 타일 추가: 각 이동 후에는 새로운 숫자 2 또는 4 타일이 빈 칸에 무작위로 생성됩니다.
6. 게임 오버: 더 이상 타일을 이동시킬 수 없게 되면(즉, 격자판에 빈 칸이 없고 인접한 타일들이 모두 다른 숫자인 경우) 게임이 종료됩니다.
7. 점수: 타일을 합칠 때마다 합쳐진 숫자만큼 점수가 올라갑니다. 최종 점수는 게임 오버 시의 총 점수입니다.

2048 게임 플레이 해보기 : [https://play2048.co/](https://play2048.co/)

## 2048 게임 환경 설정

2048 게임 환경은 OpenAI Gym 인터페이스를 사용하여 2048 게임을 구현한 것입니다.

- 상태 (State): 보드의 현재 상태를 나타내는 4x4 행렬. 보드의 각 타일 값.
- 행동 (Action): 4가지 방향(위, 오른쪽, 아래, 왼쪽).
- 보상 (Reward): 이동 후 합쳐진 타일의 총 값. 이동이 되지 않은 경우(IllegalMove) 보상은 0

### 2048 게임 구현(Game2048Env)

2048 게임의 모든 기본 게임 로직을 포함하며, 에이전트가 게임을 플레이할 수 있도록 다양한 메서드를 제공합니다.

```python
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

class IllegalMove(Exception):
    pass

class Game2048Env(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        # 게임 환경 초기화
        self.size = 4  # 보드 크기 (4x4)
        self.w = self.size
        self.h = self.size
        squares = self.size * self.size

        # 게임 점수 초기화
        self.score = 0

        # Gym 인터페이스를 위한 멤버 변수 초기화
        self.action_space = spaces.Discrete(4)  # 4가지 이동 방향 (위, 오른쪽, 아래, 왼쪽)
        self.observation_space = spaces.Box(0, 2**squares, (self.w * self.h, ), dtype=int)  # 보드 상태를 나타내는 공간
        self.reward_range = (0., float(2**squares))  # 가능한 보상 범위 설정

        # Gym 환경의 랜덤 시드 초기화
        self.seed()

        # 새 게임을 위해 보드 초기화
        self.reset()

    def seed(self, seed=None):
        # Gym 환경의 랜덤 시드 설정
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # Gym 인터페이스 구현:
    def step(self, action):
        # 게임의 한 단계를 수행 (이동 및 새 타일 추가)
        logging.debug("Action {}".format(action))
        score = 0
        done = None
        try:
            score = float(self.move(action)) # 주어진 방향으로 이동
            self.score += score
            assert score <= 2**(self.w*self.h)
            self.add_tile() # 새 타일 추가
            done = self.isend() # 게임 종료 여부 확인
            reward = float(score)
        except IllegalMove as e:
            logging.debug("Illegal move")
            done = False
            reward = 0. # 이동하지 않은 action에 대한 보상은 0, 음수 보상을 설정할 수도 있다.

        observation = self.Matrix

        # 게임의 각 단계/이동을 실행한 후 호출자에게 추가 정보를 저장
        # 테스트 및 훈련 중인 에이전트를 모니터링하기 위한 콜백 함수로 사용
        info = {"max_tile": self.highest()}

        # 관측값 (보드 상태), 보상, 종료 여부, 정보 반환
        return observation, reward, done, info

    def reset(self):
        # 게임 보드 초기화 및 2개의 타일 추가
        self.Matrix = np.zeros((self.h, self.w), int)
        self.score = 0

        logging.debug("Adding tiles")
        self.add_tile()
        self.add_tile()

        return self.Matrix

    def render(self, mode='human'):
        # 게임 보드 상태를 출력하여 렌더링
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        s = 'Score: {}\n'.format(self.score)
        s += 'Highest: {}\n'.format(self.highest())
        npa = np.array(self.Matrix)
        grid = npa.reshape((self.size, self.size))
        s += "{}\n".format(grid)
        outfile.write(s)
        return outfile

    # 2048 게임 로직 구현:
    def add_tile(self):
        # 확률에 따라 값이 2 또는 4인 새 타일 추가
        val = 0
        if self.np_random.random() > 0.8:
            val = 4
        else:
            val = 2
        empties = self.empties() # 빈 칸 목록 가져오기
        assert empties
        empty_idx = self.np_random.choice(len(empties))
        empty = empties[empty_idx]
        logging.debug("Adding %s at %s", val, (empty[0], empty[1]))
        self.set(empty[0], empty[1], val)

    def get(self, x, y):
        # 지정된 위치의 타일 값 가져오기
        return self.Matrix[x, y]

    def set(self, x, y, val):
        # 지정된 위치의 타일 값 설정
        self.Matrix[x, y] = val

    def empties(self):
        # 빈 칸의 위치 목록 반환
        empties = list()
        for y in range(self.h):
            for x in range(self.w):
                if self.get(x, y) == 0:
                    empties.append((x, y))
        return empties

    def highest(self):
        # 보드에서 가장 높은 타일 값 반환
        highest = 0
        for y in range(self.h):
            for x in range(self.w):
                highest = max(highest, self.get(x, y))
        return highest

    def move(self, direction, trial=False):
        # 지정된 방향으로 이동 수행
        # 0=위, 1=오른쪽, 2=아래, 3=왼쪽
        if not trial:
            if direction == 0:
                logging.debug("Up")
            elif direction == 1:
                logging.debug("Right")
            elif direction == 2:
                logging.debug("Down")
            elif direction == 3:
                logging.debug("Left")

        changed = False
        move_score = 0
        dir_div_two = int(direction / 2)
        dir_mod_two = int(direction % 2)
        shift_direction = dir_mod_two ^ dir_div_two # 이동 방향 설정

        # 행/열 범위 구성
        rx = list(range(self.w))
        ry = list(range(self.h))

        if dir_mod_two == 0:
            # 위 또는 아래로 이동
            for y in range(self.h):
                old = [self.get(x, y) for x in rx]
                (new, ms) = self.shift(old, shift_direction)
                move_score += ms
                if old != new:
                    changed = True
                    if not trial:
                        for x in rx:
                            self.set(x, y, new[x])
        else:
            # 왼쪽 또는 오른쪽으로 이동
            for x in range(self.w):
                old = [self.get(x, y) for y in ry]
                (new, ms) = self.shift(old, shift_direction)
                move_score += ms
                if old != new:
                    changed = True
                    if not trial:
                        for y in ry:
                            self.set(x, y, new[y])
        if changed != True:
            raise IllegalMove

        return move_score

    def combine(self, shifted_row):
        # 한쪽으로 이동할 때 동일한 타일 결합 (왼쪽으로 이동)
        move_score = 0
        combined_row = [0] * self.size
        skip = False
        output_index = 0
        for p in pairwise(shifted_row):
            if skip:
                skip = False
                continue
            combined_row[output_index] = p[0]
            if p[0] == p[1]:
                combined_row[output_index] += p[1]
                move_score += p[0] + p[1]
                # 다음 항목을 리스트에서 건너뜁니다.
                skip = True
            output_index += 1
        if shifted_row and not skip:
            combined_row[output_index] = shifted_row[-1]

        return (combined_row, move_score)

    def shift(self, row, direction):
        # 한 행을 왼쪽 또는 오른쪽으로 이동 및 결합
        length = len(row)
        assert length == self.size
        assert direction == 0 or direction == 1

        # 모든 0이 아닌 숫자를 앞으로 이동
        shifted_row = [i for i in row if i != 0]

        # 오른쪽으로 이동하는 경우 리스트를 뒤집습니다.
        if direction:
            shifted_row.reverse()

        (combined_row, move_score) = self.combine(shifted_row)

        # 오른쪽으로 이동 시 리스트 뒤집기
        if direction:
            combined_row.reverse()

        assert len(combined_row) == self.size
        return (combined_row, move_score)

    def isend(self):
        # 게임 종료 여부 확인 (2048 타일 또는 합법적인 이동이 없는 경우)
        if self.highest() == 2048:
            return True

        for direction in range(4):
            try:
                self.move(direction, trial=True)
                # 어떤 이동도 할 수 있으면 종료되지 않음
                return False
            except IllegalMove:
                pass
        return True

    def get_board(self):
        # 전체 보드 매트릭스를 반환
        return self.Matrix

    def set_board(self, new_board):
        # 전체 보드 매트릭스를 설정
        self.Matrix = new_board
```

- `__init__` : 환경을 초기화합니다. 게임 보드 크기, 점수, 액션 공간, 관측 공간 등을 설정합니다.
- `seed` : 환경의 랜덤 시드를 설정하여 게임의 재현성을 보장합니다.
- `step` : 주어진 액션에 따라 게임의 한 단계를 수행하고 보드 상태를 업데이트합니다. 새로운 상태, 보상, 게임 종료 여부, 정보를 반환합니다.
- `reset` : 게임 보드를 초기화하고 두 개의 타일을 추가합니다. 초기 상태의 보드를 반환합니다.
- `render` : 게임 보드를 출력하여 렌더링합니다. mode에 따라 출력 방법을 다르게 합니다.
- `add_tile` : 보드에 새 타일을 추가합니다. 새 타일은 2 또는 4가 될 확률이 있습니다.
- `get` : 지정된 위치의 타일 값을 가져옵니다.
- `set` : 지정된 위치에 타일 값을 설정합니다.
- `empties` : 보드에서 빈 칸의 위치 목록을 반환합니다.
- `highest` : 보드에서 가장 높은 타일 값을 반환합니다.
- `move` : 지정된 방향으로 타일을 이동시킵니다. 이동이 불가능한 경우 IllegalMove 예외를 발생시킵니다.
- `combine` : 한쪽으로 이동할 때 동일한 타일을 결합합니다.
- `shift` : 한 행을 왼쪽 또는 오른쪽으로 이동하고 결합합니다.
- `isend` : 게임 종료 여부를 확인합니다. 2048 타일을 얻거나 가능한 이동이 없을 때 종료됩니다.
- `get_board` : 보드 상태를 반환합니다.
- `set_board` : 보드 상태를 설정합니다.

2048 게임을 Open AI gym 환경에서 실행하기 위해 [https://github.com/rgal/gym-2048](https://github.com/rgal/gym-2048)의 envs/game2048_env.py을 참고하였습니다.

## 신경망 입력 전처리 과정

신경망 입력 전처리는 2048 게임 보드를 신경망이 효과적으로 학습할 수 있는 형태로 변환하는 과정입니다. 이 프로젝트에서는 두 가지 전처리 방식을 구현했습니다.

- 로그 정규화 : 보드 값을 축소하여 학습을 용이하게 합니다.
- 원-핫 인코딩 : 각 타일 값을 별도의 채널로 처리하여 보드 상태를 보다 명확하게 인식할 수 있게 합니다.

### 로그 정규화 (Log2NNInputProcessor)

로그 정규화는 게임 보드 값을 로그 스케일로 변환하여 신경망 입력으로 사용합니다. 이는 입력 값을 보다 작은 범위로 변환하여 학습을 용이하게 합니다.

```python
class Log2NNInputProcessor(Processor):
    # Keras-RL에서 환경 상태의 각 관측값을 전처리
    def process_observation(self, observation): # observation : 2048 게임의 보드 매트릭스를 나타내는 numpy.array
        observation = np.reshape(observation, (4, 4))
        observation_temp = np.where(observation <= 0, 1, observation)
        processed_observation = np.log2(observation_temp)/np.log2(65536)
        return processed_observation # 정규화된 게임 보드 매트릭스를 나타내는 numpy.array
```

- 보드 재구성: 입력 받은 보드를 4x4 형태로 변환합니다.
- 0값 처리: 로그 계산이 불가능한 0값을 1로 변환합니다.
- 로그 변환 및 정규화: 보드의 각 값을 로그 변환하고, 최대 타일 값인 65536의 로그로 나눠 정규화합니다.

### 원-핫 인코딩 (OneHotNNInputProcessor)

원-핫 인코딩은 보드의 각 값을 이진 행렬로 변환하여 신경망 입력으로 사용합니다. 이 방식은 각 타일 값에 대해 별도의 채널을 사용하여 입력을 제공합니다.

```python
class OneHotNNInputProcessor(Processor):
    def __init__(self, num_one_hot_matrices=16, window_length=1, model="dnn"): # num_one_hot_matrices: 각 게임 그리드를 인코딩하는 데 사용할 매트릭스 수
        self.num_one_hot_matrices = num_one_hot_matrices
        self.window_length = window_length
        self.model = model
        self.game_env = Game2048Env()
        self.table = {2**i:i for i in range(1,self.num_one_hot_matrices)}
        self.table[0] = 0

    # 그리드를 인코딩 : 원-핫 인코딩
    def one_hot_encoding(self, grid): # grid: 2048 게임의 보드 매트릭스를 나타내는 4x4 numpy.array
        grid_onehot = np.zeros(shape=(self.num_one_hot_matrices, 4, 4))
        for i in range(4):
            for j in range(4):
                grid_element = grid[i, j]
                grid_onehot[self.table[grid_element],i, j]=1
        return grid_onehot # num_one_hot_matrices 수의 4x4 numpy.array를 포함하는 numpy.array

    # 다음 스텝에 가능한 4개의 그리드 반환
    def get_grids_next_step(self, grid): # grid: 2048 게임의 보드 매트릭스를 나타내는 4x4 numpy.array
        grids_list = []
        for movement in range(4):
            grid_before = grid.copy()
            self.game_env.set_board(grid_before)
            try:
                _ = self.game_env.move(movement)
            except:
                pass
            grid_after = self.game_env.get_board()
            grids_list.append(grid_after)
        return grids_list # 4개의 가능한 움직임에 대한 그리드를 나타내는 numpy.arrays의 리스트

    # 환경 상태의 각 관측값 전처리
    def process_observation(self, observation): # observation: 2048 게임의 보드 매트릭스를 나타내는 numpy.array
        # 다음 2단계 동안 게임 보드 매트릭스를 나타내는 그리드의 리스트 반환
        # 각 그리드는 원-핫 인코딩 방법으로 인코딩
        observation = np.reshape(observation, (4, 4))
        grids_list_step1 = self.get_grids_next_step(observation)
        grids_list_step2 =[]
        for grid in grids_list_step1:
            grids_list_step2.append(grid)
            grids_temp = self.get_grids_next_step(grid)
            for grid_temp in grids_temp:
                grids_list_step2.append(grid_temp)
        grids_list = np.array([self.one_hot_encoding(grid) for grid in grids_list_step2])
        return grids_list

    # 전체 상태의 배치를 처리, 반환
    def process_state_batch(self, batch): # batch (list): 상태의 리스트
        return batch # 처리된 상태의 리스트
```

- `__init__` : 전처리기 생성 시 매트릭스 수, 윈도우 길이, 모델 타입을 설정합니다.
- self.table은 각 타일 값과 해당 인덱스를 매핑하는 딕셔너리입니다.
- `one_hot_encoding` : 보드를 원-핫 인코딩하여 각 타일 값에 대해 별도의 채널을 생성합니다.
- 입력 보드를 반복문을 통해 각 타일 값을 인코딩하여 다차원 배열을 생성합니다.
- `get_grids_next_step` : 다음 스텝 그리드 생성, 현재 보드 상태에서 가능한 모든 이동 후의 보드 상태를 반환합니다.
- 각 이동 방향에 대해 보드를 복사하고 이동 후 상태를 저장합니다.
- `process_observation` : 관측값 전처리, 주어진 보드를 원-핫 인코딩한 배열로 변환합니다.
- 현재 보드 상태에서 가능한 첫 번째와 두 번째 스텝의 모든 보드를 원-핫 인코딩합니다.
- `process_state_batch` : 배치 전처리, 상태 배치를 신경망 입력에 맞게 재구성합니다.

## 강화학습 알고리즘

### DQN (Deep Q-Network)

DQN은 Q-Learning을 딥러닝 모델과 결합한 강화학습 알고리즘입니다. Q-Learning은 상태-행동 가치 함수를 추정하여 최적의 정책을 학습합니다. DQN은 신경망을 사용하여 Q 함수를 근사합니다.

DQN의 주요 개념

- Q-함수 (Q-function): 특정 상태에서 특정 행동을 취했을 때 얻을 수 있는 예상 보상
- 경험 리플레이 (Experience Replay): 에이전트가 경험한 데이터를 메모리에 저장, 샘플링하여 학습에 사용
- 타겟 네트워크 (Target Network): 안정적인 학습을 위해 일정 간격으로만 업데이트 되는 네트워크

### 하이퍼 파라미터

```python
# 하이퍼파라미터:
MODEL_TYPE = 'dnn' # 모델 유형
NUM_ACTIONS_OUTPUT_NN = 4 # 출력 액션의 수
WINDOW_LENGTH = 1 # 윈도우 길이
INPUT_SHAPE = (4, 4) # 입력 형태
PREPROC = "onehot2steps" # 전처리 방법 설정
NUM_ONE_HOT_MAT = 16 # 원핫 매트릭스의 수
NB_STEPS_TRAINING = int(1e6) # 훈련 단계 수
NB_STEPS_ANNEALED = int(1e5) # Annealing 단계 수
NB_STEPS_WARMUP = 5000 # 워밍업 단계 수
MEMORY_SIZE = 6000 # 메모리 크기
TARGET_MODEL_UPDATE = 1000 # 타겟 모델 업데이트 주기
LEARNING_RATE=.00025 # 학습률 설정
GAMMA=.99 # 할인 계수 설정 (미래 보상에 대한 중요도)

# 환경 설정:
ENV_NAME = '2048' # 환경 이름
env = Game2048Env() # 2048 게임 환경 생성

random_seed = random.randint(0, 1000) # 랜덤 시드 설정
print("random_seed: ",random_seed)
random.seed(random_seed) # Python 랜덤 시드 설정
np.random.seed(random_seed) # NumPy 랜덤 시드 설정
env.seed(random_seed) # 환경 랜덤 시드 설정
```

- `MODEL_TYPE` : 사용되는 모델의 유형을 지정합니다.
- `NUM_ACTIONS_OUTPUT_NN` : 신경망의 출력 액션 수를 지정합니다. 2048 게임에서는 상, 하, 좌, 우 네 가지 액션이 있습니다.
- `WINDOW_LENGTH` : 입력으로 고려할 프레임 수를 지정합니다.
- `INPUT_SHAPE` : 신경망의 입력 형태를 지정합니다. (4, 4) (4x4 크기의 게임 보드)
- `PREPROC` : 신경망 입력의 전처리 방법을 지정합니다.
- `NUM_ONE_HOT_MAT` : 원-핫 인코딩에 사용할 매트릭스 수를 지정합니다.
- `NB_STEPS_TRAINING` : 전체 훈련 단계 수를 지정합니다. 현재 값: 1e6 = 1000000 (1백만 단계)
- `NB_STEPS_ANNEALED` : 엡실론 값을 감소시키는 단계 수를 지정합니다.
- `NB_STEPS_WARMUP` : 학습 시작 전에 워밍업으로 사용할 단계 수를 지정합니다.
- `MEMORY_SIZE` : 리플레이 메모리의 크기를 지정합니다.
- `TARGET_MODEL_UPDATE` : 타겟 모델을 업데이트하는 주기를 지정합니다.
- `LEARNING_RATE` : 옵티마이저의 학습률을 지정합니다.
- `GAMMA` : 미래 보상에 대한 할인 계수를 지정합니다. 미래 보상의 중요도를 결정합니다.

### 신경망 모델

신경망 모델은 주어진 상태에서 최적의 행동을 선택하는 DQN 에이전트를 학습하는 데 사용합니다. 2048 게임의 보드 상태를 입력으로 받아 Q 값을 출력하는 신경망 모델을 구축합니다. 모델은 다음과 같이 정의됩니다:

```python
print("훈련 모델 정의")
processor = OneHotNNInputProcessor(num_one_hot_matrices=NUM_ONE_HOT_MAT)
INPUT_SHAPE_DNN = (WINDOW_LENGTH, 4+4*4, NUM_ONE_HOT_MAT,) + INPUT_SHAPE

NUM_DENSE_NEURONS_DNN_L1 = 1024 # 첫 번째 Dense 레이어의 뉴런 수
NUM_DENSE_NEURONS_DNN_L2 = 512 # 두 번째 Dense 레이어의 뉴런 수
NUM_DENSE_NEURONS_DNN_L3 = 256 # 세 번째 Dense 레이어의 뉴런 수
ACTIVATION_FTN_DNN = 'relu' # 활성화 함수
ACTIVATION_FTN_DNN_OUTPUT = 'linear' # 출력 활성화 함수

model = Sequential() # Sequential 모델 생성
model.add(Flatten(input_shape=INPUT_SHAPE_DNN)) # Flatten 레이어 추가
model.add(Dense(units=NUM_DENSE_NEURONS_DNN_L1, activation=ACTIVATION_FTN_DNN)) # 첫 번째 Dense 레이어 추가
model.add(Dense(units=NUM_DENSE_NEURONS_DNN_L2, activation=ACTIVATION_FTN_DNN)) # 두 번째 Dense 레이어 추가
model.add(Dense(units=NUM_DENSE_NEURONS_DNN_L3, activation=ACTIVATION_FTN_DNN)) # 세 번째 Dense 레이어 추가
model.add(Dense(units=NUM_ACTIONS_OUTPUT_NN, activation=ACTIVATION_FTN_DNN_OUTPUT)) # 출력 레이어 추가
print(model.summary())
```

#### 신경망 모델 구조

모델은 Keras의 Sequential API를 사용하여 정의됩니다. keras는 tensorflow에서 제공하는 고수준 신경망 API이며, 딥러닝과 기계학습를 쉽게 할 수 있는 인터페이스입니다. 이 모델은 입력을 평탄화한 후 3개의 Dense 레이어를 거쳐 최종적으로 액션 값을 출력하는 구조입니다. 여기서 사용되는 레이어는 Flatten 레이어와 Dense 레이어입니다.

- INPUT_SHAPE_DNN : 입력 형태는 모델이 기대하는 입력 데이터의 형태를 정의합니다. 이 프로젝트에서는 2048 게임 보드 상태를 입력으로 사용합니다.

#### 신경망 모델 구성

Flatten 레이어: 입력 데이터를 1차원으로 평탄화합니다.

- 입력: 다차원 배열 (예: 4x4 게임 보드)
- 출력: 1차원 벡터

Dense 레이어: 완전 연결 계층으로, 각 레이어는 여러 개의 뉴런으로 구성됩니다.

- 첫 번째 Dense 레이어
  - 뉴런 수: 1024
  - 활성화 함수: ReLU
  - 역할: 입력 데이터를 고차원 특징 공간으로 변환하여 비선형성을 부여합니다.
- 두 번째 Dense 레이어
  - 뉴런 수: 512
  - 활성화 함수: ReLU
  - 역할: 첫 번째 레이어의 출력값을 입력으로 받아 추가적인 특징을 추출합니다.
- 세 번째 Dense 레이어
  - 뉴런 수: 256
  - 활성화 함수: ReLU
  - 역할: 두 번째 레이어의 출력값을 입력으로 받아 추가적인 특징을 추출합니다.

출력 레이어: 각 액션에 대한 Q-값을 출력합니다.

- 뉴런 수: 4 (상, 하, 좌, 우)
- 활성화 함수: 선형 (linear)
- 역할: 각 액션에 대한 Q-값을 계산하여 에이전트가 최적의 행동을 선택할 수 있도록 합니다.

### DQN 에이전트

DQN 에이전트는 keras-rl 라이브러리를 사용하여 구현됩니다.

```python
memory = SequentialMemory(limit=MEMORY_SIZE, window_length=WINDOW_LENGTH)

TRAIN_POLICY = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=0.05, value_min=0.05, value_test=0.01, nb_steps=NB_STEPS_ANNEALED)
TEST_POLICY = EpsGreedyQPolicy(eps=.01)

dqn = DQNAgent(model=model, nb_actions=NUM_ACTIONS_OUTPUT_NN, test_policy=TEST_POLICY, policy=TRAIN_POLICY, memory=memory, processor=processor,
    nb_steps_warmup=NB_STEPS_WARMUP, gamma=GAMMA, target_model_update=TARGET_MODEL_UPDATE, train_interval=4, delta_clip=1.)
dqn.compile(Adam(learning_rate=LEARNING_RATE), metrics=['mse'])

train_2048 = Train2048() # 인스턴스 생성
_callbacks = [train_2048] # 훈련 에피소드 로거 콜백
```

#### 메모리(SequentialMemory)

`memory = SequentialMemory(limit=MEMORY_SIZE, window_length=WINDOW_LENGTH)
`

- 과거의 경험을 저장하고, 학습할 때 샘플링하여 사용합니다.
- 경험 리플레이(experience replay)
- 메모리는 에이전트가 경험한 상태, 행동, 보상, 다음 상태 등을 저장합니다.
- DQN 에이전트는 학습 시 이 메모리에서 무작위로 샘플을 추출하여 학습합니다.
- 데이터 샘플링의 분산을 높여 학습의 안정성을 증가시킵니다.

#### 정책

- 에이전트가 행동을 선택하는 방법을 정의합니다.

학습 정책 (TRAIN_POLICY)

`TRAIN_POLICY = LinearAnnealedPolicy(
    EpsGreedyQPolicy(),
    attr='eps',
    value_max=1.0,
    value_min=0.1,
    value_test=0.05,
    nb_steps=NB_STEPS_ANNEALED
)`

- Epsilon-Greedy 정책 : 무작위 탐색과 최적 행동 선택 간의 균형을 조절합니다.
  - 에이전트는 epsilon 확률로 무작위 행동을 선택하고, 1 - epsilon 확률로 최적 행동을 선택합니다.
- LinearAnnealedPolicy: 학습 초기에 높은 epsilon 값을 사용하여 탐색을 많이 하고, 학습이 진행됨에 따라 epsilon 값을 점차 줄여 최적 행동을 더 많이 선택합니다.

테스트 정책 (TEST_POLICY)

`TEST_POLICY = EpsGreedyQPolicy(eps=0.05)
`

- Epsilon-Greedy 정책: 테스트 시에는 고정된 epsilon 값(0.05)을 사용하여 무작위 행동을 선택합니다.

#### DQN 에이전트 생성

`dqn = DQNAgent(
    model=model,
    nb_actions=NUM_ACTIONS_OUTPUT_NN,
    policy=TRAIN_POLICY,
    memory=memory,
    processor=processor,
    nb_steps_warmup=NB_STEPS_WARMUP,
    gamma=GAMMA,
    target_model_update=TARGET_MODEL_UPDATE,
    train_interval=4,
    delta_clip=1.0
)
`

- Keras-RL의 DQNAgent 클래스를 사용하여 생성됩니다.
- 신경망 모델, 메모리, 정책 등을 사용하여 에이전트를 구성합니다.
- 주요 파라미터
  - `nb_actions`: 에이전트가 선택할 수 있는 행동의 수 (4: 상, 하, 좌, 우)
  - `policy`: 행동을 선택하는 정책 (Epsilon-Greedy 정책)
  - `memory`: 경험을 저장하는 메모리 (SequentialMemory)
  - `processor`: 상태를 전처리하는 프로세서
  - `nb_steps_warmup`: 학습을 시작하기 전에 수행하는 단계 수
  - `gamma`: 미래 보상에 대한 할인 계수 (0.99)
  - `target_model_update`: 타겟 모델을 업데이트하는 주기 (1000 단계마다)
  - `train_interval`: 학습을 수행하는 간격 (4 단계마다)
  - `delta_clip`: 벨만 오차를 클리핑하는 값 (1.0)

#### 모델 컴파일

`dqn.compile(Adam(learning_rate=LEARNING_RATE), metrics=['mse'])
`

- DQN 에이전트를 생성한 후, 옵티마이저와 손실 함수를 설정하여 컴파일합니다.
- Adam 옵티마이저와 평균 제곱 오차(MSE) 손실 함수를 사용합니다.

## 학습 및 평가

### 학습 과정

에이전트는 환경에서 반복적으로 게임을 플레이하며, 경험 리플레이과 타겟 네트워크를 사용하여 Q 함수를 학습합니다. 학습은 다음과 같이 수행됩니다:

```python
dqn.fit(env, callbacks=_callbacks, nb_steps=NB_STEPS_TRAINING, visualize=False, verbose=0) # 모델 훈련
```

- DQN 에이전트는 fit 메서드를 사용하여 학습을 수행합니다.
- 환경, 콜백, 학습 단계 수 등을 인수로 받습니다.
  - `env`: 학습할 환경 (2048 게임 환경)
  - `callbacks`: 학습 과정에서 호출할 콜백 리스트
  - `nb_steps`: 전체 학습 단계 수
  - `visualize`: 학습 중 시각화를 사용할지 여부
  - `verbose`: 학습 과정의 로그 출력을 제어하는 인수

### 학습 평가

학습 후 에이전트의 성능을 평가합니다.

```python
env.reset() # 환경 재설정
dqn.test(env, nb_episodes=5, visualize=False, verbose=0, callbacks=[Test2048()]) # 모델 테스트
```

- `nb_episodes`: 테스트할 에피소드 수
- `visualize`: 테스트 중 시각화를 사용할지 여부
- `callbacks`: 테스트 과정에서 호출할 콜백 리스트

## 실행 결과

학습 과정에 사용된 값(episode, episode_steps, episode_reward, max_tile)들은 신경망 훈련 모니터링 `Train2048` 클래스를 통해 csv 파일로 저장됩니다.

- 파일 저장 위치 : `\Desktop\2048\dqn_2048_train.csv`

### 학습 그래프

1,000,000개의 학습 단계(NB_STEPS_TRAINING)에 대한 실행 결과의 그래프입니다.
`NB_STEPS_TRAINING = int(1e6)`

2514개의 에피소드에 대한 학습이 진행되었습니다.
학습 시간은 약 2시간 20분 소요되었습니다.

- 각 게임/에피소드에서 얻은 최대 타일

![max_tile](max_tile.png)

- 각 게임/에피소드에서 얻은 점수 보상

![reward](reward.png)

- 평균 최대 타일(50개마다 평균 계산) `self.nb_episodes_for_mean = 50`

![max_tiles_means](max_tiles_means.png)

- 평균 점수 보상(50개마다 평균 계산) `self.nb_episodes_for_mean = 50`

![rewards_means](rewards_means.png)

최대 타일 값과 점수가 에피소드의 학습이 거듭됨에 따라 증가되고 있는 것을 확인할 수 있습니다.
학습 결과 초반 에피소드의 최대 타일 값은 128~256에 머무르지만 점차 512 값도 나고, 500번 에피소드 이후로는 1024 값도 나오는 것을 확인할 수 있습니다.
후반으로 갈수록 1024 값이 자주 나오고 평균 및 점수 분포 또한 증가하는 것을 확인할 수 있습니다.

### 학습 평가

비교 : 학습 전 평가 vs. 학습 후 평가

#### 학습 전 평가

```python
Testing for 4 episodes ...
episode: 1, max tile: 128, episode reward: 1336.0, episode steps: 5105
Final Grid:
[[  2   8   4   8]
 [ 32 128  16   2]
 [  2   8  64   4]
 [  4   2  32   2]]

episode: 2, max tile: 64, episode reward: 708.0, episode steps: 2649
Final Grid:
[[ 4  8 16  4]
 [ 8 32  8  2]
 [32 16  4 16]
 [ 2  4 64  4]]

episode: 3, max tile: 64, episode reward: 380.0, episode steps: 2502
Final Grid:
[[ 2  4  2  4]
 [ 4 64  4  2]
 [16  4  8  4]
 [ 2  8  4  2]]

episode: 4, max tile: 64, episode reward: 1144.0, episode steps: 4997
Final Grid:
[[ 4  2 16  8]
 [ 2 64 32  4]
 [16  4  2 64]
 [ 2 64  4  2]]
```

- 평균 점수 : 892
- 최고 점수 : 1336

#### 학습 후 평가

```python
Testing for 4 episodes ...
episode: 1, max tile: 1024, episode reward: 10156.0, episode steps: 867
Final Grid:
[[   2   16    4    2]
 [   8   32   64    4]
 [  16    8  128   16]
 [   2 1024    8    2]]

episode: 2, max tile: 128, episode reward: 1732.0, episode steps: 539
Final Grid:
[[  2  32   4   2]
 [  4   8   2   8]
 [  8 128  32 128]
 [  2   8   2   4]]

episode: 3, max tile: 512, episode reward: 7076.0, episode steps: 922
Final Grid:
[[  4   8  32   8]
 [  8  32 512  64]
 [  4 128   8   4]
 [256  32   4   2]]

episode: 4, max tile: 256, episode reward: 3368.0, episode steps: 584
Final Grid:
[[  2   8 128   2]
...
 [  4  32   8  16]
 [ 16  64 128   4]
 [  8   4  32   2]]
```

- 평균 점수 : 5583
- 최고 점수 : 10156

#### 평가 비교

학습 전 평가와 학습 후 평가의 각 4개의 에피소드의 결과를 비교하면 다음과 같습니다.

- 평균 점수 : 892 vs. 5583 → 4691 차이
- 최고 점수 : 1336 vs. 10156 → 8820 차이

학습을 한 결과 훈련 전과 비교하여 점수가 월등히 높게 나오는 것을 확인 할 수 있습니다.

## 결과 고찰

학습 결과를 분석한 그래프와 평가를 통해, DQN 알고리즘을 사용하여 2048 게임을 학습시키는 데 성공했다는 것을 확인할 수 있습니다. 즉, 에이전트는 게임 보드에서 높은 숫자의 타일을 생성하고 보상을 최대화하기 위해 효과적으로 움직였습니다. 그러나 에피소드에서 최대 타일 값이 2024까지 가지못하고 1024에서 끝난 결과는 아쉬웠습니다. 이는 모델의 성능을 더욱 향상시키는 데 더 많은 개선의 여지가 있다는 것을 알려줍니다. 하이퍼파라미터 튜닝과 모델 구조 개선을 통해 더 나은 결과를 얻기 위한 추가적인 노력이 필요합니다. 특히 시간이 충분하다면 스탭 수를 더 많이 늘려서 학습을 진행하면 더 좋은 결과가 나올 것으로 생각됩니다. 또한 프로젝트의 결과에서 저장된 csv 파일 값을 활용하여 기존 학습 결과를 재사용하는 가능성이 있습니다. 재사용을 통해 추가적인 학습 시간과 리소스를 절약할 수 있습니다.

## 결론

이번 2048 게임에 DQN을 적용한 프로젝트를 통해 강화학습의 기초를 학습하고 적용하는 데 어려운 점도 많았지만 알고리즘과 기법을 최적화하는 과정에서 많은 것을 배울 수 있었습니다. 특히 이번에는 수업에서 배운 tensorflow의 직접 사용 대신에 keras라는 tensorflow에서 제공하는 고수준 신경망 API를 사용하였습니다. 너무나 많은 기술이 빠르게 발전하며 나오고 있고 기존의 기술을 대체해가고 있는 상황에서 앞으로의 프로젝트에서는 더 복잡한 환경과 다양한 알고리즘을 적용하여 더욱 흥미로운 결과를 얻을 수 있을 것으로 기대됩니다. 이번에는 게임 학습에만 국한되었지만 더 많은 강화학습 분야에서의 지식과 능력을 계속해서 쌓도록 노력할 것입니다.
