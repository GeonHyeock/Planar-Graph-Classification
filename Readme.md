## DATA

<details>
<summary>Version001</summary>

### 특징

- 그래프는 연결그래프만을 대상으로 한다.
- 그래프의 정보는 csv파일로 노드의 관계로 표현한다.
- Label 정보는 채색수가 2이면 1 아니면 0으로 표현한다.

#### 문제 및 고려할 점

- 채색수를 Greedy 알고리즘으로 구해 정확하지 않다.
- 최대 노드수를 결정해야 한다.
- 데이터가 랜덤하게 생성되었으므로 Train, Valid, Test 데이터에 중복이 있을 수 있다.

</details>

----------------------------------------

<details>
<summary>Version002</summary>

### 특징

- 그래프는 연결그래프만을 대상으로 한다.
- 그래프의 정보는 csv파일로 노드의 관계로 표현한다.
- 채색수를 2~10을 대상으로 한다.
- 채색수를 결정하고 그에 대응하는 데이터를 샘플링 한다.
- 최대 노드수를 50으로 진행한다.

#### 문제 및 고려할 점

- 샘플링 기법을 통하여 추출된 데이터는 채색수가 k일때 k-2의 완전그래프를 부분그래프로 갖는다. 즉 편향이 존재
- 데이터가 랜덤하게 생성되었으므로 Train, Valid, Test 데이터에 중복이 있을 수 있다.

</details>

----------------------------------------

<details>
<summary>Version003</summary>

### 특징

- 그래프는 연결그래프만을 대상으로 한다.
- 그래프의 정보는 csv파일로 노드의 관계로 표현한다.
- Label 정보는 채색수가 2이면 1 아니면 0으로 표현한다.
- 최소 노드수를 10으로 진행한다. (노드수가 작으면 동형의 그래프가 많이 생김)
- 최대 노드수를 50으로 진행한다.


#### 문제 및 고려할 점

- 데이터가 랜덤하게 생성되었으므로 Train, Valid, Test 데이터에 중복이 있을 수 있다.
- version001과 다르게 노드의 수에 비례하여 weight sampling 진행, 노드의 수가 많은수록 더 다양한 그래프가 존재하기 때문이다.
- sampling시 weight를 어떻게 줄지 좀 더 고민! 우선은 노드의 제곱값으로 weight (노드별 가능한 엣지수가 제곱에 비례)

</details>

<details>
<summary>Version004</summary>

### 특징

- 그래프는 연결그래프만을 대상으로 한다.
- 그래프의 정보는 csv파일로 노드의 관계로 표현한다.
- Label은 그래프가 포함하는 삼각형의 수
- 최소 노드수를 10으로 진행한다.
- 최대 노드수를 50으로 진행한다.


#### 문제 및 고려할 점

- 데이터가 랜덤하게 생성되었으므로 Train, Valid, Test 데이터에 중복이 있을 수 있다.
- weight sampling 진행, 노드의 수가 많은수록 더 다양한 그래프가 존재하기 때문이다.
- sampling시 weight를 어떻게 줄지 좀 더 고민! 우선은 노드의 제곱값으로 weight (노드별 가능한 엣지수가 제곱에 비례)

</details>