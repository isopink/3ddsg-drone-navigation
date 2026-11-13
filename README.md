# 3DDSG Drone Navigation

IFC 기반 **3D 동적 장면 그래프(3DDSG)** 를 지도로 삼아, 실내 드론(**Crazyflie 2.1 + Flow deck**)이
"A방 → C방" 명령을 받으면 경로를 계획하고 충돌 없이 비행하게 하는 프로젝트입니다.

> 그래프 구축은 이 저장소의 범위가 아닙니다. 이 저장소는 **그래프를 받아 드론을 움직이는 부분**을 다룹니다.
> 연구실 데이터(그래프, IFC, 포인트클라우드)는 저장소에 포함하지 않습니다.

## 진행 상황

| 단계 | 내용 | 상태 |
|---|---|---|
| Step 0 | 환경설정, Crazyflie 연결, 호버링/전진 비행 확인 | 완료 |
| Step 1 | 가짜 그래프(방 3, 문 2)로 목표 해석 + 경로 계획 | 예정 |
| Step 2 | Open3D 시각화로 충돌 검증 | 예정 |
| Step 3 | 실기체 테스트 (가상 벽) | 예정 |
| Step 4 | 실제 그래프(Neo4j) 연동 | 예정 |
| Step 5 | 동적 객체 추가 시 재계획 | 예정 |

## 하드웨어와 환경

- Bitcraze **Crazyflie 2.1** + **Flow deck v2** (광류 + ToF 거리센서, 장애물 감지 없음)
- **Crazyradio PA** (USB)
- Windows, Python 3.12, `cflib` / `cfclient`

```
py -3.12 -m venv cf-env
cf-env\Scripts\activate
pip install cfclient neo4j networkx numpy open3d
```

Windows에서는 Crazyradio용 libusb 드라이버를 Zadig로 설치해야 합니다.

## 스크립트

| 파일 | 설명 |
|---|---|
| `step0_prop_test.py` | 펌웨어 내장 프로펠러 테스트. 이륙하지 않고 모터 4개를 차례로 돌림 |
| `step0_hover_test.py` | 이륙 → 호버링 → 전진 → 착륙. 높이·거리·호버 시간을 옵션으로 지정 |

```
cf-env\Scripts\python step0_hover_test.py --height 0.3 --distance 0.5 --hover-time 3
```

`step0_hover_test.py`는 이륙 전에 아래를 점검하고, 하나라도 어긋나면 이륙하지 않습니다.

- Flow deck 인식 (`deck.bcFlow2`)
- 배터리 전압 (3.7V 이상)
- 기체 상태: 잠금/충돌/전복 여부 (`supervisor.info`)
- 지면 높이 센서: 바닥에서 80mm 미만이고 값이 변하는지

비행 중에는 높이, 모터 출력, 위치 추정(x, y), 방향(yaw)을 기록해서 비행 후 요약을 출력합니다.

## Step 0 결과

밝은 조명, 무늬 있는 바닥에서의 비행 로그 요약입니다. 높이와 위치는 모두 **드론의 추정값**이고, 실제 이동 거리는 줄자로 측정하지 못했습니다.

| 비행 | 목표 높이 | 최대 높이(센서) | 명령한 전진 | 추정 전진 거리 |
|---|---|---|---|---|
| 호버링 | 0.3 m | 0.33 m | - | (제자리, 추정 이동 약 6 cm) |
| 호버링 + 전진 | 0.3 m | 0.33 m | 0.5 m | 0.62 m |
| 호버링 + 전진 | 1.0 m | 1.07 m | 0.3 m | 0.68 m |

관찰:
- 높이 제어는 목표보다 2~7cm 더 올라가는 경향이 있음 (이륙 시 오버슈트)
- 호버링 중 방향(yaw)이 수 도에서 십수 도 틀어짐. Flow deck의 방향은 자이로 기반이라 drift가 있음
- 낮은 높이(0.3m)보다 높이 띄웠을 때(1m) 위치 추정이 더 정확했다는 것이 비행 중 관찰이었으나, 실측으로 검증하지 않았음

## 시행착오: 어두운 곳에서의 비행

<p align="center">
  <img src="docs/media/night_flight.gif" alt="어두운 곳에서의 비행 시도" width="240">
  <br>
  <sub>어두운 환경에서의 비행 시도 · <a href="docs/media/night_flight.mp4">mp4 보기</a></sub>
</p>

어두운 곳에서 비행했을 때 드론은 방향을 잃고 충돌해 추락했습니다.

- Flow deck의 광류 센서는 바닥 무늬를 촬영해 이동량을 계산하므로, 어두우면 위치 추정이 틀어짐
- 당시 로그: 높이 센서 값이 201mm로 고정, 위치 추정이 수 m 단위로 발산, 모터 출력이 최대치(65535)까지 포화, 기체가 충돌 상태(`isCrashed`)로 표시됨
- 밝은 곳, 무늬 있는 바닥에서는 같은 스크립트가 정상 동작함

이 경험을 반영해 이륙 전 점검(위 목록)을 추가했습니다.

## 비행 조건 (Flow deck)

- **충분히 밝은 환경**
- **무늬 있는 바닥** (단색, 반짝이는 바닥은 광류가 잘 안 잡힘)
- 유효 높이 0.2 ~ 3.0 m (Flow deck v2)
- 장애물을 감지하지 못하므로, 충돌 회피는 그래프의 공간 정보(복셀 점유 상태)에 의존함

## 참고 자료

- [Bitcraze Crazyflie 2.1](https://www.bitcraze.io/products/old-products/crazyflie-2-1/)
- [Bitcraze 문서: Motion Commander step-by-step](https://www.bitcraze.io/documentation/repository/crazyflie-lib-python/master/user-guides/sbs_motion_commander/)
