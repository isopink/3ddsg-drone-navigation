# 협업 규칙

세 명이 같은 코드를 동시에 고쳐도 서로 덮어쓰지 않도록 정한 규칙입니다.

## 역할

| 사람 | 담당 | 레포에서 하는 일 |
|---|---|---|
| 도현 | 3DDSG 연구 리드 (논문 1저자), 그래프·Neo4j | 그래프 내보내기, 그래프 관련 PR 리뷰 |
| 채희 | BIM (IFC): 방·문·벽 형상, 좌표계 | IFC 관련 작업 |
| 석민 | 드론: 경로 계획, 비행. **레포 관리자** | 드론 쪽 작업, **모든 PR merge** |

- **`main`에 merge하는 사람은 석민 한 명입니다.** 다른 사람은 PR까지만 올립니다.
- 석민의 PR은 가능하면 도현이 리뷰합니다. 연구 내용과 어긋나지 않는지 확인하기 위해서입니다.

## 브랜치

브랜치는 원본(`main`)을 건드리지 않고 작업하는 복사본입니다. 작업이 끝나면 PR로 `main`에 합칩니다.

```
main                     항상 작동하는 상태. 직접 push 금지
├─ feat/graph-export     예: 도현
├─ feat/bim-ifc          예: 채희
└─ feat/real-flight      예: 석민
```

**이름 규칙:** `종류/작업-내용` (사람 이름 말고 작업 내용으로)

| 종류 | 언제 | 예시 |
|---|---|---|
| `feat/` | 새 기능 | `feat/neo4j-source`, `feat/dynamic-objects` |
| `fix/` | 버그 수정 | `fix/flow-deck-check` |
| `docs/` | 문서만 수정 | `docs/graph-format` |
| `exp/` | 실험 (merge 안 할 수도 있음) | `exp/multiranger-test` |

- **작업 하나에 브랜치 하나.** 끝나면 merge하고 지웁니다. 다음 작업은 새 브랜치로 시작합니다.
- **브랜치는 짧게.** 며칠 안에 PR을 올립니다. 오래 두면 `main`과 차이가 커져서 충돌이 납니다.

## 작업 순서

```
# 1. 최신 main에서 시작
git switch main
git pull

# 2. 브랜치 만들기
git switch -c feat/작업-내용

# 3. 작업하고 커밋 (여러 번 해도 됨)
git add <파일>
git commit -m "무엇을 왜 바꿨는지 한 줄"

# 4. 테스트 (드론 없이 돌아감, 약 2분)
cf-env\Scripts\python -m pytest

# 5. GitHub에 올리기
git push -u origin feat/작업-내용
```

6. GitHub에서 **Pull Request**를 엽니다 (base: `main`).
7. 석민이 확인하고 merge합니다. 수정 요청이 오면 같은 브랜치에 커밋해서 다시 push하면 PR에 자동으로 반영됩니다.

**작업 중에 `main`이 바뀌었다면** (다른 사람 PR이 merge됨):

```
git switch main
git pull
git switch feat/작업-내용
git merge main
```

## PR 올리기 전 확인

- [ ] `pytest`가 전부 통과한다
- [ ] 연구실 데이터가 들어가지 않았다 (아래 참고)
- [ ] `.env`나 API 키가 들어가지 않았다
- [ ] PR 설명에 **무엇을, 왜** 바꿨는지와 **어떻게 확인했는지**를 적었다

## 절대 커밋하지 않는 것

- **연구실 데이터**: 그래프, IFC, 점군(`.ply`, `.pcd`), Neo4j 덤프. `data/` 폴더에 두면 자동으로 제외됩니다. 데이터는 GitHub 대신 연구실 공유 드라이브로 주고받습니다.
- **비밀 정보**: `.env` (LLM API 키 등). 형식은 `.env.example`을 참고합니다.
- **가상환경·캐시**: `cf-env/`, `cache/`, `runs/`

모두 `.gitignore`에 들어 있지만, `git add -A` 전에 `git status`로 한 번 확인합니다.

## 그래프 형식은 함께 정한다

그래프 쪽(도현·채희)과 드론 쪽(석민)은 [`docs/graph_format.md`](docs/graph_format.md) 형식(`graph.json` + `voxels.npz`)으로만 데이터를 주고받습니다. 이 형식을 지키면 서로 상대 코드를 몰라도 됩니다.

- 형식을 바꾸려면 `docs/graph_format.md`를 수정하는 PR을 따로 올리고, **세 명 모두 확인한 뒤** merge합니다.
- 내보낸 그래프는 `cf-env\Scripts\python scripts\graph_tools.py validate <폴더>`로 검증할 수 있습니다.

## 드론 비행

실기체 비행 전에는 README의 [Before flying the real drone](README.md#before-flying-the-real-drone)을 확인합니다. 특히 **밝은 곳, 무늬 있는 바닥**에서 비행하고, 반드시 시뮬레이션을 먼저 돌립니다.
