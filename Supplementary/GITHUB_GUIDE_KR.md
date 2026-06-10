# GitHub 레포지토리 & GitHub Pages 완전 가이드

> 이 문서는 Git/GitHub를 처음 사용하는 사람을 위한 단계별 안내입니다.
> 대상: macOS 사용자 (Windows 사용자는 별도 표시된 부분 참고)

---

## 전체 흐름 미리 보기

```
[STEP 0] 사전 준비 — Git 설치, GitHub 가입
    ↓
[STEP 1] GitHub에 빈 레포지토리 생성
    ↓
[STEP 2] 내 컴퓨터에 프로젝트 폴더 준비
    ↓
[STEP 3] Git 초기화 & 첫 번째 커밋
    ↓
[STEP 4] GitHub에 업로드 (push)
    ↓
[STEP 5] GitHub Pages 활성화 → 웹사이트 배포
    ↓
[STEP 6] 이후 작업 — 수정, 커밋, 푸시 반복
```

예상 소요 시간: 처음이면 약 30~60분

---

## STEP 0: 사전 준비

### 0-1. Git 설치 확인

터미널(Terminal)을 열고 아래 명령어를 입력합니다.

```bash
git --version
```

**결과가 나온다면** (예: `git version 2.39.0`) → 이미 설치됨, 다음으로.

**"command not found"가 나온다면** → 설치 필요:

- **macOS**: 터미널에 `git --version`을 치면 자동으로 Xcode Command Line Tools 설치를 제안합니다. "Install"을 클릭하세요.
- **Windows**: https://git-scm.com/download/win 에서 다운로드 후 설치. 설치 중 옵션은 모두 기본값으로 진행해도 됩니다. 설치 후 "Git Bash" 프로그램을 사용합니다.

### 0-2. GitHub 계정 만들기

1. https://github.com 에 접속
2. "Sign up" 클릭
3. 이메일, 비밀번호, 사용자명(username) 입력
   - **사용자명은 신중하게 정하세요** — GitHub Pages URL이 `https://사용자명.github.io/action-irt`가 됩니다.
   - 예: 사용자명이 `junyeong-park`이면 → `https://junyeong-park.github.io/action-irt`
4. 이메일 인증 완료

### 0-3. Git에 내 정보 등록

터미널에서 아래 두 줄을 입력합니다 (한 번만 하면 됩니다):

```bash
git config --global user.name "Junyeong Park"
git config --global user.email "your_email@example.com"
```

> ⚠️ 여기서 입력하는 이메일은 GitHub 가입에 사용한 이메일과 동일해야 합니다.

---

## STEP 1: GitHub에 빈 레포지토리 생성

1. GitHub에 로그인한 상태에서 오른쪽 상단 **"+"** 버튼 → **"New repository"** 클릭

2. 다음과 같이 설정:

   | 항목 | 입력 값 |
   |------|---------|
   | Repository name | `action-irt` |
   | Description | `Code and supplementary materials for Action-IRT paper` |
   | Public / Private | **Public** (논문 코드 공개용) |
   | Initialize this repository | **아무것도 체크하지 마세요** ⚠️ |

   > ⚠️ 중요: "Add a README file", "Add .gitignore", "Choose a license" 모두 **체크 해제** 상태로 두세요. 우리가 이미 이 파일들을 만들어 두었기 때문입니다. 여기서 체크하면 충돌이 발생합니다.

3. **"Create repository"** 클릭

4. 생성 후 나오는 화면에서 HTTPS 주소를 확인합니다:
   ```
   https://github.com/사용자명/action-irt.git
   ```
   이 주소를 메모해 두세요 (곧 사용합니다).

---

## STEP 2: 내 컴퓨터에 프로젝트 폴더 준비

### 2-1. 다운로드한 파일 압축 해제

이전에 제가 만들어 드린 `action-irt-repo.tar.gz` 파일을 다운로드했다면:

```bash
# 원하는 위치로 이동 (예: 바탕화면)
cd ~/Desktop

# 압축 해제
tar xzf action-irt-repo.tar.gz
```

이렇게 하면 `~/Desktop/action-irt/` 폴더가 생깁니다.

### 2-2. 폴더 구조 확인

```bash
cd ~/Desktop/action-irt
ls -la
```

다음과 같은 파일들이 보여야 합니다:

```
.gitignore
LICENSE
README.md
SUBMISSION_CHECKLIST.md
code/
config.yaml
docs/
manuscript/
analysis/
supplementary/
setup_r_env.R
```

### 2-3. 내 정보로 수정해야 하는 파일들

아래 파일들에서 `<username>`을 **본인의 GitHub 사용자명**으로 바꿔야 합니다.

**(1) `docs/_config.yml`** — GitHub Pages 설정

텍스트 편집기(VS Code, Sublime Text, 메모장 등)로 열어서:

```yaml
# 변경 전
url: "https://<username>.github.io"

# 변경 후 (예시)
url: "https://junyeong-park.github.io"
```

**(2) `README.md`** — 메인 문서

파일 안에서 `<username>`을 검색(Ctrl+F 또는 Cmd+F)하여 모두 교체합니다:

```markdown
# 변경 전
git clone https://github.com/<username>/action-irt.git

# 변경 후
git clone https://github.com/junyeong-park/action-irt.git
```

**(3) `docs/index.md`** — 웹사이트 메인 페이지

동일하게 `<username>`을 검색하여 교체합니다.

> 💡 팁: VS Code를 사용한다면, 폴더 전체를 열고 Ctrl+Shift+H (Cmd+Shift+H)로 전체 찾기/바꾸기를 하면 한 번에 처리됩니다.

---

## STEP 3: Git 초기화 & 첫 번째 커밋

터미널에서 프로젝트 폴더로 이동한 상태에서 시작합니다.

```bash
cd ~/Desktop/action-irt
```

### 3-1. Git 저장소 초기화

```bash
git init
```

출력: `Initialized empty Git repository in /Users/.../action-irt/.git/`

> 이 명령은 폴더 안에 숨겨진 `.git` 폴더를 만들어서, 이 폴더를 Git이 관리하도록 설정합니다.

### 3-2. 모든 파일을 스테이징 (추가 대기열에 올리기)

```bash
git add .
```

> `git add .`의 `.`은 "현재 폴더의 모든 파일"을 의미합니다. `.gitignore`에 명시된 파일들은 자동으로 제외됩니다.

### 3-3. 스테이징 상태 확인 (선택사항)

```bash
git status
```

초록색으로 표시된 파일들이 커밋 대기 중인 파일입니다. 빨간색 파일은 아직 추가되지 않은 파일입니다.

### 3-4. 첫 번째 커밋

```bash
git commit -m "Initial commit: repository structure and documentation"
```

> `-m` 뒤의 문자열은 "커밋 메시지"입니다. 이 변경사항이 무엇인지 한 줄로 설명합니다.

출력 예시:
```
[main (root-commit) a1b2c3d] Initial commit: repository structure and documentation
 22 files changed, 850 insertions(+)
 create mode 100644 .gitignore
 create mode 100644 LICENSE
 ...
```

---

## STEP 4: GitHub에 업로드 (push)

### 4-1. 원격 저장소 연결

STEP 1에서 메모한 주소를 사용합니다:

```bash
git remote add origin https://github.com/사용자명/action-irt.git
```

> 이 명령은 "내 컴퓨터의 Git 저장소"와 "GitHub의 원격 저장소"를 연결합니다. `origin`은 원격 저장소의 별명입니다.

### 4-2. 브랜치 이름 설정

```bash
git branch -M main
```

> 기본 브랜치 이름을 `main`으로 설정합니다.

### 4-3. 업로드 (push)

```bash
git push -u origin main
```

> 이때 GitHub 로그인을 요구할 수 있습니다.

**인증 방법 (처음 push할 때 한 번만):**

GitHub는 2021년부터 비밀번호 대신 **Personal Access Token (PAT)**을 사용합니다.

1. GitHub 웹사이트 → 오른쪽 상단 프로필 사진 클릭 → **Settings**
2. 왼쪽 사이드바 맨 아래 → **Developer settings**
3. **Personal access tokens** → **Tokens (classic)** → **Generate new token (classic)**
4. 설정:
   - Note: `action-irt-push` (아무 이름)
   - Expiration: 90 days (또는 원하는 기간)
   - Scopes: **repo** 체크 ✓
5. **Generate token** 클릭
6. 나오는 토큰 문자열을 **지금 복사** (페이지를 벗어나면 다시 볼 수 없습니다)

터미널에서 비밀번호를 물으면:
- Username: GitHub 사용자명
- Password: **위에서 복사한 토큰** 붙여넣기 (화면에 표시되지 않지만 입력되고 있습니다)

**push 성공 출력:**
```
Enumerating objects: 35, done.
Counting objects: 100% (35/35), done.
...
To https://github.com/사용자명/action-irt.git
 * [new branch]      main -> main
Branch 'main' set up to track remote branch 'main' from 'origin'.
```

### 4-4. 확인

브라우저에서 `https://github.com/사용자명/action-irt` 에 접속하면 업로드된 파일들이 보입니다. README.md가 자동으로 렌더링되어 표시됩니다.

---

## STEP 5: GitHub Pages 활성화

이 단계에서 `docs/` 폴더의 내용이 웹사이트로 배포됩니다.

### 5-1. Settings 진입

1. GitHub 레포지토리 페이지 (`https://github.com/사용자명/action-irt`)
2. 상단 탭에서 **Settings** 클릭 (⚙️ 아이콘)

### 5-2. Pages 설정

1. 왼쪽 사이드바에서 **Pages** 클릭
2. **Source** 섹션에서:
   - Branch: **main** 선택
   - Folder: **`/docs`** 선택
3. **Save** 클릭

### 5-3. 배포 확인 (1~3분 소요)

Save를 누르면 GitHub가 자동으로 웹사이트를 빌드합니다.

1. 1~3분 기다린 후 페이지를 새로고침
2. 상단에 초록색 배너가 나타납니다:
   ```
   ✅ Your site is live at https://사용자명.github.io/action-irt/
   ```
3. 해당 URL을 클릭하면 웹사이트가 열립니다

### 5-4. 웹사이트 동작 확인

다음 페이지들이 모두 접근 가능한지 확인합니다:

| URL | 내용 |
|-----|------|
| `https://사용자명.github.io/action-irt/` | 메인 페이지 (초록, 프레임워크 요약) |
| `.../action-irt/pages/method.html` | 방법론 상세 |
| `.../action-irt/pages/results.html` | 실증 결과 |
| `.../action-irt/pages/simulation.html` | 시뮬레이션 |
| `.../action-irt/pages/appendix.html` | 부록 |

> ⚠️ 만약 페이지가 안 보이거나 스타일이 깨진다면:
> - `docs/_config.yml`에서 `baseurl`이 `/action-irt`로 정확히 설정되어 있는지 확인
> - `url`이 `https://사용자명.github.io`로 정확히 설정되어 있는지 확인
> - GitHub 레포지토리 이름이 정확히 `action-irt`인지 확인

---

## STEP 6: 이후 작업 — 수정, 커밋, 푸시 반복

GitHub에 올린 후에도 파일을 수정하고 업데이트할 수 있습니다. 기본 워크플로우는 아래 3단계 반복입니다:

```
수정(edit) → 커밋(commit) → 푸시(push)
```

### 6-1. 파일 수정

텍스트 편집기에서 파일을 수정합니다. 예를 들어, 새로운 분석 코드를 추가하거나, 문서를 업데이트하거나, 그림 파일을 추가하는 경우입니다.

### 6-2. 변경사항 확인

```bash
cd ~/Desktop/action-irt
git status
```

수정된 파일이 빨간색으로 표시됩니다.

### 6-3. 변경사항 스테이징 & 커밋

```bash
# 모든 변경사항 추가
git add .

# 커밋 (메시지는 변경 내용을 간결하게 설명)
git commit -m "Add convergence diagnostic figures"
```

**좋은 커밋 메시지 예시:**

| 상황 | 커밋 메시지 |
|------|------------|
| 코드 추가 | `Add LSTM autoencoder training script` |
| 문서 수정 | `Update simulation results in docs` |
| 그림 추가 | `Add trace plots for alpha parameters` |
| 버그 수정 | `Fix robust scaling in run_mcmc.R` |
| 논문 수정 | `Revise Section 4.3 model specification` |

### 6-4. GitHub에 업로드

```bash
git push
```

> 첫 push 이후에는 `git push`만 입력하면 됩니다 (`-u origin main` 불필요).

### 6-5. 자동 반영

push하면 GitHub Pages도 자동으로 다시 빌드됩니다. `docs/` 폴더의 파일을 수정한 경우, 1~3분 후 웹사이트에 반영됩니다.

---

## 자주 하는 실수와 해결법

### "rejected — failed to push"

```
! [rejected]        main -> main (fetch first)
```

**원인:** GitHub에 있는 내용과 내 컴퓨터의 내용이 다릅니다 (예: GitHub 웹에서 직접 파일을 수정한 경우).

**해결:**
```bash
git pull --rebase origin main
git push
```

### "nothing to commit, working tree clean"

**원인:** 변경사항이 없거나, 이미 커밋했습니다. 정상입니다.

### GitHub Pages가 404를 보여줌

**확인 사항:**
1. Settings → Pages에서 Branch가 `main`, Folder가 `/docs`로 설정되어 있는지
2. `docs/index.md` 파일이 존재하는지
3. `docs/_config.yml`의 `baseurl`이 `/action-irt`인지

### 실수로 데이터 파일을 올려버린 경우

`.gitignore`에 패턴이 있더라도, 이미 `git add`로 추가한 파일은 무시되지 않습니다.

```bash
# 추적에서 제거 (파일 자체는 삭제하지 않음)
git rm --cached data/sensitive_file.csv
git commit -m "Remove accidentally tracked data file"
git push
```

---

## 요약: 핵심 명령어 모음

```bash
# 최초 1회
git init
git remote add origin https://github.com/사용자명/action-irt.git
git branch -M main
git add .
git commit -m "Initial commit"
git push -u origin main

# 이후 반복
git add .
git commit -m "설명 메시지"
git push

# 상태 확인
git status          # 현재 변경 상태
git log --oneline   # 커밋 이력 보기
git diff            # 변경 내용 상세 보기
```

---

## 다음에 해야 할 일 (우선순위순)

1. ✅ **STEP 0~5 완료** — 레포지토리 생성 & Pages 배포
2. 📝 **`docs/_config.yml`과 README의 `<username>` 교체**
3. 📂 **나머지 코드 파일 추가** — Python 코드 (embedding, LSTM AE)를 `code/02_embedding/`, `code/03_dimension_reduction/`에 배치
4. 📂 **`manuscript/`에 LaTeX 파일 배치** — `main.tex`, `reference.bib`
5. 📂 **`supplementary/llm_prompts/`에 LLM 프롬프트 텍스트 저장**
6. 📊 **분석 그림 추가** — trace plot, ΔE 분포 등을 `analysis/figures/`에 저장
7. 🔧 **`SUBMISSION_CHECKLIST.md` 항목 채우기** — 펀딩, ORCID 등
