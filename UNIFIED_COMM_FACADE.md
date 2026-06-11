# stComm — Unified `Comm` Facade 설계 (작업 핸드오프)

> 상태: **stComm `Comm` facade 구현 완료(v2).** §5 수정 목록 전부 적용, §7 결정 확정.
> 빌드 통과 + 전 템플릿 경로 컴파일-온리 인스턴스화 검증됨(실행 검증은 GPU+mpirun 필요, 미실행).
> 범위: **stComm 레포만.** fullchipUSC 마이그레이션(Phase 2)은 이후 별도 작업.
> 구현 파일: `include/stComm/comm.h`(facade), `types.h`(`ReduceOp`), `request.h`(`getBackend()` 태그), `stComm.h`(include).

---

## 1. 목표

MPI(host)와 NCCL(device) 통신을 **하나의 `Comm` 객체**로 묶는다. 지금은
소비자가 `MPIComm`과 주입된 `NCCLComm`을 따로 들고 다님(fullchipUSC의
`USCSolver<MPIComm>` + `shared_ptr<NCCLComm>`). stComm은 fullchipUSC 전용이 아니라
**여러 프로젝트에서 재사용**할 라이브러리로 만들고 싶으므로, 사용성이 핵심.

## 2. 왜 지금 두 객체로 갈라져 있나 (근본 원인)

- **템플릿 멤버는 virtual이 될 수 없음.** 그래서 `CommBase`(추상 base)는
  `getRank/getSize/getBackend/barrier` 같은 비-템플릿 스칼라만 담은 껍데기가 됐고,
  정작 데이터를 옮기는 typed collective(`bcast<T>` 등)는 base로 묶이지 못함.
- **백엔드 능력 비대칭.** MAXLOC/exscan은 MPI에만, stream/group은 NCCL에만 존재.
  (단, NCCL에 없는 연산을 emulate하는 선례 있음: `NCCLComm::barrier`.)
- **메모리 공간이 타입이 아니라 포인터에 숨음.** `bcast(T*, …)`의 `T*`가
  host냐 device냐는 시그니처로 드러나지 않음.
- **번갈아가 아니라 동시에 씀.** solve() 한 루프에서 MPI MAXLOC + NCCL device
  bcast를 같이 호출 → "한 번에 한 모드" 모델이 애초에 안 맞음.

## 3. 설계 접근

**concrete facade `Comm`** — 추상 base가 아니라 두 백엔드를 *소유*하는 구체 클래스.
base가 아니므로 **메서드를 템플릿으로 유지** → 컴파일타임 타입 안전성 + zero-overhead
디스패치를 지키면서 단일 객체 제공.

- **`Space` 컴파일타임 태그**(`Host`/`Device`) + `if constexpr` 디스패치.
  - zero-overhead (런타임 분기 없음)
  - **타입 비대칭 컴파일 문제 해결**: host 전용 타입이 NCCL 분기를 인스턴스화하지
    않음(`NCCLTypeMap<T>` 미정의 에러 회피). 선택된 분기만 인스턴스화됨.
  - `Space`는 "NCCL 사용법"이 아니라 **사용자 자신의 데이터 위치**(내 버퍼가 GPU냐
    CPU냐)라서 백엔드 지식 누수가 아님 → 유지 OK.
- **`ReduceOp` enum**(`Sum/Max/Min/Prod`) — `MPI_Op`/`ncclRedOp_t`를 사용자에게 노출 안 함.

## 4. 캡슐화 원칙 (★ 사용자 요구로 확정)

> **핵심 요구: `Comm` 사용자는 NCCL/MPI의 구체적 사용법을 몰라도 되어야 한다.**
> **단, 두 가지는 예외로 노출되어야 한다(아래 4b, 4c).**

### 4a. 공통 경로에서 숨겨야 할 백엔드 specifics
- `MPI_Comm` (생성자에서 `MPI_COMM_WORLD` 타이핑 강요 X)
- `MPI_Op` (→ `ReduceOp`)
- `ncclUniqueId` / `ncclCommInitRank` 핸드셰이크 (생성자 내부로)
- CUDA stream, `cudaSetDevice` 등 (내부로)
- MPI 생명주기 `MPI_Init/Finalize` → `Comm::initialize/finalize`로 감싸 "MPI" 단어 노출 X

### 4b. ★ 하부 communicator "pool"은 노출되어야 함 (사용자 요구)
완전 은닉은 과함. 고급 사용자는 하부 백엔드 통신자(= "pool")에 접근할 수 있어야 함.
→ **제어된 escape hatch 제공**: `mpi()` → `MPIComm&`, `nccl()` → `NCCLComm&`
(혹은 raw `MPI_Comm` / `ncclComm_t`까지). v1 초안엔 `mpi()/nccl()`가 있었음 — **유지/강화**.
- 미해결: "pool"의 정확한 의미 확정 필요. (a) `MPIComm&`/`NCCLComm&` 참조, (b) raw
  핸들(`MPI_Comm`, `ncclComm_t`), (c) 둘 다 — 중 무엇을 노출할지. **새 세션에서 사용자 확인.**

### 4c. ★ Request sync는 MPI/NCCL이 달라서 각각 노출 필요 (사용자 요구)
완료 동기화 메커니즘이 백엔드마다 다름:
- MPI: `MPI_Wait/MPI_Test`(on `MPI_Request`) — `MPIRequest::getHandle()`
- NCCL: `cudaStreamSynchronize`(on stream) — `NCCLRequest::getStream()`

현재 `Request` base의 `wait()/test()`가 이미 다형적으로 추상화함(공통 경로엔 충분).
그러나 **백엔드별 sync 핸들을 꺼낼 방법**이 필요(예: NCCL stream에 커널 추가 enqueue 후
한 번에 sync, 또는 device-resident 루프에서 stream 재사용). 설계 옵션:

- **Option 1 — Space 기반 downcast 헬퍼**: `bcast<Space::Device>`를 쓴 호출자는
  반환이 `NCCLRequest`임을 앎 → `static_cast<NCCLRequest*>(req.get())->getStream()`.
  최소 변경, 약간 unsafe.
- **Option 2 — Space별 반환 타입**: `bcast<Sp>`가 `if constexpr`로 device면
  `NCCLRequestPtr`, host면 `MPIRequestPtr` 반환. 가장 타입안전, 제네릭 코드 복잡↑.
- **Option 3 — `Comm`에 stream/pool 접근자**: `Comm::deviceStream()` 등으로 NCCL
  stream을 직접 노출 + 완료는 base `wait()`. device-resident 루프에 가장 유용.

→ **추천: Option 3(stream 접근자) + 공통 `wait()` 유지**, 필요 시 Option 2 병행. 새 세션에서 확정.

## 5. v1 초안(`include/stComm/comm.h`)에서 고친 것 — ✅ 전부 적용 (v2)

- [x] **exscan이 `MPI_Op` 노출** → `ReduceOp` enum(`types.h`) + 내부 `detail::to_mpi_op()` 매핑으로 교체.
- [x] **생성자가 `MPI_Comm`을 전면에 요구** → host는 `Comm()`(기본 `MPI_COMM_WORLD`),
      device는 `static Comm onDevice(int device_id, MPI_Comm=MPI_COMM_WORLD)` 팩토리로.
      device 부트스트랩 생성자는 **private**. `onDevice`는 prvalue 반환 → C++17 보장 복사 생략.
- [x] **생명주기 래퍼 추가**: `static initialize(int*,char***)` / `static finalize()` → `MPIComm` 포워딩.
- [x] **pool 접근자 명문화**(4b): `mpi()→MPIComm&` / `nccl()→NCCLComm&` (래퍼 참조만; 결정 §7.1).
      raw 핸들이 필요하면 백엔드 객체의 `getHandle()`로 내려감 — specifics는 백엔드가 노출.
- [x] **Request 백엔드별 sync 노출**(4c): **Option 2** 채택 — `bcast<Device>` 등이 concrete
      `shared_ptr<NCCLRequest>` 반환(→ 캐스팅 없이 `getStream()`), host는 base `RequestPtr`.
      추가로 base `Request`에 `getBackend()` 태그 → type-erased 보관 시 안전 식별/다운캐스트.
- [x] device 분기에 `assert(nccl_)` 가드.
- [x] **추가 결정**: 두 백엔드 모두 `unique_ptr`로 보유(대칭). `mpi_` 항상 존재, `nccl_` null ⇒ host-only.

## 6. 제안 API 스케치 (목표 형태)

```cpp
namespace stComm {

enum class Space    { Host, Device };
enum class ReduceOp { Sum, Max, Min, Prod };

class Comm {
public:
  // ---- 생명주기 (MPI init/finalize 은닉) ----
  static void initialize(int* argc, char*** argv);   // → MPIComm::initialize
  static void finalize();                            // → MPIComm::finalize

  // ---- 생성 ----
  explicit Comm(MPI_Comm comm = MPI_COMM_WORLD);     // host-only
  static Comm onDevice(int device_id,                // host+device, NCCL 부트스트랩 내부
                       MPI_Comm comm = MPI_COMM_WORLD);

  Comm(const Comm&) = delete;                        // 통신자 핸들: 비복사/비이동
  Comm& operator=(const Comm&) = delete;

  int  getRank()   const;
  int  getSize()   const;
  bool hasDevice() const;
  void barrier();

  // ---- 공간-선택 collective (Sp로 백엔드 결정) ----
  template<Space Sp, typename T> RequestPtr bcast(T*, std::size_t, int root);
  template<Space Sp, typename T> RequestPtr allgatherv(const T*, int, T*, const int*);
  template<Space Sp, typename T> RequestPtr alltoallv(const T*, const int*, T*, const int*);

  // ---- host control-plane (NCCL 프리미티브 없음; 지금은 host 전용) ----
  template<typename T> std::pair<T,int> allreduceMaxloc(T value);
  template<typename T> T exscan(T value, ReduceOp op = ReduceOp::Sum);

  // ---- 4b: pool 노출 (escape hatch) ----  [형태 확정 필요]
  MPIComm&  mpi();
  NCCLComm& nccl();
  // 또는 raw: MPI_Comm mpiHandle(); ncclComm_t ncclHandle();

  // ---- 4c: 백엔드별 sync 노출 ----  [형태 확정 필요]
  // 예) cudaStream_t deviceStream();  // NCCL stream 직접 접근

private:
  Comm(MPI_Comm comm, int device_id);   // onDevice 전용; NCCL 핸드셰이크 내부
  MPIComm                   mpi_;
  std::unique_ptr<NCCLComm> nccl_;       // null ⇒ host-only
};

} // namespace stComm
```
- `stComm.h`에 `#include "stComm/comm.h"` 한 줄 추가.
- **헤더 온리** → `src/stComm/CMakeLists.txt` 변경 불필요.

## 7. 미해결 결정 사항 (새 세션에서 확정)

1. ~~**"pool"의 노출 형태**~~ → ✅ **래퍼 참조만**(`mpi()/nccl()`). raw 핸들은 백엔드 `getHandle()`로. (4b)
2. ~~**Request 백엔드별 sync 노출 방식**~~ → ✅ **Option 2**(Space별 반환 타입) + base `Request::getBackend()` 태그. (4c)
3. ~~**`ReduceOp` 위치**~~ → ✅ **`types.h`**(재사용성).
4. **device_id 자동 선택**(`rank % numGPUs`) 제공 여부 — 지금은 명시 인자, 자동은 YAGNI. (미정, 보류)
5. **NCCL MAXLOC/exscan device 변형** — 후속(별도). 라이브러리 완전성 차원에서 언젠가 구현하되
   AllGather+로컬 reduce 방식(타입무관·정확) 권장, 비트팩 ncclMax는 최적화. 단 이 박스(2×L4,
   no-NVLink)에선 small-msg latency 때문에 host보다 느림 → **L4에선 채택 X, NVLink에서 의미**.

## 8. 구현/검증 메모

- 빌드: `cd ~/tickets/stComm && ./build.sh` → 설치: `./install.sh ~/install/stComm`.
- 툴체인 확인됨: `nvcc`(/usr/local/cuda), `mpicxx`(/usr/local/bin), `nccl.h`(/usr/include).
  CUDA arch 기본값 `75;80;89`에 L4(89) 포함. C++ standard: CUDA 17 → `if constexpr` OK.
- ✅ 빌드 통과 + 전 템플릿 경로 컴파일-온리 인스턴스화 검증(host/device 모두).
- ✅ **facade 단위테스트 추가됨**: `tests/test_comm.cpp` (CMake에 등록). host `bcast/allgatherv/
  alltoallv` + `allreduceMaxloc` + `exscan(ReduceOp)` + `Request::getBackend()` 태그를 해석적
  기대값과 대조. device `bcast<Space::Device>`는 concrete `NCCLRequest`/`getStream()`까지 검증하되
  랭크별 GPU 부족 시 `GTEST_SKIP`.
- ✅ **실행 결과(이 박스, 단일 MX450)**: `mpirun -n 2`에서 host 6개 PASS, device 1개 SKIP, 기존 29 PASS 회귀 없음.
  ⚠️ **device 경로 런타임 검증은 2-GPU 박스 필요**(`NCCL_P2P_DISABLE=1`) — 아직 미실행.

## 9. 스코프 경계

- **지금**: stComm `Comm` facade만. (이 문서.)
- **Phase 2 (후속)**: fullchipUSC를 `Comm`으로 전환 — `USCSolver` de-template화,
  `main.cpp`/`tests`의 수동 NCCL 부트스트랩을 `Comm::onDevice`로 축소, 27 ctest + 4스케일
  `selected` bit-identical(353/5378/14889/35824) 검증. **이 작업은 별도 세션.**
- **후속**: NCCL device MAXLOC/exscan (§7.5).
