# Windows Gen1 client validation — Alpha.1, 2026-11-18

This record describes the original Alpha.1 build. See the
[subsequent acceptance record](windows-client-acceptance-20261118.md) for the installation,
upgrade, UI and live-provider results from later builds.

## Delivery and baseline

- Status: **local development Alpha; not a public Windows release**.
- Version: `1.1.0-alpha.1`, Windows x64, NSIS current-user installer.
- Public Web parent: `cace08540495604605623aae843b9a21525df6bd` in
  `Moonaria123/Deep-Research-Studio`. The parent was verified against remote `main`.
- New branch: `codex/gen1-windows-client`.
- Worktree: `D:\deepresearch-gen1-windows`.
- The existing Web checkout and `D:\deepresearch` Gen2 checkout retain their
  pre-existing changes. No commit, push, release publication or production data
  migration was performed.

The client bundles the existing Gen1 UI and a frozen local Python API. Desktop
uses HashRouter, an exclusive loopback listener, a separate profile, DPAPI
provider secrets and in-process jobs. The Rust shell owns the entire child
process tree. Interrupted job records are reconciled on the next desktop start;
completed and deliberately paused runs are preserved.

## Delivered installer

`release-artifacts/windows/1.1.0-alpha.1/Deep Research Studio_1.1.0-alpha.1_x64-setup.exe`

| Item | Value |
| --- | --- |
| Installer bytes | 33,092,201 |
| Installer SHA-256 | `328290dc74c53480762389e39b8a5c7f26da65fa755abbad387ad95b538fda4e` |
| Authenticode | NotSigned |
| Bundled sidecar bytes | 30,400,505 |
| Sidecar SHA-256 | `6e8aaf1e222bea16ae888767d8db4bc6e07af6fbbe1f7b53de2e609f1e8f97be` |
| Sidecar source inputs | 119 allowlisted files |
| Sidecar source SHA-256 | `93d7fb0d7c22f3e47b4c05466fa5561cb5f9b1a9caa99984f0045963f4a94de1` |

The delivery directory also contains `SHA256SUMS.txt`, `BUILD-INFO.json`, the
sidecar manifest, usage instructions and this validation record. Build products,
synthetic test profiles and local logs are ignored by Git.

## Executed verification

Counts below overlap and must not be added together.

| Verification | Result | Local evidence |
| --- | --- | --- |
| Final `pytest tests/desktop -q` | 27 passed, 0 skipped | `.tmp/windows-desktop-final-tests.xml` |
| Runtime, API CORS, Gen1 Web boundary, vault and document-import regression | 77 passed, 1 skipped | `.tmp/windows-runtime-tests.xml` |
| Final runtime, packaging and document-import regression after parser stdin fix | 54 passed | `.tmp/windows-parser-final-tests.xml` |
| Library, Memory, API-origin and runtime frontend tests | 49 passed in 4 files | Agent command output, exit 0 |
| Frontend TypeScript lint | Passed | Agent command output, exit 0 |
| Web, Desktop and Demo frontend builds | All passed | Agent command output, exit 0 |
| Final Rust `cargo test` | 5 passed | Agent command output, exit 0 |
| Rust `cargo clippy --tests -- -D warnings` and `cargo fmt --check` | Passed | Agent command output, exit 0 |
| NSIS build | Passed, 1 installer | `.tmp/windows-installer-build-final.log` |
| Frozen sidecar integration smoke | 8 checks passed; cold start 4.797 seconds | `.tmp/windows-packaged-smoke-final/result.json` |
| Native client startup | Home screen verified through accessibility tree before final timeout fix; final shell API readiness checked separately | Native CUA observation and `.tmp/windows-native-smoke.json` |
| Abrupt native parent exit | Both owned sidecar processes exited; port 18789 closed | `.tmp/windows-native-smoke.json` |

The one skipped test requires a Linux/Docker root key and is not a Windows DPAPI
test. The Windows DPAPI round trip passed in the real frozen executable. Frontend
builds report the existing Vite warning about chunks over 500 kB.

The frozen smoke exercised real HTTP requests, duplicate-start rejection before
opening a second profile, exact CORS/Host rejection, synthetic credential save
and decrypt without a provider probe, DOCX parsing through the packaged child,
session persistence after restart, graceful stdin shutdown and parent-pipe EOF.
No paid provider or production request was made. A disabled provider used only a
synthetic secret; the secret is absent from the result record and plaintext
configuration.

The initial native test launched the release shell with its bundled sidecar, observed
the four mode controls and a connected fresh profile at `http://tauri.localhost/`,
then terminated only that owned shell process to verify Windows Job cleanup.
After the final startup-timeout fix, the shell was rebuilt and API readiness plus
whole-tree cleanup were checked again with a separate synthetic profile. The
frozen API source and hash did not change. The installer itself was not executed.
Generated NSIS instructions include the sidecar and
LICENSE/NOTICE/THIRD_PARTY_NOTICES; this static check is not an installation test.

### Reproducible commands

From the repository root, after the build prerequisites in [the guide](windows-client.md):

```powershell
& .tmp/desktop-sidecar-env/Scripts/python.exe -m pytest tests/desktop -q `
  --junitxml=.tmp/windows-desktop-final-tests.xml
& .tmp/desktop-sidecar-env/Scripts/python.exe -m pytest `
  tests/desktop/test_runtime.py tests/baseline/test_api_cors.py `
  tests/baseline/test_gen1_web_boundary.py tests/baseline/test_secret_vault_config.py `
  tests/baseline/test_t56_local_document_import.py -q
& .tmp/desktop-sidecar-env/Scripts/python.exe -m pytest `
  tests/desktop/test_runtime.py tests/desktop/test_packaging.py `
  tests/baseline/test_t56_local_document_import.py -q
& .tmp/desktop-sidecar-env/Scripts/python.exe scripts/verify_windows_sidecar.py `
  --executable apps/desktop/src-tauri/binaries/deepresearch-api-x86_64-pc-windows-msvc.exe `
  --output-dir .tmp/windows-packaged-smoke-new
npm --prefix apps/web test -- --run src/pages/Library.test.tsx `
  src/pages/Memory.test.tsx src/lib/apiOrigin.test.ts src/lib/runtime.test.ts
npm --prefix apps/web run lint
npm --prefix apps/web run build
npm --prefix apps/web run build:desktop
npm --prefix apps/web run build:demo
cargo test --manifest-path apps/desktop/src-tauri/Cargo.toml --locked
cargo clippy --manifest-path apps/desktop/src-tauri/Cargo.toml --tests -- -D warnings
cargo fmt --manifest-path apps/desktop/src-tauri/Cargo.toml --check
npm --prefix apps/desktop run build -- --bundles nsis
git diff --check
```

During development, the first test environment lacked `httpx2`; the existing
development requirements were installed into the isolated desktop environment
and the suite passed. A real parser timeout exposed inherited stdin from the
shell lifetime pipe; setting the parser child's stdin to `DEVNULL` fixed it,
and source plus frozen document imports passed. The first NSIS attempt was
blocked by a file lock from the simultaneously running native test; after owned
process cleanup, the complete final installer build passed.

Independent review covered 25 core implementation, configuration, packaging,
test and validation files. It found a READY/45-second-timeout race, which was
fixed with one atomic `STARTING -> READY / FAILED` transition. Three additional
Rust tests cover competing threads and both event orders. The reviewer then
confirmed the finding was closed and reported no remaining critical, high or
medium findings. This code review does not replace the manual release gates below.

## Remaining release gates

- Native UI click coverage is incomplete: the desktop control tool could read
  the accessibility tree but failed to activate the captured window or resolve
  its click geometry. Settings flows, external-link handling, exports and normal
  window-close behavior have not received end-to-end native interaction acceptance.
- A clean Windows VM/Sandbox installation, missing-WebView2 bootstrap, upgrade
  and uninstall have **not run**. Tauri's generated uninstaller offers an explicit
  app-data deletion option; data retention must be verified with that option off.
- The installer is unsigned. No signing certificate was requested or used.
- Real-provider research, workload cancellation/recovery under paid requests,
  citation-quality acceptance and target-device performance have **not run**.
  The measured 4.797-second value is API readiness for this developer machine,
  not end-to-end GUI startup or a research latency acceptance result.
- The native fresh profile remains at
  `%LOCALAPPDATA%\app.deepresearch.studio.gen1\backend`. No Web/Gen2 data was copied
  into it. Local test evidence under `.tmp` contains synthetic test data only.

The artifact is suitable for local Alpha review. These results do not establish
public-release readiness or close inherited Gen1 quality/performance gates.
