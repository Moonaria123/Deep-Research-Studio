# Windows desktop client (Gen1 development)

This branch adds a Windows x64 client to the published Gen1 Web application. Its
parent is public Web commit `cace08540495604605623aae843b9a21525df6bd`.
The desktop package is `1.1.0-alpha.3`, a local development build. It does not
replace the published Web `v1.0.0` release.

See the [Windows acceptance record](windows-client-acceptance-20261118.md) for
the installer identity and results. The [Alpha 1 validation record](windows-client-validation-20261118.md)
documents the preceding build.

## Install and use

Double-click `Deep Research Studio_1.1.0-alpha.3_x64-setup.exe` from the Windows
account that will use the app. A standard account can complete the installation.
When WebView2 has no registered version, keep the computer online so the installer
can download Microsoft's bootstrapper. In our standard-user test, it installed a
new runtime in the user's profile. An elevated Sandbox session failed during the
Microsoft update step. If you encounter that failure after choosing **Run as
administrator**, reopen the installer normally. [Installer test results](qa/windows-installer-acceptance-20261118.md#online-webview2-bootstrap)

The client includes its local API and requires WebView2. Python, Docker and Redis
are unnecessary on the destination machine.
If the registered version is unavailable, the client looks for a machine-installed
Evergreen runtime and checks the Microsoft signatures on its browser and loader
before using it. This development installer is unsigned.

Open **Deep Research Studio** from the Start menu. The startup window stays
visible until the bundled API is listening. Configure your own model and search
providers in Settings. The client reuses the four research modes, library,
reports, Memory and knowledge features of Gen1 Web.

Only `127.0.0.1:18789` is bound. Another running copy or an occupied port causes a
startup error; the client does not connect to an unrelated service. Close the
existing copy before starting another one.

## Data and credentials

The default backend profile is
`%LOCALAPPDATA%\app.deepresearch.studio.gen1\backend`. It contains
`desktop-profile.json`, `baseline.sqlite3`, `config.json`, snapshots, and
`provider-secrets.dpapi.json`. Provider secrets use Windows DPAPI; config stores
references only. The research database is ordinary SQLite. DPAPI is bound to
the Windows user/machine context; copying a profile to another account is not
a credential recovery procedure.

The client creates a separate profile. It does not import Web databases, Docker
volumes, environment credentials or second-generation data. Unknown nonempty
directories are rejected. For isolated testing, an absolute local
`DEEPRESEARCH_DESKTOP_DATA_DIR` can select a separate desktop profile. Network
drives and unrecognized profiles are rejected.

Background jobs run inside the client. Closing the app stops its local service;
research cannot continue while the application is closed. Interrupted research
must be retried or continued using an available Deep Research checkpoint.
Saved reports, sessions, imports and credentials remain in the profile.

## Build from source

Use 64-bit Windows, Python 3.12, Node.js 22, Rust/MSVC and the Windows SDK. From
the repository root:

```powershell
python -m venv .tmp/desktop-sidecar-env
& .tmp/desktop-sidecar-env/Scripts/python.exe -m pip install `
  -r apps/desktop/requirements-sidecar.txt -r apps/desktop/requirements-build.txt
npm --prefix apps/web ci
npm --prefix apps/desktop ci
npm --prefix apps/desktop run icons
& .tmp/desktop-sidecar-env/Scripts/python.exe apps/desktop/scripts/build_sidecar.py `
  --python .tmp/desktop-sidecar-env/Scripts/python.exe
npm --prefix apps/desktop run build -- --bundles nsis
```

The sidecar builder records SHA-256, locked dependency versions and source
hashes next to the binary in `apps/desktop/src-tauri/binaries/`. A source change
during packaging fails the identity check. User databases, config and secrets
are not bundle inputs. Build output does not modify the Web deployment.

The installer is under
`apps/desktop/src-tauri/target/release/bundle/nsis/`. Distribute the installer:
the small Rust executable alone needs its accompanying API executable and is
not a portable single-file distribution.

After building the sidecar, `npm --prefix apps/desktop run dev` launches Vite
desktop mode. Only debug launches permit the exact origin
`http://127.0.0.1:5173`; release uses `http://tauri.localhost`.

## Verify

```powershell
& .tmp/desktop-sidecar-env/Scripts/python.exe -m pip install -r requirements-dev.txt
& .tmp/desktop-sidecar-env/Scripts/python.exe -m pytest tests/desktop
& .tmp/desktop-sidecar-env/Scripts/python.exe scripts/verify_windows_sidecar.py `
  --executable apps/desktop/src-tauri/binaries/deepresearch-api-x86_64-pc-windows-msvc.exe `
  --output-dir .tmp/windows-packaged-smoke
npm --prefix apps/web run test -- src/lib/apiOrigin.test.ts src/lib/runtime.test.ts
cargo test --manifest-path apps/desktop/src-tauri/Cargo.toml --locked
```

The smoke script requires a new output directory and an unused desktop port. It
preserves a synthetic profile and a secret-free result record. It exercises real
startup, exact CORS/Host boundaries, duplicate-start rejection, DPAPI save/reopen,
document parsing in the frozen executable, session persistence, graceful exit
and parent-pipe loss. It makes no paid provider requests.

The [acceptance record](windows-client-acceptance-20261118.md) identifies the
exact installers used for native UI and live-provider tests. It also records
the failed attempts and the citation coverage displayed for each real report.

## 简体中文

从日常使用的 Windows 账户双击安装包。如果系统没有注册 WebView2，请保持联网，
安装程序会下载微软引导程序。普通账户已经实测可以完成安装；若从提升权限的会话启动时
遇到 WebView2 安装错误，请返回普通桌面重新打开安装包。

本分支在已公开的第一代 Web 基线上新增 Windows x64 客户端，采用 Tauri 2 与随安装包
携带的本地 API。终端用户无需安装 Python、Docker 或 Redis。界面仍为英文，首次启动后
在 Settings 配置自己的 Provider。

客户端使用独立数据目录和 DPAPI 凭据库，不自动迁移 Web / 第二代数据。关闭窗口会停止
研究进程，已保存的数据保留。当前安装包是未签名的本地开发版本。各版本的安装与升级、
桌面界面和真实研究结果见[验收记录](windows-client-acceptance-20261118.md)。
