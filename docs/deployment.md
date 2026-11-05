# Deployment and recovery

[简体中文](#简体中文)

Deep Research Studio runs as a local six-service Docker Compose application: Web, API, research worker, semantic worker, Redis, and a network-isolated key holder. The published launcher builds the images, creates or unlocks the installation, and runs database and credential-vault checks.

## Requirements

- Git
- Python 3.12
- Docker Engine with Compose v2
- Windows: Docker Desktop configured for Linux containers
- Linux/WSL: an ext4 (or equivalent native Linux) home directory for the root key

Confirm the tools before cloning:

```text
docker version
docker compose version
python --version
```

The Python command may be `py -3.12` on Windows or `python3` on Linux/WSL.

## Windows with Docker Desktop

Open PowerShell:

```powershell
git clone https://github.com/Moonaria123/Deep-Research-Studio.git
Set-Location Deep-Research-Studio
py -3.12 scripts\start_gen1_web.py --initialize --build --data-dir D:\DeepResearchStudioData\main
```

The data directory must be absolute and empty for `--initialize`. The launcher creates a random root key and protects it for the current Windows user with DPAPI. Its default location is:

```text
%LOCALAPPDATA%\DeepResearchStudio\deep-research-studio\vault-root.dpapi
```

If the first initialization fails after creating its key binding (for example, because a port is occupied), correct the external cause and rerun `--initialize` with the same project and data directory. Only a matching installation still marked `initializing` can resume. After the launcher reports success, omit `--initialize` for every normal start.

For later starts, omit `--initialize` and normally omit `--build`:

```powershell
py -3.12 scripts\start_gen1_web.py --data-dir D:\DeepResearchStudioData\main
```

## Native Linux or WSL2

Clone the repository into the Linux filesystem, not `/mnt/c` or `/mnt/d`. From a Linux shell:

```bash
git clone https://github.com/Moonaria123/Deep-Research-Studio.git
cd Deep-Research-Studio
python3 scripts/start_gen1_web.py --initialize --build --data-dir "$HOME/.local/share/deep-research-studio/main"
```

The launcher stores the 32-byte root key at:

```text
${XDG_STATE_HOME:-$HOME/.local/state}/deep-research-studio/deep-research-studio/vault-root.key
```

The parent directory must belong to the current user with mode `0700`; the key must be a regular file owned by that user with mode `0600`. The launcher rejects Windows, removable, and network-style mounts that cannot enforce the expected Linux permissions.

For later starts:

```bash
python3 scripts/start_gen1_web.py --data-dir "$HOME/.local/share/deep-research-studio/main"
```

## First use

The default UI is <http://127.0.0.1:15175>; the local API is <http://127.0.0.1:18789>. The launcher prints the effective UI address after all health checks pass.

1. Open the UI and go to **Settings**.
2. Add at least one supported model provider and assign a default model.
3. Add a search provider when you want external Web retrieval.
4. Run the connection tests in Settings.
5. Start with **Fast Search**, then choose a deeper mode when the task needs broader research.

Provider keys are submitted to the local API and written to the encrypted credential vault. Configuration stores opaque references rather than the raw values.

## Custom project and ports

Use a distinct lowercase project name and ports when running more than one installation. Set the ports in the shell before invoking the launcher and use the same `--project` value on every start.

PowerShell:

```powershell
$env:DEEPRESEARCH_WEB_PORT = "15176"
$env:DEEPRESEARCH_API_PORT = "18790"
py -3.12 scripts\start_gen1_web.py --project deep-research-studio-qa --initialize --build --data-dir D:\DeepResearchStudioData\qa
```

Linux/WSL:

```bash
export DEEPRESEARCH_WEB_PORT=15176
export DEEPRESEARCH_API_PORT=18790
python3 scripts/start_gen1_web.py --project deep-research-studio-qa --initialize --build --data-dir "$HOME/.local/share/deep-research-studio/qa"
```

The project name selects the root-key location. Do not point two project names at the same data directory, and do not reuse one project's key with another project's data.

## Read-only installation check

Run the launcher with `--check` to validate the key/data binding, SQLite integrity, encrypted credential vault, and stored credential references without starting the application services:

Windows:

```powershell
py -3.12 scripts\start_gen1_web.py --check --data-dir D:\DeepResearchStudioData\main
```

Linux/WSL:

```bash
python3 scripts/start_gen1_web.py --check --data-dir "$HOME/.local/share/deep-research-studio/main"
```

The check runs one isolated, self-removing doctor container. It does not start or change the long-running key-holder, Redis, Web, API, or worker services, and it does not call a provider. The business database and credential vault are opened in read-only/query-only mode; SQLite may maintain lock or WAL coordination sidecars while taking a consistent snapshot.

## Stop the stack

Set the same absolute data directory and stop the exact Compose project:

PowerShell:

```powershell
$env:DEEPRESEARCH_DATA_DIR = "D:/DeepResearchStudioData/main"
docker compose -p deep-research-studio -f docker/compose.gen1.yml down
```

Linux/WSL:

```bash
export DEEPRESEARCH_DATA_DIR="$HOME/.local/share/deep-research-studio/main"
docker compose -p deep-research-studio -f docker/compose.gen1.yml down
```

This removes the stopped containers and network for that project. It does not delete the host data directory, images, or persistent host root key. Do not add `--volumes`: the temporary key volume is recreated safely at startup, but broad volume deletion is unnecessary and can target the wrong project if the name is incorrect.

## Backup and restore

Create a cold backup. Stop the exact project first, then copy these items to separate protected locations:

1. the complete data directory, including hidden files;
2. the protected root-key file for the same project;
3. a note containing the project name and release version, but no key material.

Windows DPAPI data normally needs the same Windows user and machine context to decrypt. A copied DPAPI blob is not a portable plaintext recovery key. Maintain an operating-system recovery strategy appropriate to your Windows account.

On Linux/WSL, the root-key file is the recovery key. Store its backup on encrypted or offline media, preserve mode `0600` after restoration, and never place it in the repository or beside an unencrypted data backup.

Restore in this order:

1. install the same or a compatible release without running `--initialize`;
2. restore the complete data directory to an absolute host path;
3. restore the matching protected root-key file to the project's state path;
4. run `--check` against the restored data directory;
5. start normally only after the check passes.

The launcher refuses to initialise a non-empty data directory, replace an existing key, or start when the key/data binding does not match. Do not work around these checks by deleting the binding record or generating another key.

## Update

Before updating, make a cold backup. Fetch the intended release, review its release notes, rebuild the public images, and start with the same project and data directory:

```text
git fetch --tags
git checkout v1.0.0
```

Then run the normal launcher command with `--build`. Do not use `--initialize` for an existing installation.

## Troubleshooting

| Message or symptom | Action |
| --- | --- |
| `--data-dir is required` | Pass an absolute host path. |
| Initialisation was refused | Use a new empty directory and a project with no existing root key, or restore the existing matching installation. |
| Root key is missing | Restore the matching key. Do not initialise over existing data. |
| Key does not match the data directory | Check the project name and restore the corresponding key/data pair. |
| Linux key permissions are rejected | Move the key under the native Linux home filesystem and restore directory mode `0700` and file mode `0600`. |
| A port is in use | Select unused ports and keep them consistent for that installation. |
| Docker build fails | Confirm Docker can pull the pinned public base images and that the engine uses Linux containers. |

The launcher deliberately bounds error output so that private paths and credentials do not appear in normal failures. Inspect Docker status without printing environment variables or configuration values.

---

## 简体中文

Deep Research Studio 以六个本地 Docker Compose 服务运行：Web、API、研究 Worker、语义 Worker、Redis，以及一个无网络的 key-holder。公开启动器负责构建镜像、创建或解锁安装，并检查数据库和加密凭据库。

### 前置条件

- Git
- Python 3.12
- Docker Engine 与 Compose v2
- Windows：使用 Linux 容器的 Docker Desktop
- Linux/WSL：用于保存根密钥的 ext4（或同类原生 Linux）用户目录

先运行 `docker version`、`docker compose version` 和 `python --version` 检查工具。Windows 的 Python 命令通常是 `py -3.12`，Linux/WSL 通常是 `python3`。

### Windows + Docker Desktop

在 PowerShell 中执行：

```powershell
git clone https://github.com/Moonaria123/Deep-Research-Studio.git
Set-Location Deep-Research-Studio
py -3.12 scripts\start_gen1_web.py --initialize --build --data-dir D:\DeepResearchStudioData\main
```

首次初始化要求数据目录为绝对路径且为空。启动器生成随机根密钥，并用当前 Windows 用户的 DPAPI 保护。默认位置为：

```text
%LOCALAPPDATA%\DeepResearchStudio\deep-research-studio\vault-root.dpapi
```

如果首次初始化在写入密钥绑定后失败（例如端口被占用），先修复外部原因，再以相同项目名和数据目录重新运行 `--initialize`。只有密钥匹配且状态仍为 `initializing` 的安装可以续装。启动器报告成功后，所有日常启动都应移除 `--initialize`。

日常启动不带 `--initialize`，通常也不需要 `--build`：

```powershell
py -3.12 scripts\start_gen1_web.py --data-dir D:\DeepResearchStudioData\main
```

### 原生 Linux 或 WSL2

仓库应克隆到 Linux 文件系统，不要放在 `/mnt/c` 或 `/mnt/d`：

```bash
git clone https://github.com/Moonaria123/Deep-Research-Studio.git
cd Deep-Research-Studio
python3 scripts/start_gen1_web.py --initialize --build --data-dir "$HOME/.local/share/deep-research-studio/main"
```

根密钥默认保存到：

```text
${XDG_STATE_HOME:-$HOME/.local/state}/deep-research-studio/deep-research-studio/vault-root.key
```

父目录必须属于当前用户且权限为 `0700`；密钥必须是当前用户所有、权限为 `0600` 的普通文件。无法落实这些权限的 Windows、移动介质和网络式挂载会被拒绝。

日常启动：

```bash
python3 scripts/start_gen1_web.py --data-dir "$HOME/.local/share/deep-research-studio/main"
```

### 首次使用

默认界面地址是 <http://127.0.0.1:15175>，本地 API 地址是 <http://127.0.0.1:18789>。全部健康检查通过后，启动器会输出实际界面地址。

进入 **Settings**，添加模型和搜索 Provider，并运行连接检查。Provider 密钥只提交给本地 API，配置保存的是凭据引用而不是原始值。

### 检查安装

加入 `--check` 可以检查密钥与数据绑定、SQLite 完整性、加密凭据库和引用，不启动 Web、API 或 Worker，也不调用 Provider：

```powershell
py -3.12 scripts\start_gen1_web.py --check --data-dir D:\DeepResearchStudioData\main
```

```bash
python3 scripts/start_gen1_web.py --check --data-dir "$HOME/.local/share/deep-research-studio/main"
```

该检查只运行一个隔离且自动移除的一次性 doctor 容器，不会启动或改变长期运行的 key-holder、Redis、Web、API 或 Worker。业务数据库和凭据库以只读/query-only 模式打开；SQLite 在生成一致快照时可能维护锁或 WAL 协调文件。

### 备份与恢复

先停止准确的 Compose 项目，再分别备份完整数据目录（包括隐藏文件）和对应项目的根密钥文件。不要把密钥放进仓库，也不要与未加密的数据副本放在一起。

Windows DPAPI 文件通常需要相同的 Windows 用户和机器上下文。它不是可跨机器使用的明文恢复密钥。Linux/WSL 的根密钥文件就是恢复密钥；备份时应使用加密或离线介质，恢复后保持 `0600`。

恢复顺序：

1. 安装相同或兼容版本，但不要执行 `--initialize`；
2. 恢复完整数据目录；
3. 恢复该项目对应的根密钥；
4. 对恢复的数据运行 `--check`；
5. 检查通过后再正常启动。

启动器会拒绝在非空目录上初始化、替换已有密钥或使用不匹配的密钥启动。不要通过删除绑定记录或生成新密钥来绕过检查。

### 更新

更新前先做冷备份。切换到目标版本后，使用原项目名、数据目录和带 `--build` 的正常启动命令重建镜像。已有安装不要使用 `--initialize`。

故障处理表和自定义端口的完整示例请参考本文英文部分；两种语言描述的是同一套启动器参数和安全边界。
