# Security policy

[简体中文](#简体中文)

## Supported versions

Security fixes are applied to the latest published release of Deep Research Studio.

## Report a vulnerability

Do not open a public issue for a suspected vulnerability or exposed credential. Use GitHub's private vulnerability reporting feature in this repository. Include the affected version, reproduction steps, impact, and any suggested mitigation. Do not include real provider keys, user reports, or private data.

If private reporting is unavailable, open a public issue that contains no exploit details or sensitive values and ask the maintainer to establish a private channel.

## Security model

- The production Web deployment listens on the local host by default. Do not expose it directly to the public internet.
- Provider credentials are entered locally. The application stores references in configuration and keeps credential values in an encrypted provider vault.
- Windows protects the persistent root key with DPAPI. Linux/WSL protects its root-key file with owner-only filesystem permissions and injects it into a memory-backed container volume at startup.
- Linux permission mode `0600` is access control, not encryption at rest. Use full-disk encryption and keep the key under the Linux filesystem rather than `/mnt/c`, `/mnt/d`, removable media, or a network share.
- A lost root key cannot be recreated for an existing encrypted vault. Back up the data directory and root key separately.
- The hosted Demo has no provider keys or model access. It uses fixed public fixtures and resets its in-memory state on reload.

See [Deployment and recovery](docs/deployment.md) for operational safeguards.

---

## 简体中文

### 支持版本

安全修复应用于 Deep Research Studio 的最新公开版本。

### 报告漏洞

疑似漏洞或凭据泄露不要提交公开 Issue。请使用本仓库的 GitHub 私密漏洞报告功能，并提供受影响版本、复现步骤、影响和建议的缓解方法。不要附带真实 Provider 密钥、用户报告或私人数据。

若私密报告功能不可用，请创建不含利用细节和敏感值的公开 Issue，请维护者建立私密沟通渠道。

### 安全边界

- 正式 Web 部署默认只监听本机地址，不应直接暴露到公网。
- Provider 凭据在本机输入。配置只保存引用，凭据值保存在加密凭据库中。
- Windows 使用 DPAPI 保护持久化根密钥；Linux/WSL 使用仅限文件所有者访问的根密钥文件，并在启动时将其注入内存卷。
- Linux 的 `0600` 权限属于访问控制，不等同于静态加密。建议启用全盘加密，并把密钥保存在 Linux 文件系统中，不要放在 `/mnt/c`、`/mnt/d`、移动介质或网络共享上。
- 已有加密凭据库的根密钥丢失后无法重新生成。数据目录和根密钥应分别备份。
- 线上 Demo 不含 Provider 密钥，也不能调用模型；它使用固定公开样例，并在刷新后重置内存状态。

操作防护与恢复流程见[部署与恢复](docs/deployment.md)。
