# v1.0.0 release scope

v1.0.0 is the initial release of this repository under the PolyForm Noncommercial License 1.0.0. It preserves the current Gen1 Web feature set. See [licensing](licensing.md) for permitted uses, commercial permission and mandatory original-author notices.

This repository contains the first-generation Web application and the backend services it needs. It starts from the T59-based Gen1 release branch, including the subsequent report-confidence, background verification, Memory queue, and semantic-recall fixes.

The public package contains:

- the React Web client and a separate, read-only static preview build;
- the FastAPI gateway and Gen1 research, retrieval, citation, Memory, and graph runtime;
- Docker deployment, a Windows/Linux bootstrap and diagnostic tool, and regression tests;
- public sample data, documentation, and application-only demonstration media.

The package excludes the second-generation development tree, desktop shell, retired workflow runtime, private operational records, local databases, credentials, and historical acceptance tooling that depends on an unpublished workspace.

The static preview uses preset examples. Its checks establish interface behaviour and data isolation; real-provider checks belong to the local application and are recorded separately.

## 简体中文

本仓库发布第一代 Web 应用及其必需后端，以 T59 第一代发布分支为基础，包含后续的报告置信度、后台核验、Memory 队列和语义召回修复。

发布范围包括 Web 前端、第一代 API 与研究运行时、Docker 部署和 Windows/Linux 启动诊断工具、回归测试、公开样例及展示素材。第二代开发树、桌面壳、退役运行时、私有运维记录、本地数据库、凭据和依赖原工作区的历史验收工具不在发布范围内。

线上预览使用预设样例；其验收证明界面行为和数据隔离，真实模型验收与之分别记录。
