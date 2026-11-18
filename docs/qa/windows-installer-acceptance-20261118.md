# Windows installer acceptance — 2026-11-18

## Scope and isolation

- Candidate: `Deep Research Studio 1.1.0-alpha.3` NSIS x64 installer.
- Clean-machine coverage used Windows Sandbox (`WDAGUtilityAccount`) with networking, clipboard and vGPU disabled. Inputs were mapped read-only and evidence output was mapped read/write.
- Host coverage used only dedicated install, backend-profile and WebView2-profile directories under `.tmp/host-installer-acceptance`.
- The existing default host profile under `%LOCALAPPDATA%\app.deepresearch.studio.gen1` was read only to compute file-count, byte-count and content-hash tree identity; its files were not parsed, copied or modified. The identity stayed at 213 files, 14,328,144 bytes and SHA-256 `239e680c191a8ab733dd9a376e6914d39fa87dc50bed791a521423f12dff9414` through alpha.1, alpha.2 and alpha.3 host upgrades.
- Provider credentials and credential-bearing profile files were not read, copied or recorded by installer acceptance.

## Locked artifacts

| Artifact | SHA-256 |
| --- | --- |
| Alpha.1 installer | `328290dc74c53480762389e39b8a5c7f26da65fa755abbad387ad95b538fda4e` |
| Alpha.3 installer | `57db92560c21d96a87d7c2f7cabb183248d37f0a7faac0d98d7bd4c94cb9abff` |
| Alpha.3 desktop shell | `bacb081fbce039be36ccee923c7e087e88bef34df9003afa95515461b1a28d6e` |
| Alpha.3 API sidecar | `8c87c3518ee410e5f868508fbb868b29e59178f0c2a6653d555e577f0a91da0b` |

## WebView2 recovery diagnosis

The clean Sandbox image exposed a stale/misaligned Evergreen registration: HKLM reported `157.0.4341.42`, while the signed runtime on disk was `157.0.4341.46`.

- The locked `WebView2Loader.dll` returned `0x80070002` and no version for default discovery.
- The same loader returned success (`0x00000000`) and `157.0.4341.46` when given the actual runtime directory.
- `GetFinalPathNameByHandleW` succeeded for Program Files (x86), the version directory, `msedgewebview2.exe`, and `EBWebView\x64\EmbeddedBrowserWebView.dll`. All resolved to the same paths with only the normal `\\?\` prefix and none was a reparse point.
- The Rust recovery harness verified both required runtime files as Microsoft-signed, selected the on-disk `.46` directory without a parent-process override, and then returned `Ok("157.0.4341.46")` from the version probe (`1 passed`, 0.48 s).
- Microsoft online bootstrapper and signed x64 Evergreen standalone repair attempts both returned `0x80040902` in Windows Sandbox and did not repair its registration. The product fix therefore recovers the already installed, signed runtime instead of weakening the desktop Job protection or bundling a fixed runtime.

## Clean Alpha.3 install and uninstall

Result: **PASS** in a new Windows Sandbox instance.

- Installer, registry version, file version, shell hash, sidecar hash and both shortcuts matched the locked Alpha.3 candidate.
- `WEBVIEW2_BROWSER_EXECUTABLE_FOLDER` was absent before launch; shell and sidecar process counts were both zero.
- Cold start reached `/api/v1/health` in 13,007 ms with `status=ok` and `job_backend=thread`.
- Backend profile and WebView2 user-data directories were created, and the main window closed normally.
- Silent uninstall removed the uninstall registration, installed binaries, desktop shortcut and Start-menu shortcut while preserving the dedicated test backend profile.

Evidence: `.tmp/sandbox-installer-output/alpha3-fresh-install.json`.

## Alpha.1 to Alpha.3 upgrade in clean Sandbox

Result: **PASS** in a separate new Windows Sandbox instance.

- Alpha.1 was installed from the locked real Alpha.1 package. Because Alpha.1 predates stale-registration recovery, only that Alpha.1 process received the explicitly disclosed `.46` runtime-folder override.
- Alpha.1 created session marker `sess_62324bb52d83`, reproduced the known premature duplicate-profile-directory behavior, kept the primary service healthy, and closed normally.
- Alpha.3 upgraded in place with the locked installer, shell and sidecar identities. Registry and file versions were `1.1.0-alpha.3` and both shortcuts remained present.
- Before the first Alpha.3 launch, the inherited legacy override existed; `Remove-Item Env:WEBVIEW2_BROWSER_EXECUTABLE_FOLDER` removed it and the process environment subsequently reported it absent. Shell and sidecar process counts were zero before launch.
- Alpha.3 discovered/recovered WebView2 itself, reached health in 12,128 ms, preserved the Alpha.1 marker, did not create the nonexistent duplicate profile directory, kept the primary service healthy and closed normally.
- Silent uninstall removed registration, installed binaries and both shortcuts while preserving the upgrade-test profile.

Evidence: `.tmp/sandbox-installer-output/installer-acceptance.json`.

## Host upgrade, real Provider and protected-data check

Result: **PASS**.

- The dedicated host install was upgraded Alpha.1 → Alpha.2 → Alpha.3.
- Alpha.3 registry/file versions and deployed hashes matched the locked candidate.
- The synthetic Alpha.1 marker persisted, duplicate launch left its nonexistent profile directory absent, the primary service remained healthy, and the window closed normally.
- The installed Alpha.1 build completed a real-Provider Fast research before upgrade. The installed Alpha.2 build completed a real-Provider Deep research with 9 external hits, 5 external citations and a formal report. After the final Alpha.3 upgrade, installed-path run `run_1eb2758363b0` completed with two Brave searches, one Firecrawl extraction, successful Poe writing, 83% SQLite citation coverage and a real 4,114-byte Markdown export; its window then closed normally.
- Installer acceptance did not inspect or copy any credential or result content from the isolated real-Provider profile.
- The protected default profile tree identity remained byte-for-byte unchanged across all three stages.
- After final Provider and UI acceptance released port 18789, Alpha.3 silent uninstall removed the uninstall registration, installed binaries, desktop shortcut and Start-menu shortcut. The dedicated acceptance profile was preserved and the protected default profile retained the exact original tree identity.

Evidence: `.tmp/host-installer-acceptance/state.json`.

## Final result

**PASS.** Clean install, real Alpha.1 → Alpha.3 upgrade, host Alpha.2 → Alpha.3 upgrade, stale WebView2 registration recovery, profile persistence, duplicate-instance safety, real-Provider operation, export, normal close and silent uninstall were all exercised. No installer-acceptance failure remains open.
