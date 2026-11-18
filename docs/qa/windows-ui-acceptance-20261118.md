# Windows native UI acceptance — 2026-11-18

## Scope and isolation

- Candidate exercised through the packaged Tauri release shell and bundled `deepresearch-api.exe`, not the Vite development server.
- Application data: `.tmp/ui-acceptance-20261118/profile` with the desktop profile marker and a UI-acceptance nonce.
- WebView2 data: a dedicated directory below `.tmp/ui-acceptance-20261118`; no default browser or application profile was reused.
- CDP and the API listened on loopback only (`127.0.0.1:9223` and `127.0.0.1:18789`) for the lifetime of the owned test process.
- Provider configuration was checked before every mutating automation run. The script refuses to write if an LLM or search provider is configured.
- Research, external-source, memory, library and report records used in this lane were synthetic fixtures. No provider request was made.

The reusable acceptance driver also validates that its recorded PID belongs to the executable inside the dedicated QA root, and writes only records with the acceptance fixture names or IDs. It is not suitable for a normal user profile.

## Earlier UI candidate

The initial interaction sweep used the packaged Alpha 2 candidates. The following shell includes the desktop external-link correction and predates the Sandbox runtime changes. Its executable and sidecar were copied to the isolated QA directory.

| Artifact | SHA-256 | Size |
| --- | --- | ---: |
| `deep-research-studio-desktop.exe` | `3310e50335305ae5999d1f657dbd5befdb3fb3b73d41d594a5213c18d618550a` | 11,808,256 bytes |
| `deepresearch-api.exe` | `9dc34ed3cf1ea1058fd7f19e0d7472992b88ec66b0d30c8ff6cdb95e2e481445` | 30,399,847 bytes |

## Native interaction results

| Area | Native interaction | Result |
| --- | --- | --- |
| Launch/runtime | Packaged shell reached `http://tauri.localhost/`; title and initial page rendered | Pass |
| Navigation | New, History, Library, Memory, Knowledge Graph and Settings routes opened through the sidebar | Pass |
| Theme/sidebar | Light/dark round trip and sidebar collapse/expand | Pass |
| Modes | Fast Search, Multiple Search, Deep Research and Models Council selection; selected state checked | Pass |
| Provider guard | Submission with no configured provider produced the expected error and created no fake research | Pass |
| History | History, Archived and back navigation | Pass |
| Library | XLSX import/open, note creation/open, Markdown report upload/preview | Pass |
| Library provenance | Standalone local import shows zero research links and has no `#/session/null` link | Pass after frontend and API fixes |
| Memory | Add, edit, pin, unpin, archive, restore; daily dreaming schedule and post-turn setting round trip | Pass |
| Graph | Rebuild plus Wiki Map and Session Map switching | Pass |
| Settings | Deep checkpoint preference saved, page reloaded and value persisted | Pass |
| Restart | Settings, imported workbook, note, updated memory and daily schedule persisted across a packaged-app restart | Pass |
| Completed fixture | Synthetic completed run opened from History and rendered its final answer | Pass |
| Export | Markdown export downloaded and its title, mode/status, answer and limitations were inspected | Pass |
| Result actions | Copy changed to `Copied`; Save to memory changed to `Saved` | Pass |
| External link | Alpha 2 exposed a Tauri ACL error for `_blank`; the repackaged `system-browser` path reached a unique loopback nonce and created no child WebView | Pass after fix |
| Native close | `CloseMainWindow()` returned true; root process, API port and CDP port all exited within 20 seconds | Pass |

The exported fixture is `.tmp/ui-acceptance-20261118/native-export.md` (196 bytes). Its limitations section explicitly states that the result is synthetic and made no provider request.

## Defects found and fixes

### Standalone imports appeared linked to a null run

The source provenance API returned a local-import provenance row with both `session_id` and `run_id` set to null. The UI counted that row and constructed `#/session/null`.

The frontend filters provenance rows through `artifactHref()` before counting or rendering links. The API also omits rows with empty session and run IDs. `Library.test.tsx` covers the null-provenance response, the updated `Research links: 0` count and absence of a null link.

### Desktop external links were rejected by the shell ACL

With `target="_blank"`, the Tauri initialisation script intercepted the click and raised `shell|open not allowed by ACL` before Rust `on_new_window` could apply its HTTP/HTTPS-only system-browser policy.

Desktop assets now use the named target `system-browser` for Answer Markdown, Library sources and SourcesRail cards. Web/demo builds keep `_blank`; `rel="noreferrer noopener"` remains present. A pre-package proof changed only the synthetic link target at runtime and clicked a unique loopback URL. The local listener received `GET /ui-alpha2-system-browser-7f19c2`, the Tauri context stayed at one page, and the app remained visible.

The freshly packaged candidate then repeated the check without DOM target changes. The driver asserted the source-generated `target="system-browser"` and `rel` value, clicked the link, observed no new WebView page and no console/page error, and the listener received `GET /ui-final-system-browser-c92d6e`.

## Verification evidence

- `npm test -- --run src/components/AnswerBody.test.tsx src/components/SourcesRail.test.tsx src/pages/Library.test.tsx src/pages/Memory.test.tsx src/lib/apiOrigin.test.ts src/lib/runtime.test.ts` — 6 files, 64 tests passed.
- `npm run lint` — TypeScript `--noEmit` passed.
- `npm run build:desktop` — 2,589 modules transformed; desktop asset build completed successfully.
- The generated desktop JavaScript contains `system-browser`.
- `node --check .tmp/windows-native-acceptance.mjs` — passed.
- `git diff --check -- apps/web docs/qa/windows-ui-acceptance-20261118.md` — passed.

The external-link candidate used these commands from `apps/web` after the temporary driver was moved to the ignored `.tmp` directory.

```powershell
$env:WINDOWS_UI_QA_ROOT = 'D:\deepresearch-gen1-windows\.tmp\ui-acceptance-20261118'
$env:WINDOWS_UI_PHASE = 'final'
$env:WINDOWS_UI_EXTERNAL_PROBE_URL = 'http://127.0.0.1:19322/ui-final-system-browser-c92d6e'
node .tmp/windows-native-acceptance.mjs
```

The final phase on that candidate reported `passed: true` for restart persistence, completed-run history and export, Copy and Save to memory, and external system-browser routing. Its `console_errors` list was empty. The earlier initial-phase driver reached its final screenshot, then exited nonzero when the browser logged the deliberate unconfigured-provider HTTP 409.

Screenshots and raw automation output are kept under `.tmp/ui-acceptance-20261118` and are intentionally excluded from Git.

## Installed Alpha 2 repetition

The installer SHA-256 was `1afdb1e23a1c8148fc490c65c78c35f420042c9d3218d0794949add898b0dfa5`. The installed Tauri shell was 11,831,808 bytes, SHA-256 `57784877d9db036a80fe2b41eeb8f8559b02e9095348cce4992b397cfc5ba0a3`. The installed sidecar matched the 30,399,847-byte binary in the table above. Both installed files were copied to the isolated directory and rehashed before each launch.

The initial phase passed all 13 groups of interactions in the table above. Creating a run in the empty-provider profile returned HTTP 409 on `POST /api/v1/runs`, displayed the missing-provider message and kept the user on the home screen. That was the only unsuccessful HTTP response. During the first repetition a test locator also matched a second rendering of memory text. The driver was corrected to retain the selected record's ID while pinning and unpinning. The rerun passed with zero unexpected console errors.

After a normal close and restart, all four final-phase groups passed. The library showed `Research links: 0` for the standalone workbook and had no link to a null session. The test opened the completed fixture, downloaded a 196-byte Markdown file, copied the answer and saved it to memory. The exported bytes matched SHA-256 `5321635ddac60c1b2488a100c7e316a879c936b90e47a1286664f5554a15cbf6`.

For this repetition, the test source already contained the loopback URL at rest. Clicking its unmodified anchor generated a system-browser request to the unique fixture path. No child WebView appeared. The final phase recorded no unsuccessful HTTP responses or console errors. Both native closes succeeded and released the API, CDP and probe ports. Detailed receipts are stored as `alpha2-installed-recheck-{initial,final}.json` and `alpha2-installed-recheck-{initial,final}-close.json` in the isolated QA directory.

## Final installed Alpha 3

The final NSIS package had SHA-256 `57db92560c21d96a87d7c2f7cabb183248d37f0a7faac0d98d7bd4c94cb9abff`. The check used copies of the two binaries from the upgraded installation, with hashes verified after copying and again before launch.

| Installed binary | Bytes | SHA-256 |
| --- | ---: | --- |
| `deep-research-studio-desktop.exe` | 11,838,976 | `bacb081fbce039be36ccee923c7e087e88bef34df9003afa95515461b1a28d6e` |
| `deepresearch-api.exe` | 30,399,138 | `8c87c3518ee410e5f868508fbb868b29e59178f0c2a6653d555e577f0a91da0b` |

All four final-phase groups passed on these files. Settings, memory and library records survived restart. The standalone workbook showed zero research links. A source fixture with a linked run showed exactly one research link with the expected run URL. History opened the saved answer, Copy and Save to memory succeeded, and the 196-byte Markdown export matched the Alpha 2 export hash.

Clicking the source's unmodified external link generated `GET /ui-installed-f8c43f851305440f` in the system browser. The native application kept one WebView. There were zero unsuccessful HTTP responses and zero console errors. The shell closed normally and two post-exit checks found zero listeners on the API, CDP and probe ports. The isolated receipts are `alpha3-installed-final.json` and `alpha3-installed-final-close.json`.

## Boundaries

- Real-provider Fast/Deep research is a separate acceptance lane using explicit credentials and budgets; this synthetic UI lane does not claim that coverage.
- Installer install/upgrade/uninstall evidence is recorded by the installer lane; it is not duplicated here.
