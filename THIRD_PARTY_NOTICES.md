# Third-party notices

Deep Research Studio v1.0.0 and later are distributed under the [PolyForm Noncommercial License 1.0.0](LICENSE); see [commercial permission and attribution requirements](docs/licensing.md). Third-party packages and container images retain their own licences and copyright notices. The repository licence does not replace those terms.

The table below lists direct runtime dependencies in this release. Transitive dependencies are recorded by `apps/web/package-lock.json` and the packages installed from `docker/requirements-api.txt`; consult each distributed package for its complete notice text.

## Web runtime

| Package | Licence |
| --- | --- |
| React, React DOM, React Router, React Markdown, remark-gfm | MIT |
| @dagrejs/dagre, @xyflow/react | MIT |
| clsx, tailwind-merge | MIT |
| class-variance-authority | Apache-2.0 |
| Lucide React | ISC |
| GSAP | [GSAP Standard No Charge License](https://gsap.com/standard-license/) |

## API runtime

| Package | Licence |
| --- | --- |
| FastAPI, Pydantic, openpyxl | MIT |
| Uvicorn, pypdf | BSD-3-Clause |
| python-multipart | Apache-2.0 |
| NumPy | BSD-3-Clause and bundled component licences |
| Redis Python client | MIT |
| RQ | BSD-2-Clause |
| pysqlcipher3 | BSD-3-Clause |

## Windows desktop development package

The Gen1 Windows package adds the components below. The Rust dependency
inventory is locked in `apps/desktop/src-tauri/Cargo.lock`; the isolated Python
runtime and build inventories are in `apps/desktop/requirements-sidecar.txt` and
`apps/desktop/requirements-build.txt`. Windows uses the API dependencies above
without Redis, RQ or pysqlcipher3. These desktop components are not additions to
the published Web v1.0.0 artifacts.

| Component | Licence |
| --- | --- |
| Tauri, tauri-plugin-shell, serde_json, windows-sys | MIT OR Apache-2.0 |
| Bundled CPython runtime | Python Software Foundation licence and included component notices |
| PyInstaller bootloader and packaging runtime | GPL-2.0-or-later with the PyInstaller distribution exception; see its `COPYING.txt` |
| PyInstaller community runtime hooks | Apache-2.0; build-time hooks use GPL-2.0-or-later |
| Microsoft Edge WebView2 | Microsoft terms; installed separately by the Windows bootstrapper when required |

## Container base images

The deployment builds from the Python, Node.js, nginx, and Redis images named in the Dockerfiles and Compose file. Those images include operating-system and language packages under multiple licences. Review their image documentation and included notice files before redistribution.

This notice is informational and is not legal advice. If a dependency version changes, regenerate and review the dependency inventory before publishing a release.
