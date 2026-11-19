# Deep Research Studio for Windows

Gen1 Windows x64 client, built with Tauri 2 and a packaged Python API sidecar.
The frontend is reused from `apps/web`; destination machines do not need Python,
Docker or Redis.

See [the Windows guide](../../docs/windows-client.md) for installation, isolated
data storage, reproducible builds and validation boundaries.

Build the API sidecar first, then run `npm run build -- --bundles nsis` here.
`npm run dev` uses Vite desktop mode and the same prebuilt sidecar.
