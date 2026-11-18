#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::ffi::c_void;
use std::io;
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicBool, AtomicU8, Ordering},
    Mutex,
};
use std::time::Duration;

use tauri::{
    webview::NewWindowResponse, Manager, RunEvent, Url, WebviewUrl, WebviewWindow,
    WebviewWindowBuilder,
};
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;

#[cfg(windows)]
mod webview_runtime;

const API_HOST: &str = "127.0.0.1";
const API_PORT: &str = "18789";
const READY_MARKER: &str = "DEEPRESEARCH_DESKTOP_READY";
const STARTUP_TIMEOUT: Duration = Duration::from_secs(45);
const EXIT_TIMEOUT: Duration = Duration::from_secs(12);
const STARTUP_STARTING: u8 = 0;
const STARTUP_READY: u8 = 1;
const STARTUP_FAILED: u8 = 2;

#[derive(Default)]
struct DesktopState {
    child: Mutex<Option<CommandChild>>,
    startup: AtomicU8,
    exit_requested: AtomicBool,
    expected_sidecar_stop: AtomicBool,
}

fn claim_startup_ready(startup: &AtomicU8) -> bool {
    startup
        .compare_exchange(
            STARTUP_STARTING,
            STARTUP_READY,
            Ordering::SeqCst,
            Ordering::SeqCst,
        )
        .is_ok()
}

fn claim_startup_failed(startup: &AtomicU8) -> bool {
    startup
        .compare_exchange(
            STARTUP_STARTING,
            STARTUP_FAILED,
            Ordering::SeqCst,
            Ordering::SeqCst,
        )
        .is_ok()
}

fn main() {
    // The desktop process enters the job before it creates the sidecar. Windows then
    // places every descendant (including PyInstaller's extracted child) in the same
    // kill-on-close boundary without a post-spawn assignment race.
    if install_process_tree_guard().is_err() {
        fatal_startup_error();
    }
    #[cfg(windows)]
    webview_runtime::recover_stale_registration();

    let app = tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(DesktopState::default())
        .setup(|app| {
            create_startup_window(app.handle())?;
            let data_dir = desktop_data_dir(app.handle())?;
            // The sidecar validates the profile and owns its creation after binding the port.
            spawn_api_sidecar(app.handle(), data_dir)?;
            start_timeout_watch(app.handle());
            Ok(())
        })
        .build(tauri::generate_context!())
        .unwrap_or_else(|_| fatal_startup_error());

    app.run(|app_handle, event| {
        if let RunEvent::ExitRequested { api, .. } = event {
            let state = app_handle.state::<DesktopState>();
            if state.exit_requested.load(Ordering::SeqCst) {
                // A second request is an explicit force-exit path. Do not trap it.
                return;
            }

            if let Ok(mut slot) = state.child.lock() {
                if let Some(child) = slot.as_mut() {
                    // Set the state while holding the child lock. The termination event
                    // takes the same lock, so it cannot misclassify this requested stop
                    // as a crash between the stdin write and the state transition.
                    state.exit_requested.store(true, Ordering::SeqCst);
                    claim_startup_failed(&state.startup);
                    api.prevent_exit();
                    if child.write(b"shutdown\n").is_ok() {
                        hide_app_windows(app_handle);
                        start_exit_timeout(app_handle);
                    } else {
                        app_handle.exit(0);
                    }
                }
            };
        }
    });
}

fn create_startup_window(app: &tauri::AppHandle) -> tauri::Result<WebviewWindow> {
    if let Some(window) = app.get_webview_window("startup") {
        return Ok(window);
    }
    let window = WebviewWindowBuilder::new(
        app,
        "startup",
        WebviewUrl::External("about:blank".parse().expect("valid startup URL")),
    )
    .title("Deep Research Studio")
    .inner_size(520.0, 320.0)
    .resizable(false)
    .center()
    .build()?;
    let encoded = serde_json::to_string(startup_html()).expect("static startup HTML is valid");
    window.eval(format!(
        "document.open();document.write({encoded});document.close();"
    ))?;
    Ok(window)
}

fn startup_html() -> &'static str {
    r#"<!doctype html><html><head><meta charset="utf-8"><meta name="color-scheme" content="light dark">
<style>body{font-family:system-ui,sans-serif;margin:0;min-height:100vh;display:grid;place-items:center;background:#0f172a;color:#e2e8f0}main{max-width:390px;padding:36px;text-align:center}h1{font-size:22px;margin:0 0 14px}p{color:#94a3b8;line-height:1.5;margin:0}.dot{width:10px;height:10px;margin:24px auto;border:3px solid #475569;border-top-color:#2dd4bf;border-radius:50%;animation:s 1s linear infinite}@keyframes s{to{transform:rotate(360deg)}}</style></head>
<body><main><h1>Starting Deep Research Studio</h1><p id="status">Preparing the private local research service…</p><div class="dot" id="spinner"></div></main></body></html>"#
}

fn show_failure(app: &tauri::AppHandle, message: &str) {
    if app
        .state::<DesktopState>()
        .exit_requested
        .load(Ordering::SeqCst)
    {
        return;
    }
    let window = match create_startup_window(app) {
        Ok(window) => window,
        Err(error) => {
            eprintln!("[desktop] could not display startup failure: {error}");
            app.exit(1);
            return;
        }
    };
    if let Some(main) = app.get_webview_window("main") {
        let _ = main.close();
    }
    let encoded = serde_json::to_string(message)
        .unwrap_or_else(|_| "\"The local service could not be started.\"".to_string());
    let script = format!(
        "document.querySelector('h1').textContent='Startup failed';document.getElementById('status').textContent={encoded};document.getElementById('spinner').style.display='none';"
    );
    let _ = window.eval(&script);
    let _ = window.show();
    let _ = window.set_focus();
}

fn show_main_window(app: &tauri::AppHandle) -> Result<(), Box<dyn std::error::Error>> {
    if app
        .state::<DesktopState>()
        .exit_requested
        .load(Ordering::SeqCst)
    {
        return Ok(());
    }
    let app_for_links = app.clone();
    let window = WebviewWindowBuilder::new(app, "main", WebviewUrl::App("index.html".into()))
        .title("Deep Research Studio")
        .inner_size(1280.0, 840.0)
        .min_inner_size(960.0, 640.0)
        .resizable(true)
        .on_navigation(allowed_main_navigation)
        .on_new_window(move |url, _features| {
            if matches!(url.scheme(), "http" | "https") {
                #[allow(deprecated)]
                if let Err(error) = app_for_links.shell().open(url.to_string(), None) {
                    eprintln!("[desktop] could not open the external link: {error}");
                }
            }
            // External URLs go to the system browser. All other popup requests are
            // rejected so untrusted content never receives a child WebView.
            NewWindowResponse::Deny
        })
        .build()?;
    window.show()?;
    window.set_focus()?;
    if let Some(startup) = app.get_webview_window("startup") {
        let _ = startup.close();
    }
    Ok(())
}

fn allowed_main_navigation(url: &Url) -> bool {
    let has_no_credentials = url.username().is_empty() && url.password().is_none();
    if url.scheme() == "http"
        && url.host_str() == Some("tauri.localhost")
        && url.port().is_none()
        && has_no_credentials
    {
        return true;
    }
    cfg!(debug_assertions)
        && url.scheme() == "http"
        && url.host_str() == Some("127.0.0.1")
        && url.port() == Some(5173)
        && has_no_credentials
}

fn desktop_data_dir(app: &tauri::AppHandle) -> Result<PathBuf, Box<dyn std::error::Error>> {
    if let Some(value) = std::env::var_os("DEEPRESEARCH_DESKTOP_DATA_DIR") {
        let path = PathBuf::from(value);
        if !path.is_absolute() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "DEEPRESEARCH_DESKTOP_DATA_DIR must be an absolute path",
            )
            .into());
        }
        return Ok(path);
    }
    Ok(app.path().app_local_data_dir()?.join("backend"))
}

fn spawn_api_sidecar(
    app: &tauri::AppHandle,
    data_dir: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut command = app
        .shell()
        .sidecar("deepresearch-api")?
        .env_clear()
        .env("DEEPRESEARCH_DESKTOP", "1")
        .env("DEEPRESEARCH_DESKTOP_DATA_DIR", data_dir)
        .env("DEEPRESEARCH_API_HOST", API_HOST)
        .env("DEEPRESEARCH_API_PORT", API_PORT);
    if cfg!(debug_assertions) {
        // Development uses the exact Vite origin declared by tauri.conf.json. The
        // packaged release never receives this exception.
        command = command.env("DEEPRESEARCH_DESKTOP_DEV", "1");
    }

    // The desktop profile must never silently inherit provider credentials, tracing
    // exporters, database locations, or another DeepResearch instance's configuration.
    // Keep only OS variables required by a packaged Windows process and its temp files.
    for key in [
        "APPDATA",
        "LOCALAPPDATA",
        "PATH",
        "PATHEXT",
        "PROGRAMDATA",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "WINDIR",
    ] {
        if let Some(value) = std::env::var_os(key) {
            command = command.env(key, value);
        }
    }
    let (mut events, child) = command.spawn()?;

    app.state::<DesktopState>()
        .child
        .lock()
        .map_err(|_| io::Error::other("sidecar state lock poisoned"))?
        .replace(child);

    let app_handle = app.clone();
    tauri::async_runtime::spawn(async move {
        while let Some(event) = events.recv().await {
            match event {
                CommandEvent::Stdout(bytes) => {
                    let line = String::from_utf8_lossy(&bytes);
                    if line.trim() == READY_MARKER {
                        let state = app_handle.state::<DesktopState>();
                        if claim_startup_ready(&state.startup)
                            && show_main_window(&app_handle).is_err()
                        {
                            state.startup.store(STARTUP_FAILED, Ordering::SeqCst);
                            show_failure(
                                &app_handle,
                                "The application window could not be opened. Close other copies of the app and try again.",
                            );
                            stop_api_after_failure(&app_handle);
                        }
                    }
                }
                CommandEvent::Stderr(_) => {
                    // The backend owns log redaction. Do not copy its raw diagnostic
                    // stream into a GUI error or a second unmanaged log channel.
                }
                CommandEvent::Terminated(_payload) => {
                    let state = app_handle.state::<DesktopState>();
                    if let Ok(mut slot) = state.child.lock() {
                        slot.take();
                    }
                    if state.exit_requested.load(Ordering::SeqCst) {
                        app_handle.exit(0);
                    } else if !state.expected_sidecar_stop.load(Ordering::SeqCst) {
                        state.startup.store(STARTUP_FAILED, Ordering::SeqCst);
                        show_failure(
                            &app_handle,
                            "The private local service stopped unexpectedly. Close other copies of the app and check local security permissions before trying again.",
                        );
                    }
                    break;
                }
                _ => {}
            }
        }
    });
    Ok(())
}

fn start_timeout_watch(app: &tauri::AppHandle) {
    let app_handle = app.clone();
    std::thread::spawn(move || {
        std::thread::sleep(STARTUP_TIMEOUT);
        let state = app_handle.state::<DesktopState>();
        if claim_startup_failed(&state.startup) && !state.exit_requested.load(Ordering::SeqCst) {
            show_failure(
                &app_handle,
                "The private local service did not become ready within 45 seconds.",
            );
            stop_api_after_failure(&app_handle);
        }
    });
}

fn hide_app_windows(app: &tauri::AppHandle) {
    if let Some(main) = app.get_webview_window("main") {
        let _ = main.hide();
    }
    if let Some(startup) = app.get_webview_window("startup") {
        let _ = startup.hide();
    }
}

fn start_exit_timeout(app: &tauri::AppHandle) {
    let app_handle = app.clone();
    std::thread::spawn(move || {
        std::thread::sleep(EXIT_TIMEOUT);
        if app_handle
            .state::<DesktopState>()
            .exit_requested
            .load(Ordering::SeqCst)
        {
            // Closing the process activates the Windows job object's whole-tree kill.
            app_handle.exit(0);
        }
    });
}

fn stop_api_after_failure(app: &tauri::AppHandle) {
    let state = app.state::<DesktopState>();
    state.startup.store(STARTUP_FAILED, Ordering::SeqCst);
    state.expected_sidecar_stop.store(true, Ordering::SeqCst);
    let child = state.child.lock().ok().and_then(|mut slot| slot.take());
    if let Some(mut child) = child {
        // sidecar.py treats any stdin line (or EOF) as a graceful shutdown request.
        // Give uvicorn a short drain interval, then force the direct child if needed;
        // the Windows job guard remains the final whole-tree cleanup boundary.
        let _ = child.write(b"shutdown\n");
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_secs(2));
            let _ = child.kill();
        });
    }
}

#[cfg(windows)]
fn install_process_tree_guard() -> io::Result<()> {
    windows_job::install()
}

#[cfg(not(windows))]
fn install_process_tree_guard() -> io::Result<()> {
    Ok(())
}

#[cfg(windows)]
fn fatal_startup_error() -> ! {
    use std::os::windows::ffi::OsStrExt;

    #[link(name = "user32")]
    extern "system" {
        fn MessageBoxW(
            window: *mut c_void,
            text: *const u16,
            caption: *const u16,
            kind: u32,
        ) -> i32;
    }

    let text: Vec<u16> = std::ffi::OsStr::new(
        "Deep Research Studio could not start its private local service. Close other copies of the app and check local security permissions before trying again.",
    )
    .encode_wide()
    .chain(Some(0))
    .collect();
    let caption: Vec<u16> = std::ffi::OsStr::new("Deep Research Studio — Startup failed")
        .encode_wide()
        .chain(Some(0))
        .collect();
    // SAFETY: the pointers reference NUL-terminated UTF-16 buffers that remain alive
    // until MessageBoxW returns. A null owner creates a process-modal error dialog.
    unsafe {
        MessageBoxW(std::ptr::null_mut(), text.as_ptr(), caption.as_ptr(), 0x10);
    }
    std::process::exit(1)
}

#[cfg(not(windows))]
fn fatal_startup_error() -> ! {
    eprintln!("Deep Research Studio could not start its private local service.");
    std::process::exit(1)
}

#[cfg(windows)]
mod windows_job {
    use super::{c_void, io};
    use std::mem::{size_of, zeroed};

    type Handle = *mut c_void;
    const JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE: u32 = 0x0000_2000;
    const JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS: i32 = 9;

    #[repr(C)]
    #[derive(Default)]
    struct BasicLimitInformation {
        per_process_user_time_limit: i64,
        per_job_user_time_limit: i64,
        limit_flags: u32,
        minimum_working_set_size: usize,
        maximum_working_set_size: usize,
        active_process_limit: u32,
        affinity: usize,
        priority_class: u32,
        scheduling_class: u32,
    }

    #[repr(C)]
    #[derive(Default)]
    struct IoCounters {
        read_operation_count: u64,
        write_operation_count: u64,
        other_operation_count: u64,
        read_transfer_count: u64,
        write_transfer_count: u64,
        other_transfer_count: u64,
    }

    #[repr(C)]
    #[derive(Default)]
    struct ExtendedLimitInformation {
        basic_limit_information: BasicLimitInformation,
        io_info: IoCounters,
        process_memory_limit: usize,
        job_memory_limit: usize,
        peak_process_memory_used: usize,
        peak_job_memory_used: usize,
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn CreateJobObjectW(attributes: *const c_void, name: *const u16) -> Handle;
        fn SetInformationJobObject(
            job: Handle,
            info_class: i32,
            info: *const c_void,
            info_length: u32,
        ) -> i32;
        fn AssignProcessToJobObject(job: Handle, process: Handle) -> i32;
        fn GetCurrentProcess() -> Handle;
        fn CloseHandle(object: Handle) -> i32;
    }

    pub(super) fn install() -> io::Result<()> {
        // SAFETY: all pointers are either null (documented defaults) or point to a
        // correctly sized repr(C) structure for the duration of each Win32 call.
        unsafe {
            let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if job.is_null() {
                return Err(io::Error::last_os_error());
            }
            let mut limits: ExtendedLimitInformation = zeroed();
            limits.basic_limit_information.limit_flags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            if SetInformationJobObject(
                job,
                JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
                &limits as *const _ as *const c_void,
                size_of::<ExtendedLimitInformation>() as u32,
            ) == 0
            {
                let error = io::Error::last_os_error();
                CloseHandle(job);
                return Err(error);
            }
            if AssignProcessToJobObject(job, GetCurrentProcess()) == 0 {
                let error = io::Error::last_os_error();
                CloseHandle(job);
                return Err(error);
            }

            // Intentionally keep the only handle open until process teardown. Closing it
            // earlier would activate KILL_ON_JOB_CLOSE and terminate this process too.
            let _job_handle_kept_until_process_exit = job;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        allowed_main_navigation, claim_startup_failed, claim_startup_ready, startup_html,
        STARTUP_FAILED, STARTUP_READY, STARTUP_STARTING,
    };
    use std::sync::{
        atomic::{AtomicU8, Ordering},
        Arc, Barrier,
    };
    use std::thread;

    #[test]
    fn startup_page_does_not_claim_readiness() {
        let html = startup_html();
        assert!(html.contains("Starting Deep Research Studio"));
        assert!(!html.contains("DEEPRESEARCH_DESKTOP_READY"));
    }

    #[test]
    fn release_navigation_is_confined_to_the_app_origin() {
        let local = "http://tauri.localhost/history".parse().unwrap();
        let lookalike = "http://tauri.localhost.evil.example/".parse().unwrap();
        let wrong_scheme = "https://tauri.localhost/".parse().unwrap();
        let explicit_port = "http://tauri.localhost:18789/".parse().unwrap();
        let userinfo = "http://user@tauri.localhost/".parse().unwrap();
        let remote = "https://example.com/".parse().unwrap();
        assert!(allowed_main_navigation(&local));
        assert!(!allowed_main_navigation(&lookalike));
        assert!(!allowed_main_navigation(&wrong_scheme));
        assert!(!allowed_main_navigation(&explicit_port));
        assert!(!allowed_main_navigation(&userinfo));
        assert!(!allowed_main_navigation(&remote));
    }

    #[test]
    fn ready_and_timeout_have_exactly_one_winner() {
        for _ in 0..64 {
            let startup = Arc::new(AtomicU8::new(STARTUP_STARTING));
            let barrier = Arc::new(Barrier::new(3));
            let ready_state = Arc::clone(&startup);
            let ready_barrier = Arc::clone(&barrier);
            let ready = thread::spawn(move || {
                ready_barrier.wait();
                claim_startup_ready(&ready_state)
            });
            let failed_state = Arc::clone(&startup);
            let failed_barrier = Arc::clone(&barrier);
            let failed = thread::spawn(move || {
                failed_barrier.wait();
                claim_startup_failed(&failed_state)
            });
            barrier.wait();
            let ready_won = ready.join().unwrap();
            let failed_won = failed.join().unwrap();
            assert_ne!(ready_won, failed_won);
            assert!(matches!(
                startup.load(Ordering::SeqCst),
                STARTUP_READY | STARTUP_FAILED
            ));
        }
    }

    #[test]
    fn timeout_then_ready_cannot_open_the_main_window() {
        let startup = AtomicU8::new(STARTUP_STARTING);
        assert!(claim_startup_failed(&startup));
        assert!(!claim_startup_ready(&startup));
        assert_eq!(startup.load(Ordering::SeqCst), STARTUP_FAILED);
    }

    #[test]
    fn ready_then_timeout_cannot_stop_the_sidecar() {
        let startup = AtomicU8::new(STARTUP_STARTING);
        assert!(claim_startup_ready(&startup));
        assert!(!claim_startup_failed(&startup));
        assert_eq!(startup.load(Ordering::SeqCst), STARTUP_READY);
    }
}
