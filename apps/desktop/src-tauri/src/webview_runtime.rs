//! Recover when EdgeUpdate registers a different Evergreen version than the one on disk.

use std::ffi::OsString;
use std::fs;
use std::mem::size_of;
use std::os::windows::ffi::{OsStrExt, OsStringExt};
use std::path::{Path, PathBuf};
use std::ptr::{null, null_mut};
use windows_sys::Win32::Foundation::MAX_PATH;
use windows_sys::Win32::Security::Cryptography::{
    CertGetNameStringW, CERT_NAME_SIMPLE_DISPLAY_TYPE,
};
use windows_sys::Win32::Security::WinTrust::*;
use windows_sys::Win32::UI::Shell::{
    SHGetFolderPathW, CSIDL_PROGRAM_FILES, CSIDL_PROGRAM_FILESX86,
};

const OVERRIDE: &str = "WEBVIEW2_BROWSER_EXECUTABLE_FOLDER";

pub(super) fn recover_stale_registration() {
    if std::env::var_os(OVERRIDE).is_some_and(|folder| !folder.is_empty())
        || tauri::webview_version().is_ok()
    {
        return;
    }

    let mut versions = Vec::new();
    for id in [CSIDL_PROGRAM_FILESX86, CSIDL_PROGRAM_FILES] {
        let Some(root) = program_files(id) else {
            continue;
        };
        let Ok(canonical_root) = root.canonicalize() else {
            continue;
        };
        let application = Path::new(r"Microsoft\EdgeWebView\Application");
        let Ok(entries) = fs::read_dir(root.join(application)) else {
            continue;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            if let Some(version) = name.to_str().and_then(parse_version) {
                versions.push((
                    version,
                    entry.path(),
                    canonical_root.join(application).join(name),
                ));
            }
        }
    }
    versions.sort_unstable_by_key(|(version, _, _)| std::cmp::Reverse(*version));
    for (_, folder, expected) in versions {
        // The canonical path must still match the exact location under Program Files.
        // This also checks intermediate junctions and links to individual loader files.
        if ![
            "msedgewebview2.exe",
            r"EBWebView\x64\EmbeddedBrowserWebView.dll",
        ]
        .iter()
        .all(|file| {
            let path = folder.join(file);
            path.canonicalize().is_ok_and(|p| p == expected.join(file)) && microsoft_signed(&path)
        }) {
            continue;
        }
        std::env::set_var(OVERRIDE, &folder);
        if tauri::webview_version().is_ok() {
            return;
        }
        std::env::remove_var(OVERRIDE);
    }
}

fn program_files(id: u32) -> Option<PathBuf> {
    let mut path = [0u16; MAX_PATH as usize];
    // SAFETY: SHGetFolderPathW requires a writable MAX_PATH buffer.
    if unsafe { SHGetFolderPathW(null_mut(), id as i32, null_mut(), 0, path.as_mut_ptr()) } != 0 {
        return None;
    }
    let length = path.iter().position(|&c| c == 0)?;
    Some(OsString::from_wide(&path[..length]).into())
}

fn parse_version(name: &str) -> Option<[u32; 4]> {
    let mut parts = name.split('.');
    let mut version = [0; 4];
    for part in &mut version {
        let text = parts.next()?;
        if !text.bytes().all(|c| c.is_ascii_digit()) {
            return None;
        }
        *part = text.parse().ok()?;
    }
    parts.next().is_none().then_some(version)
}

fn microsoft_signed(path: &Path) -> bool {
    let wide: Vec<_> = path.as_os_str().encode_wide().chain(Some(0)).collect();
    let mut file = WINTRUST_FILE_INFO {
        cbStruct: size_of::<WINTRUST_FILE_INFO>() as u32,
        pcwszFilePath: wide.as_ptr(),
        ..Default::default()
    };
    let mut data = WINTRUST_DATA {
        cbStruct: size_of::<WINTRUST_DATA>() as u32,
        dwUIChoice: WTD_UI_NONE,
        dwUnionChoice: WTD_CHOICE_FILE,
        Anonymous: WINTRUST_DATA_0 { pFile: &mut file },
        dwStateAction: WTD_STATEACTION_VERIFY,
        dwProvFlags: WTD_CACHE_ONLY_URL_RETRIEVAL,
        ..Default::default()
    };
    let mut policy = WINTRUST_ACTION_GENERIC_VERIFY_V2;
    // SAFETY: file, filename and state remain valid through both WinVerifyTrust calls.
    // The verified chain's borrowed pointers are only inspected before closing its state.
    unsafe {
        let verified = WinVerifyTrustEx(null_mut(), &mut policy, &mut data) == 0;
        let signed_by_microsoft = verified
            && (|| {
                let provider = WTHelperProvDataFromStateData(data.hWVTStateData);
                if provider.is_null() {
                    return None;
                }
                let signer = WTHelperGetProvSignerFromChain(provider, 0, 0, 0);
                if signer.is_null() {
                    return None;
                }
                let cert = WTHelperGetProvCertFromChain(signer, 0).as_ref()?;
                let mut name = [0u16; 128];
                let len = CertGetNameStringW(
                    cert.pCert,
                    CERT_NAME_SIMPLE_DISPLAY_TYPE,
                    0,
                    null(),
                    name.as_mut_ptr(),
                    name.len() as u32,
                );
                (len > 1
                    && name[..len as usize - 1]
                        .iter()
                        .copied()
                        .eq("Microsoft Corporation".encode_utf16()))
                .then_some(())
            })()
            .is_some();
        data.dwStateAction = WTD_STATEACTION_CLOSE;
        WinVerifyTrustEx(null_mut(), &mut policy, &mut data);
        signed_by_microsoft
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evergreen_candidates_are_four_numeric_components() {
        assert_eq!(parse_version("157.0.4341.46"), Some([157, 0, 4341, 46]));
        assert!(parse_version("157.0.4341.46") > parse_version("99.9.9999.99"));
        for name in [
            "SetupMetrics",
            "1.2.3",
            "1.2.3.4.5",
            "1.2.3.+4",
            "1.2.3.bad",
        ] {
            assert!(parse_version(name).is_none());
        }
    }

    #[test]
    fn unsigned_executable_cannot_supply_the_webview() {
        // The local Rust test harness has no publisher signature.
        assert!(!microsoft_signed(&std::env::current_exe().unwrap()));
    }

    #[test]
    #[ignore = "requires a machine-installed x64 Evergreen runtime"]
    fn authenticates_installed_runtime_files() {
        let root = program_files(CSIDL_PROGRAM_FILESX86)
            .unwrap()
            .join(r"Microsoft\EdgeWebView\Application");
        assert!(fs::read_dir(root).unwrap().flatten().any(|entry| {
            parse_version(&entry.file_name().to_string_lossy()).is_some()
                && microsoft_signed(&entry.path().join("msedgewebview2.exe"))
                && microsoft_signed(
                    &entry
                        .path()
                        .join(r"EBWebView\x64\EmbeddedBrowserWebView.dll"),
                )
        }));
    }

    #[test]
    #[ignore = "requires a machine-installed x64 Evergreen runtime; run in a separate process"]
    fn recovers_machine_runtime() {
        for initial in [None, Some(OsString::new())] {
            match &initial {
                Some(value) => std::env::set_var(OVERRIDE, value),
                None => std::env::remove_var(OVERRIDE),
            }
            let default_version = tauri::webview_version().ok();
            recover_stale_registration();
            let version = tauri::webview_version().expect("runtime available after recovery");
            let folder = std::env::var_os(OVERRIDE);
            if let Some(default) = default_version {
                assert_eq!(folder, initial);
                assert_eq!(version, default);
            } else {
                let folder = PathBuf::from(folder.expect("recovered machine runtime folder"));
                assert_eq!(
                    folder.file_name().and_then(|n| n.to_str()),
                    Some(version.as_str())
                );
            }
        }
    }
}
