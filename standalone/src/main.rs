#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};

use tauri::{Manager, RunEvent, Url};

const BACKEND_PORT: u16 = 5000;
const BACKEND_URL: &str = "http://127.0.0.1:5000/";
const STARTUP_TIMEOUT: Duration = Duration::from_secs(150);
const CREATE_NO_WINDOW: u32 = 0x0800_0000;
const LOG_MAX_BYTES: u64 = 5 * 1024 * 1024;
const LOG_ROTATIONS: u32 = 3;

static EXITING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn log_dir() -> PathBuf {
    let base = std::env::var("LOCALAPPDATA").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("."));
    base.join("BuddyAd")
}

fn rotate_log() {
    let dir = log_dir();
    let _ = std::fs::create_dir_all(&dir);
    let base = dir.join("buddyad.log");
    for n in (1..=LOG_ROTATIONS).rev() {
        let to = dir.join(format!("buddyad.log.{}", n + 1));
        let _ = std::fs::remove_file(&to);
        let from = if n == 1 {
            base.clone()
        } else {
            dir.join(format!("buddyad.log.{n}"))
        };
        let _ = std::fs::rename(&from, &to);
    }
    let _ = std::fs::remove_file(dir.join(format!("buddyad.log.{}", LOG_ROTATIONS + 1)));
}

fn log_write(msg: impl AsRef<str>) {
    let dir = log_dir();
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("buddyad.log");
    if let Ok(meta) = std::fs::metadata(&path) {
        if meta.len() >= LOG_MAX_BYTES {
            rotate_log();
        }
    }
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(f, "{} {}", chrono_like_now(), msg.as_ref());
    }
}

fn chrono_like_now() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    format!("[{secs}]")
}

fn backend_alive(port: u16) -> bool {
    let mut stream = match TcpStream::connect(("127.0.0.1", port)) {
        Ok(s) => s,
        Err(_) => return false,
    };
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(2)));
    if stream.write_all(b"GET / HTTP/1.0\r\nConnection: close\r\n\r\n").is_err() {
        return false;
    }
    let mut buf = [0u8; 256];
    let mut got = 0usize;
    while got < buf.len() {
        match stream.read(&mut buf[got..]) {
            Ok(0) | Err(_) => break,
            Ok(n) => got += n,
        }
    }
    let head = String::from_utf8_lossy(&buf[..got]);
    head.contains(" 200 ")
}

struct Backend {
    child: Option<Child>,
}

fn prepare_command(mut cmd: Command) -> Command {
    cmd.env("SBS_PORT", BACKEND_PORT.to_string());
    cmd.env("SBS_NO_BROWSER", "1");
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd
}

fn backend_command() -> Option<Command> {
    if let Ok(p) = std::env::var("BUDDY_AD_BACKEND") {
        if !p.trim().is_empty() {
            log_write(&format!("using BUDDY_AD_BACKEND={p}"));
            return Some(prepare_command(Command::new(p)));
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let bundled = dir.join("BuddyADWeb.exe");
            if bundled.exists() {
                log_write("using bundled BuddyADWeb.exe");
                return Some(prepare_command(Command::new(bundled)));
            }
        }
    }
    if cfg!(debug_assertions) {
        let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
        for py in ["webapp_entry.py"] {
            let script = repo.join(py);
            if script.exists() {
                log_write(&format!("dev: python {}", script.display()));
                let mut cmd = Command::new("python");
                cmd.arg(script);
                return Some(prepare_command(cmd));
            }
        }
    }
    None
}

#[derive(Clone)]
enum StartFail {
    Missing,
    SpawnFailed(String),
    Timeout,
}

fn start_backend(already_running: bool) -> (Option<Child>, Option<StartFail>) {
    if already_running {
        return (None, None);
    }
    let mut cmd = match backend_command() {
        Some(c) => c,
        None => return (None, Some(StartFail::Missing)),
    };
    match cmd.spawn() {
        Ok(child) => (Some(child), None),
        Err(e) => (None, Some(StartFail::SpawnFailed(e.to_string()))),
    }
}

fn kill_backend(app: &tauri::AppHandle) {
    if let Some(b) = app.try_state::<Mutex<Backend>>() {
        if let Ok(mut guard) = b.lock() {
            if let Some(mut child) = guard.child.take() {
                let _ = child.kill();
                let _ = child.wait();
                log_write("backend child stopped");
            }
        }
    }
}

fn percent_encode(s: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut out = String::new();
    for &b in s.as_bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~') {
            out.push(b as char);
        } else {
            out.push('%');
            out.push(HEX[(b >> 4) as usize] as char);
            out.push(HEX[(b & 0x0f) as usize] as char);
        }
    }
    out
}

fn data_url_for_error(t_zh: &str, t_en: &str, m_zh: &str, m_en: &str) -> Option<Url> {
    let html = format!(
        "<!doctype html><meta charset=\"utf-8\"><title>Buddy AD</title>\
         <style>body{{font-family:'Segoe UI',system-ui,sans-serif;background:#f5f6f8;color:#1f2328;margin:0;height:100vh;display:flex;align-items:center;justify-content:center;}}\
         .box{{max-width:560px;text-align:center;padding:24px;}}h1{{font-size:20px;margin:0 0 10px;}}p{{font-size:14px;color:#57606a;line-height:1.55;margin:6px 0;}}</style>\
         <body><div class=\"box\"><h1>{t_zh} / {t_en}</h1><p>{m_zh}</p><p>{m_en}</p></div></body>"
    );
    Url::parse(&format!("data:text/html,{}", percent_encode(&html))).ok()
}

fn show_error_and_exit(handle: &tauri::AppHandle, kind: StartFail, kill: bool) -> ! {
    if kill {
        kill_backend(handle);
    }
    let (t_zh, t_en, m_zh, m_en) = match &kind {
        StartFail::Missing => (
            "後端遺失".to_string(),
            "Backend missing".to_string(),
            "搵唔到後端 BuddyADWeb.exe。請確保 BuddyAD.exe 同 BuddyADWeb.exe 位於同一個資料夾，或設定 BUDDY_AD_BACKEND 環境變數。".to_string(),
            "Backend BuddyADWeb.exe was not found next to BuddyAD.exe. Keep both files in the same folder, or set the BUDDY_AD_BACKEND environment variable.".to_string(),
        ),
        StartFail::SpawnFailed(e) => (
            "啟動失敗".to_string(),
            "Launch failed".to_string(),
            format!("後端程序啟動失敗：{e}。詳情請睇 buddyad.log。"),
            format!("Failed to launch the backend process: {e}. See buddyad.log for details."),
        ),
        StartFail::Timeout => (
            "啟動逾時".to_string(),
            "Startup timeout".to_string(),
            "後端未能喺指定時間內啟動。程式即將關閉，詳情請睇 buddyad.log。".to_string(),
            "The backend did not become ready in time. The app will close now. See buddyad.log for details.".to_string(),
        ),
    };
    log_write("startup failed, showing error page");
    if let Some(url) = data_url_for_error(&t_zh, &t_en, &m_zh, &m_en) {
        if let Some(win) = handle.get_webview_window("main") {
            let _ = win.navigate(url);
        }
    }
    thread::sleep(Duration::from_secs(8));
    std::process::exit(1);
}

fn main() {
    let already_running = backend_alive(BACKEND_PORT);
    log_write(if already_running {
        format!("backend already running on :{BACKEND_PORT}, not spawning")
    } else {
        "backend not running, will spawn".to_string()
    });

    let (child, fail) = start_backend(already_running);
    let child_is_some = child.is_some();
    let child_killable = child_is_some;
    let already = already_running;

    tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            if let Some(w) = app.get_webview_window("main") {
                let _ = w.set_focus();
            }
        }))
        .manage(Mutex::new(Backend { child }))
        .setup(move |app| {
            let handle = app.handle().clone();
            thread::spawn(move || {
                if let Some(kind) = fail {
                    show_error_and_exit(&handle, kind, child_killable);
                }
                let deadline = Instant::now() + STARTUP_TIMEOUT;
                loop {
                    if backend_alive(BACKEND_PORT) {
                        break;
                    }
                    if Instant::now() >= deadline {
                        log_write("backend did not become ready, showing error");
                        show_error_and_exit(&handle, StartFail::Timeout, child_killable);
                    }
                    thread::sleep(Duration::from_millis(500));
                }
                thread::sleep(Duration::from_millis(if already { 150 } else { 400 }));
                if let Some(win) = handle.get_webview_window("main") {
                    if let Ok(url) = Url::parse(BACKEND_URL) {
                        log_write("navigating to backend");
                        let _ = win.navigate(url);
                    }
                }
            });
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(move |app, event| match event {
            RunEvent::ExitRequested { .. } => {
                if !EXITING.swap(true, std::sync::atomic::Ordering::SeqCst) {
                    log_write("exit requested, cleaning up");
                    if child_is_some {
                        kill_backend(app);
                    }
                }
                std::process::exit(0);
            }
            RunEvent::Exit => {
                if !EXITING.swap(true, std::sync::atomic::Ordering::SeqCst) {
                    log_write("app exited");
                    if child_is_some {
                        kill_backend(app);
                    }
                }
            }
            _ => {}
        });
}