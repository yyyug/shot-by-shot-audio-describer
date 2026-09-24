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

static EXITING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn log_dir() -> PathBuf {
    let base = std::env::var("LOCALAPPDATA").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("."));
    base.join("BuddyAd")
}

fn log_write(msg: impl AsRef<str>) {
    let dir = log_dir();
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("buddyad_standalone.log");
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

fn spawn_backend() -> Option<Child> {
    let mut cmd = backend_command()?;
    match cmd.spawn() {
        Ok(child) => Some(child),
        Err(e) => {
            log_write(&format!("failed to spawn backend: {e}"));
            None
        }
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

fn main() {
    let already_running = backend_alive(BACKEND_PORT);
    log_write(if already_running {
        format!("backend already running on :{BACKEND_PORT}, not spawning")
    } else {
        "backend not running, will spawn".to_string()
    });

    let child = if already_running { None } else { spawn_backend() };
    let child_is_some = child.is_some();
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
                let deadline = Instant::now() + STARTUP_TIMEOUT;
                loop {
                    if backend_alive(BACKEND_PORT) {
                        break;
                    }
                    if Instant::now() >= deadline {
                        log_write("backend did not become ready, exiting");
                        handle.exit(1);
                        return;
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
            RunEvent::ExitRequested { api, .. } => {
                if !EXITING.swap(true, std::sync::atomic::Ordering::SeqCst) {
                    api.prevent_exit();
                    log_write("exit requested, cleaning up");
                    if child_is_some {
                        kill_backend(app);
                    }
                    app.exit(0);
                } else {
                    log_write("exit requested, already exiting");
                    app.exit(0);
                }
            }
            RunEvent::Exit => {
                log_write("app exiting");
                if child_is_some {
                    kill_backend(app);
                }
            }
            _ => {}
        });
}