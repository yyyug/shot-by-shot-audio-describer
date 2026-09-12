; Inno Setup script for Shot-by-Shot
; Creates a single-file Setup.exe with GUI wizard + uninstaller.
; Requires Inno Setup 6 (https://jrsoftware.org/isinfo.php)
;
; Build: iscc packaging\ShotByShot.iss
; (or run build.ps1 which drives PyInstaller then ISCC automatically)

#define MyAppName "Shot-by-Shot"
#define MyAppVersion "1.0.0"
#define MyAppExeName "ShotByShotDesktop.exe"

[Setup]
AppId={{D0E6A4D2-9B45-4C6E-9A37-2F1B6A3C8D4E}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher=Shot-by-Shot
DefaultDirName={autopf}\Shot-by-Shot
DisableDirPage=no
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=..\dist
OutputBaseFilename=ShotByShot-Setup
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
SetupIconFile=shot.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
UninstallDisplayName={#MyAppName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\dist\ShotByShotPortable\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; Evergreen WebView2 Runtime bootstrapper (~2MB, needs network when run).
; Only executed on machines where the runtime is missing (fresh Win10 etc.).
Source: "redist\MicrosoftEdgeWebview2Setup.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall

[Dirs]
; Pre-create the data folder so the "Outputs" shortcut works even before first run.
Name: "{localappdata}\ShotByShot"

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{group}\{#MyAppName} (Web)"; Filename: "{app}\ShotByShotWeb.exe"; WorkingDir: "{app}"; Comment: "Run in the browser (local web server)"
Name: "{group}\{#MyAppName} Outputs"; Filename: "{localappdata}\ShotByShot"; Comment: "Open output CSVs and log folder"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Code]
const
  Webview2Guid = '{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}';

function Webview2Installed(): Boolean;
begin
  Result :=
    RegKeyExists(HKLM, 'SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\' + Webview2Guid) or
    RegKeyExists(HKLM, 'SOFTWARE\Microsoft\EdgeUpdate\Clients\' + Webview2Guid) or
    RegKeyExists(HKCU, 'Software\Microsoft\EdgeUpdate\Clients\' + Webview2Guid);
end;

[Run]
Filename: "{tmp}\MicrosoftEdgeWebview2Setup.exe"; \
  Parameters: "/install /quiet /norestart"; \
  StatusMsg: "Installing Microsoft Edge WebView2 Runtime..."; \
  Check: "not Webview2Installed"
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent
