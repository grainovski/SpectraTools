; SpectraTools Inno Setup script
; Build: cd <project-root> && .\packaging\windows\build.ps1
; Output: packaging\windows\output\SpectraToolsSetup.exe

#define AppName    "SpectraTools"
#define AppVersion "5.0.2"
#define AppExe     "SpectraTools.exe"

[Setup]
AppId={{A8F3C2D1-5E74-4B92-8301-7C6E9D0A4F25}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} v{#AppVersion}
DefaultDirName={autopf}\SpectraTools
DefaultGroupName=SpectraTools
OutputDir=output
OutputBaseFilename=SpectraToolsSetup
SetupIconFile=..\..\assets\icon.ico
UninstallDisplayIcon={app}\{#AppExe}
; lzma2/max + solid: the onedir tree is hundreds of loose DLLs and .pyd files
; that compress well; solid mode exploits their similarity.  (CalEnEff uses zip
; instead because MKL DLLs crash ISCC under lzma2; SpectraTools has no MKL.)
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
DisableProgramGroupPage=yes
; Allow non-admin install to AppData\Local\Programs, or admin to Program Files
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
MinVersion=10.0
; App is 64-bit — install to Program Files, not Program Files (x86)
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=yes
RestartApplications=no

[Files]
; The whole --onedir tree (SpectraTools.exe plus its _internal\ folder).
; recursesubdirs/createallsubdirs are both required: without them Inno
; silently ships only the top-level files and the app dies at startup.
Source: "..\..\dist\SpectraTools\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked

[Run]
Filename: "{app}\{#AppExe}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}"

; ── Uninstall-previous-version logic ────────────────────────────────────────
; Detects an existing installation via the AppId registry key and asks whether
; to remove it before installing the new version.
[Code]
function GetUninstallString(Hive: Integer): String;
var
  RegPath: String;
  UninstStr: String;
begin
  RegPath   := 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{A8F3C2D1-5E74-4B92-8301-7C6E9D0A4F25}_is1';
  UninstStr := '';
  RegQueryStringValue(Hive, RegPath, 'UninstallString', UninstStr);
  Result := UninstStr;
end;

function FindPreviousUninstaller(): String;
var
  S: String;
begin
  S := GetUninstallString(HKLM);
  if S = '' then S := GetUninstallString(HKCU);
  Result := S;
end;

function InitializeSetup(): Boolean;
var
  UninstStr:  String;
  UninstExe:  String;
  ResultCode: Integer;
  Answer:     Integer;
begin
  Result    := True;
  UninstStr := FindPreviousUninstaller();
  if UninstStr = '' then Exit;

  Answer := MsgBox(
    'A previous installation of {#AppName} was found.' + #13#10 +
    'It is recommended to remove it before installing a new version.' + #13#10#13#10 +
    'Do you want to uninstall the previous version now?',
    mbConfirmation, MB_YESNO);

  if Answer = IDYES then begin
    UninstExe := RemoveQuotes(UninstStr);
    if not Exec(UninstExe, '/SILENT /NORESTART', '', SW_SHOW,
                ewWaitUntilTerminated, ResultCode) then begin
      MsgBox('The uninstaller could not be launched: ' + SysErrorMessage(ResultCode),
             mbError, MB_OK);
    end;
  end;
end;
