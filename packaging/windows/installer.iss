[Setup]
AppName=SpectraTools
AppVersion=2.1.1
DefaultDirName={autopf}\SpectraTools
DefaultGroupName=SpectraTools
OutputDir=output
OutputBaseFilename=SpectraToolsSetup
Compression=none
PrivilegesRequiredOverridesAllowed=commandline

[Files]
Source: "..\..\dist\SpectraTools.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\SpectraTools"; Filename: "{app}\SpectraTools.exe"
Name: "{autodesktop}\SpectraTools"; Filename: "{app}\SpectraTools.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop icon"; GroupDescription: "Additional icons:"; Flags: unchecked

[UninstallDelete]
Type: filesandordirs; Name: "{app}"
