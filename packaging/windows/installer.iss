[Setup]
AppName=Histogram Viewer
AppVersion=0.1.0
DefaultDirName={autopf}\HistogramViewer
DefaultGroupName=Histogram Viewer
OutputDir=output
OutputBaseFilename=HistogramViewerSetup
Compression=none
PrivilegesRequiredOverridesAllowed=commandline

[Files]
Source: "..\..\dist\HistogramViewer.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\Histogram Viewer"; Filename: "{app}\HistogramViewer.exe"
Name: "{autodesktop}\Histogram Viewer"; Filename: "{app}\HistogramViewer.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop icon"; GroupDescription: "Additional icons:"

[UninstallDelete]
Type: filesandordirs; Name: "{app}"
