; Installeur d'Eyetrack (Inno Setup 6). Construit par : python tools/build_exe.py --installer
;
; - installation pour l'utilisateur, sans droits administrateur (%LOCALAPPDATA%\Programs\Eyetrack) ;
; - français ou anglais : la langue choisie ici devient celle d'Eyetrack (si pas encore réglée) ;
; - OpenTrack proposé s'il n'est pas installé (winget, sinon page de téléchargement) ;
; - modèle de regard de précision NON inclus (licence non commerciale d'un tiers) : proposé en
;   téléchargement depuis sa source officielle, case décochée, après acceptation de sa licence.
;   Installation silencieuse : /DOWNLOADMODEL=1 pour le télécharger ;
; - la désinstallation garde la calibration et les réglages, sauf si l'utilisateur demande de
;   les effacer.

#ifndef AppVersion
  #define AppVersion "dev"
#endif
#ifndef SourceDir
  #define SourceDir "..\dist\Eyetrack"
#endif
; Mêmes valeurs que gaze_model.py (vérifié par tests/test_modele.py)
#define ModelFile "mobileone_s0_gaze.onnx"
#define ModelUrl "https://github.com/yakhyo/gaze-estimation/releases/download/weights/mobileone_s0_gaze.onnx"
#define ModelSource "https://github.com/yakhyo/gaze-estimation"
#define ModelSha256 "8b4fdc4e3da44733c9a82e7776b411e4a39f94e8e285aee0fc85a548a55f7d9f"

[Setup]
AppId={{5C2949D4-5DE9-47C6-B3F5-5FC67F400126}
AppName=Eyetrack
AppVersion={#AppVersion}
AppVerName=Eyetrack {#AppVersion}
AppPublisher=Eyetrack
DefaultDirName={autopf}\Eyetrack
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist\installer
OutputBaseFilename=Eyetrack-Setup
SetupIconFile=..\assets\eyetrack.ico
UninstallDisplayIcon={app}\Eyetrack.exe
UninstallDisplayName=Eyetrack
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ShowLanguageDialog=yes
LanguageDetectionMethod=uilanguage
; Eyetrack lancé : l'installeur demande de le quitter (verrou d'instance unique de winapp.py)
AppMutex=Local\Eyetrack.Tracker
CloseApplications=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "french"; MessagesFile: "compiler:Languages\French.isl"

[CustomMessages]
english.DesktopIcon=Create a desktop shortcut
french.DesktopIcon=Créer un raccourci sur le Bureau
english.OpenTrackTask=Install OpenTrack (it passes your head movements to the game). Downloaded with winget; Windows will ask for permission.
french.OpenTrackTask=Installer OpenTrack (il transmet les mouvements de la tête au jeu). Téléchargé avec winget ; Windows demandera une autorisation.
english.InstallingOpenTrack=Installing OpenTrack...
french.InstallingOpenTrack=Installation d'OpenTrack...
english.ModelTitle=Precision gaze model (optional)
french.ModelTitle=Modèle de regard de précision (optionnel)
english.ModelSubtitle=More precise gaze tracking, from a third party, for non-commercial use only.
french.ModelSubtitle=Un suivi du regard plus précis, fourni par un tiers, pour un usage non commercial.
english.ModelText=Eyetrack works without it, in simple mode. The precision model (MobileOne-S0, trained on the Gaze360 dataset) is not included with Eyetrack: it belongs to a third party and its license only allows non-commercial use.%n%nIf you tick the box, Setup downloads it from its official source and checks it:%n{#ModelSource}%n%nYou can also download it later from the Eyetrack panel.
french.ModelText=Eyetrack fonctionne sans lui, en mode simple. Le modèle de précision (MobileOne-S0, entraîné sur le jeu de données Gaze360) n'est pas fourni avec Eyetrack : il appartient à un tiers et sa licence n'autorise qu'un usage non commercial.%n%nSi vous cochez la case, l'installeur le télécharge depuis sa source officielle et le vérifie :%n{#ModelSource}%n%nVous pourrez aussi le télécharger plus tard depuis le panneau d'Eyetrack.
english.ModelCheck=Download the precision model and accept its license (personal, non-commercial use)
french.ModelCheck=Télécharger le modèle de précision et accepter sa licence (usage personnel, non commercial)
english.Downloading=Downloading the precision gaze model...
french.Downloading=Téléchargement du modèle de regard de précision...
english.ModelFailed=The precision model could not be downloaded:%n%1%n%nEyetrack will use simple mode; you can download the model later from its panel.
french.ModelFailed=Le modèle de précision n'a pas pu être téléchargé :%n%1%n%nEyetrack utilisera le mode simple ; vous pourrez télécharger le modèle plus tard depuis son panneau.
english.LaunchApp=Start Eyetrack
french.LaunchApp=Lancer Eyetrack
english.DeleteData=Also delete your Eyetrack calibration and settings?%n%n%1%n%nChoose No to keep them for a future installation.
french.DeleteData=Supprimer aussi votre calibration et vos réglages d'Eyetrack ?%n%n%1%n%nChoisissez Non pour les garder en vue d'une prochaine installation.

[Tasks]
Name: "desktopicon"; Description: "{cm:DesktopIcon}"
Name: "opentrack"; Description: "{cm:OpenTrackTask}"; Check: not OpenTrackInstalled

[InstallDelete]
; Ancienne version : bibliothèques retirées (aucun mélange de versions de DLL)
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\Eyetrack"; Filename: "{app}\Eyetrack.exe"; WorkingDir: "{app}"
Name: "{autodesktop}\Eyetrack"; Filename: "{app}\Eyetrack.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{code:WingetExe}"; Parameters: "install --id opentrack.opentrack -e --source winget"; StatusMsg: "{cm:InstallingOpenTrack}"; Tasks: opentrack; Check: WingetAvailable; Flags: waituntilterminated
Filename: "https://github.com/opentrack/opentrack/releases"; Tasks: opentrack; Check: not WingetAvailable; Flags: shellexec nowait
Filename: "{app}\Eyetrack.exe"; Description: "{cm:LaunchApp}"; WorkingDir: "{app}"; Flags: nowait postinstall skipifsilent

[Code]
var
  ModelPage: TInputOptionWizardPage;

function UserDataDir: String;
begin
  { Données de l'utilisateur (paths.py : USER_DIR), distinctes du programme }
  Result := ExpandConstant('{localappdata}\Eyetrack');
end;

function ModelPath: String;
begin
  Result := UserDataDir + '\models\{#ModelFile}';
end;

function OpenTrackInstalled: Boolean;
begin
  { Mêmes emplacements que winapp.find_opentrack }
  Result := FileExists(ExpandConstant('{commonpf32}\opentrack\opentrack.exe')) or
            FileExists(ExpandConstant('{commonpf64}\opentrack\opentrack.exe')) or
            FileExists(ExpandConstant('{localappdata}\opentrack\opentrack.exe')) or
            FileExists(ExpandConstant('{localappdata}\Programs\opentrack\opentrack.exe'));
end;

function WingetExe(Param: String): String;
begin
  Result := ExpandConstant('{localappdata}\Microsoft\WindowsApps\winget.exe');
end;

function WingetAvailable: Boolean;
begin
  Result := FileExists(WingetExe(''));
end;

procedure InitializeWizard;
begin
  ModelPage := CreateInputOptionPage(wpSelectTasks, CustomMessage('ModelTitle'),
    CustomMessage('ModelSubtitle'), CustomMessage('ModelText'), False, False);
  ModelPage.Add(CustomMessage('ModelCheck'));
  ModelPage.Values[0] := False;  { décochée : accord explicite de l'utilisateur }
end;

function ShouldSkipPage(PageID: Integer): Boolean;
begin
  Result := (PageID = ModelPage.ID) and FileExists(ModelPath);
end;

function WantModel: Boolean;
begin
  if WizardSilent then
    Result := ExpandConstant('{param:DOWNLOADMODEL|0}') = '1'
  else
    Result := ModelPage.Values[0];
  Result := Result and not FileExists(ModelPath);
end;

function OnDownloadProgress(const Url, FileName: String; const Progress, ProgressMax: Int64): Boolean;
begin
  Result := True;
end;

procedure DownloadModel;
var
  Tmp: String;
begin
  WizardForm.StatusLabel.Caption := CustomMessage('Downloading');
  try
    { Source officielle, empreinte SHA-256 vérifiée par Inno Setup avant d'aller plus loin }
    DownloadTemporaryFile('{#ModelUrl}', '{#ModelFile}', '{#ModelSha256}', @OnDownloadProgress);
    Tmp := ExpandConstant('{tmp}\{#ModelFile}');
    ForceDirectories(ExtractFileDir(ModelPath));
    if not FileCopy(Tmp, ModelPath, False) then
      RaiseException('copie impossible vers ' + ModelPath);
    Log('Modèle de précision installé : ' + ModelPath);
  except
    Log('Échec du téléchargement du modèle : ' + GetExceptionMessage);
    SuppressibleMsgBox(FmtMessage(CustomMessage('ModelFailed'), [GetExceptionMessage]), mbError, MB_OK, IDOK);
  end;
end;

procedure WriteLanguage;
var
  F: String;
begin
  { Langue de l'installeur reprise par Eyetrack, seulement si l'utilisateur ne l'a pas déjà réglée }
  F := UserDataDir + '\data\app_settings.json';
  if not FileExists(F) then
  begin
    ForceDirectories(ExtractFileDir(F));
    if ActiveLanguage = 'french' then
      SaveStringToFile(F, '{"language": "fr"}', False)
    else
      SaveStringToFile(F, '{"language": "en"}', False);
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    WriteLanguage;
    if WantModel then
      DownloadModel;
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  { Calibration et réglages gardés par défaut (et toujours en désinstallation silencieuse) }
  if (CurUninstallStep = usPostUninstall) and DirExists(UserDataDir) and not UninstallSilent then
    if MsgBox(FmtMessage(CustomMessage('DeleteData'), [UserDataDir]), mbConfirmation,
              MB_YESNO or MB_DEFBUTTON2) = IDYES then
      DelTree(UserDataDir, True, True, True);
end;
