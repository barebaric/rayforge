; NSIS Script for Rayforge Installer

!include "LogicLib.nsh"

;--------------------------------
; Defines
; These are variables passed from our build script using the -D flag
!define PRODUCT_NAME "Rayforge"
!ifndef APP_VERSION
  !define APP_VERSION "0.0.0"
!endif
!ifndef APP_DIR_NAME
  !define APP_DIR_NAME "rayforge-v0.0.0"
!endif
!ifndef EXECUTABLE_NAME
  !define EXECUTABLE_NAME "rayforge.exe"
!endif
!ifndef ICON_FILE
  !define ICON_FILE "rayforge.ico"
!endif

;--------------------------------
; Variables
Var PreviousInstallDir
Var MovedInstallDir

;--------------------------------
; General

RequestExecutionLevel admin ; Request admin rights for installation
SetCompressor lzma ; Use modern, efficient compression

; Installer attributes
Name "${PRODUCT_NAME} ${APP_VERSION}"
OutFile "..\..\dist\rayforge-v${APP_VERSION}-installer.exe"
InstallDir "$PROGRAMFILES64\${PRODUCT_NAME}"
InstallDirRegKey HKLM "Software\${PRODUCT_NAME}" "Install_Dir"
Icon "..\..\${ICON_FILE}"
UninstallIcon "..\..\${ICON_FILE}"

;--------------------------------
; Pages

Page directory
Page instfiles
UninstPage uninstConfirm
UninstPage instfiles

;--------------------------------
; Running instance check
; The bundled executable name contains the version, so processes are
; matched on the product name instead of a fixed file name. The
; installer process itself shares that name prefix, so it is excluded
; from the match.

!macro CheckAppNotRunningBody LABEL
${LABEL}:
  ExecWait `"$SYSDIR\cmd.exe" /C tasklist /FI "IMAGENAME eq ${PRODUCT_NAME}*" /FO CSV /NH | "$SYSDIR\find.exe" /I "${PRODUCT_NAME}" | "$SYSDIR\find.exe" /V /I "installer.exe" > nul` $0
  ${If} $0 = 0
    MessageBox MB_RETRYCANCEL|MB_ICONEXCLAMATION "${PRODUCT_NAME} is still running.$\n$\nPlease close it and click Retry, or click Cancel to stop." /SD IDCANCEL IDRETRY ${LABEL}
    Abort
  ${EndIf}
!macroend

Function CheckAppNotRunning
  !insertmacro CheckAppNotRunningBody check_running
FunctionEnd

Function un.CheckAppNotRunning
  !insertmacro CheckAppNotRunningBody un_check_running
FunctionEnd

;--------------------------------
; File type registration
; .png, .bmp, .jpeg, .jpg, .svg, .svgz and .dxf are shared with other
; applications, so Rayforge never claims their default association:
; it only registers its ProgIDs and lists them under the extension's
; OpenWithProgids subkey, which adds Rayforge to the "Open with" menu.
; Uninstalling therefore only removes registrations Rayforge owns and
; never needs to restore anything. The only case where a shared
; extension's default value is touched is when it still points at a
; Rayforge ProgID, which would dangle after the ProgID is deleted.

!macro RegisterSharedFileType EXT PROGID DESCRIPTION
  WriteRegStr HKCR "${PROGID}" "" "${DESCRIPTION}"
  WriteRegStr HKCR "${PROGID}\DefaultIcon" "" "$INSTDIR\${EXECUTABLE_NAME},0"
  WriteRegStr HKCR "${PROGID}\shell\open\command" "" '"$INSTDIR\${EXECUTABLE_NAME}" "%1"'
  WriteRegStr HKCR ".${EXT}\OpenWithProgids" "${PROGID}" ""
!macroend

!macro UnregisterSharedFileType EXT PROGID
  DeleteRegValue HKCR ".${EXT}\OpenWithProgids" "${PROGID}"
  DeleteRegKey /ifempty HKCR ".${EXT}\OpenWithProgids"
  ReadRegStr $R0 HKCR ".${EXT}" ""
  ${If} $R0 == "${PROGID}"
    DeleteRegValue HKCR ".${EXT}" ""
  ${EndIf}
  DeleteRegKey /ifempty HKCR ".${EXT}"
  DeleteRegKey HKCR "${PROGID}"
!macroend

;--------------------------------
; Installer Section

Section "MainSection" SEC01
  ; A running instance keeps files locked, so make sure it is closed
  ; before anything on disk is touched.
  Call CheckAppNotRunning

  ; Move the previous installation aside instead of deleting it in
  ; place. A rename fails harmlessly while any file in it is still
  ; locked, and the intact copy stays available for restoration if
  ; writing the new files fails.
  StrCpy $PreviousInstallDir ""
  StrCpy $MovedInstallDir ""
  ReadRegStr $R0 HKLM "Software\${PRODUCT_NAME}" "Install_Dir"
  ${If} $R0 != ""
    ${If} ${FileExists} "$R0.old"
      RMDir /r "$R0.old"
    ${EndIf}
    ${If} ${FileExists} "$R0"
      Rename "$R0" "$R0.old"
      ${If} ${FileExists} "$R0"
        MessageBox MB_ICONSTOP "Could not move the previous installation at$\n$R0$\n$\nPlease make sure Rayforge is not running, then run the installer again."
        Abort
      ${EndIf}
      StrCpy $PreviousInstallDir "$R0"
      StrCpy $MovedInstallDir "$R0.old"
    ${EndIf}
  ${EndIf}

  SetOutPath "$INSTDIR"

  ; Copy all files from the PyInstaller output directory
  File /r "..\..\dist\${APP_DIR_NAME}\*.*"

  ; Store installation folder
  WriteRegStr HKLM "Software\${PRODUCT_NAME}" "Install_Dir" "$INSTDIR"

  ; Write the uninstaller
  WriteUninstaller "$INSTDIR\uninstall.exe"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}" "DisplayName" "${PRODUCT_NAME}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}" "UninstallString" '"$INSTDIR\uninstall.exe"'
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}" "DisplayIcon" "$INSTDIR\${EXECUTABLE_NAME}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}" "DisplayVersion" "${APP_VERSION}"
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}" "NoModify" 1
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}" "NoRepair" 1

  ; Register .ryp file type
  WriteRegStr HKCR ".ryp" "" "Rayforge.ProjectFile"
  WriteRegStr HKCR "Rayforge.ProjectFile" "" "Rayforge Project File"
  WriteRegStr HKCR "Rayforge.ProjectFile\DefaultIcon" "" "$INSTDIR\${EXECUTABLE_NAME},0"
  WriteRegStr HKCR "Rayforge.ProjectFile\shell\open\command" "" '"$INSTDIR\${EXECUTABLE_NAME}" "%1"'

  ; Register .rfs file type
  WriteRegStr HKCR ".rfs" "" "Rayforge.SketchFile"
  WriteRegStr HKCR "Rayforge.SketchFile" "" "Rayforge Sketch File"
  WriteRegStr HKCR "Rayforge.SketchFile\DefaultIcon" "" "$INSTDIR\${EXECUTABLE_NAME},0"
  WriteRegStr HKCR "Rayforge.SketchFile\shell\open\command" "" '"$INSTDIR\${EXECUTABLE_NAME}" "%1"'

  ; Register .rd file type (Ruida)
  WriteRegStr HKCR ".rd" "" "Rayforge.RuidaFile"
  WriteRegStr HKCR "Rayforge.RuidaFile" "" "Ruida Laser Cutter File"
  WriteRegStr HKCR "Rayforge.RuidaFile\DefaultIcon" "" "$INSTDIR\${EXECUTABLE_NAME},0"
  WriteRegStr HKCR "Rayforge.RuidaFile\shell\open\command" "" '"$INSTDIR\${EXECUTABLE_NAME}" "%1"'

  ; Register shared file types
  !insertmacro RegisterSharedFileType "png" "Rayforge.PngFile" "PNG Image"
  !insertmacro RegisterSharedFileType "bmp" "Rayforge.BmpFile" "BMP Image"
  !insertmacro RegisterSharedFileType "jpeg" "Rayforge.JpegFile" "JPEG Image"
  !insertmacro RegisterSharedFileType "jpg" "Rayforge.JpgFile" "JPEG Image"
  !insertmacro RegisterSharedFileType "svg" "Rayforge.SvgFile" "SVG Image"
  !insertmacro RegisterSharedFileType "svgz" "Rayforge.SvgzFile" "Compressed SVG Image"
  !insertmacro RegisterSharedFileType "dxf" "Rayforge.DxfFile" "DXF Drawing"

  ; Create Start Menu shortcuts
  CreateDirectory "$SMPROGRAMS\${PRODUCT_NAME}"
  CreateShortCut "$SMPROGRAMS\${PRODUCT_NAME}\${PRODUCT_NAME}.lnk" "$INSTDIR\${EXECUTABLE_NAME}"
  CreateShortCut "$SMPROGRAMS\${PRODUCT_NAME}\Uninstall ${PRODUCT_NAME}.lnk" "$INSTDIR\uninstall.exe"

  ; The new installation is complete, so the moved-aside copy of the
  ; previous version is no longer needed.
  ${If} $MovedInstallDir != ""
    RMDir /r "$MovedInstallDir"
  ${EndIf}
SectionEnd

;--------------------------------
; Uninstaller Section

Section "Uninstall"
  Call un.CheckAppNotRunning

  ; Remove registry keys
  DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}"
  DeleteRegKey HKLM "Software\${PRODUCT_NAME}"

  ; Unregister .ryp file type
  DeleteRegKey HKCR ".ryp"
  DeleteRegKey HKCR "Rayforge.ProjectFile"

  ; Unregister .rfs file type
  DeleteRegKey HKCR ".rfs"
  DeleteRegKey HKCR "Rayforge.SketchFile"

  ; Unregister .rd file type (Ruida)
  DeleteRegKey HKCR ".rd"
  DeleteRegKey HKCR "Rayforge.RuidaFile"

  ; Unregister shared file types without touching their associations
  !insertmacro UnregisterSharedFileType "png" "Rayforge.PngFile"
  !insertmacro UnregisterSharedFileType "bmp" "Rayforge.BmpFile"
  !insertmacro UnregisterSharedFileType "jpeg" "Rayforge.JpegFile"
  !insertmacro UnregisterSharedFileType "jpg" "Rayforge.JpgFile"
  !insertmacro UnregisterSharedFileType "svg" "Rayforge.SvgFile"
  !insertmacro UnregisterSharedFileType "svgz" "Rayforge.SvgzFile"
  !insertmacro UnregisterSharedFileType "dxf" "Rayforge.DxfFile"

  ; Remove the entire installation directory
  ; We delete the uninstaller first, then recursively remove its parent directory.
  Delete "$INSTDIR\uninstall.exe"
  RMDir /r "$INSTDIR"

  ; Remove shortcuts
  Delete "$SMPROGRAMS\${PRODUCT_NAME}\*.*"
  RMDir /r "$SMPROGRAMS\${PRODUCT_NAME}"
SectionEnd

;--------------------------------
; Put the previous installation back in place when an upgrade fails,
; so the user is not left with a partially deleted installation.

Function .onInstFailed
  ${If} $MovedInstallDir != ""
    RMDir /r "$INSTDIR"
    Rename "$MovedInstallDir" "$PreviousInstallDir"
  ${EndIf}
FunctionEnd
