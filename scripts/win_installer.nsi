; NSIS Script for Rayforge Installer

;--------------------------------
; Defines
; These are variables passed from our build script using the -D flag
!define PRODUCT_NAME "Rayforge"
!ifndef APP_VERSION
  !define APP_VERSION "0.0.0"
!endif
!ifndef EXECUTABLE_NAME
  !define EXECUTABLE_NAME "rayforge.exe"
!endif
!ifndef ICON_FILE
  !define ICON_FILE "rayforge.ico"
!endif

;--------------------------------
; General

RequestExecutionLevel admin ; Request admin rights for installation
SetCompressor lzma ; Use modern, efficient compression

; Installer attributes
Name "${PRODUCT_NAME} ${APP_VERSION}"
OutFile "..\dist\${PRODUCT_NAME}-v${APP_VERSION}-installer.exe"
InstallDir "$PROGRAMFILES64\${PRODUCT_NAME}"
InstallDirRegKey HKLM "Software\${PRODUCT_NAME}" "Install_Dir"
Icon "..\${ICON_FILE}"
UninstallIcon "..\${ICON_FILE}"

;--------------------------------
; Pages

Page directory
Page instfiles
UninstPage uninstConfirm
UninstPage instfiles

;--------------------------------
; Installer Section

Section "MainSection" SEC01
  SetOutPath "$INSTDIR"
  
  File "..\dist\${EXECUTABLE_NAME}"
  
  File "..\${ICON_FILE}"
  
  ; Store installation folder
  WriteRegStr HKLM "Software\${PRODUCT_NAME}" "Install_Dir" "$INSTDIR"
  
  ; Write the uninstaller
  WriteUninstaller "$INSTDIR\uninstall.exe"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}" "DisplayName" "${PRODUCT_NAME}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}" "UninstallString" '"$INSTDIR\uninstall.exe"'
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}" "DisplayIcon" "$INSTDIR\${ICON_FILE}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}" "DisplayVersion" "${APP_VERSION}"
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}" "NoModify" 1
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}" "NoRepair" 1
  
  ; Create Start Menu shortcuts
  CreateDirectory "$SMPROGRAMS\${PRODUCT_NAME}"
  CreateShortCut "$SMPROGRAMS\${PRODUCT_NAME}\${PRODUCT_NAME}.lnk" "$INSTDIR\${EXECUTABLE_NAME}" "" "$INSTDIR\${ICON_FILE}" 0
  CreateShortCut "$SMPROGRAMS\${PRODUCT_NAME}\Uninstall ${PRODUCT_NAME}.lnk" "$INSTDIR\uninstall.exe"
SectionEnd

;--------------------------------
; Uninstaller Section

Section "Uninstall"
  ; Remove registry keys
  DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}"
  DeleteRegKey HKLM "Software\${PRODUCT_NAME}"

  ; Remove files and directories
  Delete "$INSTDIR\${EXECUTABLE_NAME}"
  Delete "$INSTDIR\${ICON_FILE}"
  Delete "$INSTDIR\uninstall.exe"
  RMDir "$INSTDIR"

  ; Remove shortcuts
  Delete "$SMPROGRAMS\${PRODUCT_NAME}\*.*"
  RMDir "$SMPROGRAMS\${PRODUCT_NAME}"
SectionEnd
