; NSIS Script for Rayforge Installer

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
; Installer Section

Section "MainSection" SEC01
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

  ; Register .png file type
  WriteRegStr HKCR ".png" "" "Rayforge.PngFile"
  WriteRegStr HKCR "Rayforge.PngFile" "" "PNG Image"
  WriteRegStr HKCR "Rayforge.PngFile\DefaultIcon" "" "$INSTDIR\${EXECUTABLE_NAME},0"
  WriteRegStr HKCR "Rayforge.PngFile\shell\open\command" "" '"$INSTDIR\${EXECUTABLE_NAME}" "%1"'

  ; Register .bmp file type
  WriteRegStr HKCR ".bmp" "" "Rayforge.BmpFile"
  WriteRegStr HKCR "Rayforge.BmpFile" "" "BMP Image"
  WriteRegStr HKCR "Rayforge.BmpFile\DefaultIcon" "" "$INSTDIR\${EXECUTABLE_NAME},0"
  WriteRegStr HKCR "Rayforge.BmpFile\shell\open\command" "" '"$INSTDIR\${EXECUTABLE_NAME}" "%1"'

  ; Register .jpeg file type
  WriteRegStr HKCR ".jpeg" "" "Rayforge.JpegFile"
  WriteRegStr HKCR "Rayforge.JpegFile" "" "JPEG Image"
  WriteRegStr HKCR "Rayforge.JpegFile\DefaultIcon" "" "$INSTDIR\${EXECUTABLE_NAME},0"
  WriteRegStr HKCR "Rayforge.JpegFile\shell\open\command" "" '"$INSTDIR\${EXECUTABLE_NAME}" "%1"'

  ; Register .jpg file type
  WriteRegStr HKCR ".jpg" "" "Rayforge.JpgFile"
  WriteRegStr HKCR "Rayforge.JpgFile" "" "JPEG Image"
  WriteRegStr HKCR "Rayforge.JpgFile\DefaultIcon" "" "$INSTDIR\${EXECUTABLE_NAME},0"
  WriteRegStr HKCR "Rayforge.JpgFile\shell\open\command" "" '"$INSTDIR\${EXECUTABLE_NAME}" "%1"'

  ; Register .svg file type
  WriteRegStr HKCR ".svg" "" "Rayforge.SvgFile"
  WriteRegStr HKCR "Rayforge.SvgFile" "" "SVG Image"
  WriteRegStr HKCR "Rayforge.SvgFile\DefaultIcon" "" "$INSTDIR\${EXECUTABLE_NAME},0"
  WriteRegStr HKCR "Rayforge.SvgFile\shell\open\command" "" '"$INSTDIR\${EXECUTABLE_NAME}" "%1"'

  ; Register .dxf file type
  WriteRegStr HKCR ".dxf" "" "Rayforge.DxfFile"
  WriteRegStr HKCR "Rayforge.DxfFile" "" "DXF Drawing"
  WriteRegStr HKCR "Rayforge.DxfFile\DefaultIcon" "" "$INSTDIR\${EXECUTABLE_NAME},0"
  WriteRegStr HKCR "Rayforge.DxfFile\shell\open\command" "" '"$INSTDIR\${EXECUTABLE_NAME}" "%1"'
  
  ; Create Start Menu shortcuts
  CreateDirectory "$SMPROGRAMS\${PRODUCT_NAME}"
  CreateShortCut "$SMPROGRAMS\${PRODUCT_NAME}\${PRODUCT_NAME}.lnk" "$INSTDIR\${EXECUTABLE_NAME}"
  CreateShortCut "$SMPROGRAMS\${PRODUCT_NAME}\Uninstall ${PRODUCT_NAME}.lnk" "$INSTDIR\uninstall.exe"
SectionEnd

;--------------------------------
; Uninstaller Section

Section "Uninstall"
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

  ; Unregister .png file type
  DeleteRegKey HKCR ".png"
  DeleteRegKey HKCR "Rayforge.PngFile"

  ; Unregister .bmp file type
  DeleteRegKey HKCR ".bmp"
  DeleteRegKey HKCR "Rayforge.BmpFile"

  ; Unregister .jpeg file type
  DeleteRegKey HKCR ".jpeg"
  DeleteRegKey HKCR "Rayforge.JpegFile"

  ; Unregister .jpg file type
  DeleteRegKey HKCR ".jpg"
  DeleteRegKey HKCR "Rayforge.JpgFile"

  ; Unregister .svg file type
  DeleteRegKey HKCR ".svg"
  DeleteRegKey HKCR "Rayforge.SvgFile"

  ; Unregister .dxf file type
  DeleteRegKey HKCR ".dxf"
  DeleteRegKey HKCR "Rayforge.DxfFile"

  ; Remove the entire installation directory
  ; We delete the uninstaller first, then recursively remove its parent directory.
  Delete "$INSTDIR\uninstall.exe"
  RMDir /r "$INSTDIR"

  ; Remove shortcuts
  Delete "$SMPROGRAMS\${PRODUCT_NAME}\*.*"
  RMDir /r "$SMPROGRAMS\${PRODUCT_NAME}"
SectionEnd
