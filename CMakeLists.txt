cmake_minimum_required(VERSION 3.6.0)

project(fcitx5-voice VERSION 0.1.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(ECM 1.0.0 REQUIRED)
set(CMAKE_MODULE_PATH ${ECM_MODULE_PATH} ${CMAKE_MODULE_PATH})
include(FeatureSummary)
include(GNUInstallDirs)
include(ECMUninstallTarget)

find_package(Gettext REQUIRED)
find_package(Fcitx5Core 5.0.0 REQUIRED)
find_package(PkgConfig REQUIRED)

pkg_check_modules(sdl2 REQUIRED IMPORTED_TARGET sdl2)

add_definitions(-DFCITX_GETTEXT_DOMAIN=\"fcitx5-voice\")
fcitx5_add_i18n_definition()

include("${FCITX_INSTALL_CMAKECONFIG_DIR}/Fcitx5Utils/Fcitx5CompilerSettings.cmake")

add_subdirectory(whisper.cpp)
add_subdirectory(src)

feature_summary(WHAT ALL FATAL_ON_MISSING_REQUIRED_PACKAGES)
