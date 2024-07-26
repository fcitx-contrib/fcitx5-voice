if (APPLE)
    set(PLATFORM_SRC macos.mm)
    find_library(AVFoundation_LIB AVFoundation REQUIRED)
    find_library(Cocoa_LIB Cocoa REQUIRED)
    set(PLATFORM_LIB ${AVFoundation_LIB} ${Cocoa_LIB} objc)
endif()

add_library(voice MODULE
    voice.cpp
    ${PLATFORM_SRC}
    "${PROJECT_SOURCE_DIR}/whisper.cpp/examples/common-sdl.cpp"
    "${PROJECT_SOURCE_DIR}/whisper.cpp/examples/common.cpp")
target_include_directories(voice PRIVATE
    "${PROJECT_SOURCE_DIR}/whisper.cpp/examples")
target_link_libraries(voice Fcitx5::Core whisper PkgConfig::sdl2 ${PLATFORM_LIB})
install(TARGETS voice DESTINATION "${FCITX_INSTALL_LIBDIR}/fcitx5")

fcitx5_translate_desktop_file(voice.conf.in voice.conf)
configure_file(voice-addon.conf.in.in voice-addon.conf.in)
fcitx5_translate_desktop_file("${CMAKE_CURRENT_BINARY_DIR}/voice-addon.conf.in" voice-addon.conf)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/voice.conf" DESTINATION "${FCITX_INSTALL_PKGDATADIR}/inputmethod" COMPONENT config)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/voice-addon.conf" RENAME voice.conf DESTINATION "${FCITX_INSTALL_PKGDATADIR}/addon" COMPONENT config)
