if (APPLE)
    set(PLATFORM_SRC macos.mm)
    find_library(AudioToolbox_LIB AudioToolbox REQUIRED)
    find_library(GameController_LIB GameController REQUIRED)
    find_library(ForceFeedback_LIB ForceFeedback REQUIRED)
    find_library(CoreAudio_LIB CoreAudio REQUIRED)
    find_library(CoreVideo_LIB CoreVideo REQUIRED)
    find_library(Carbon_LIB Carbon REQUIRED)
    find_library(IOKit_LIB IOKit REQUIRED)
    find_library(CoreHaptics_LIB CoreHaptics REQUIRED)
    find_library(AVFoundation_LIB AVFoundation REQUIRED)
    find_library(Cocoa_LIB Cocoa REQUIRED)
    set(PLATFORM_LIB
        "${sdl2_LIBDIR}/libSDL2.a"
        ${AudioToolbox_LIB}
        ${GameController_LIB}
        ${ForceFeedback_LIB}
        ${CoreAudio_LIB}
        ${CoreVideo_LIB}
        ${Carbon_LIB}
        ${IOKit_LIB}
        ${CoreHaptics_LIB}
        ${AVFoundation_LIB}
        ${CoreHaptics_LIB}
        ${Cocoa_LIB})
endif()

add_library(voice MODULE
    voice.cpp
    ${PLATFORM_SRC}
    "${PROJECT_SOURCE_DIR}/whisper.cpp/examples/common-sdl.cpp"
    "${PROJECT_SOURCE_DIR}/whisper.cpp/examples/common.cpp")
target_include_directories(voice PRIVATE
    "${sdl2_INCLUDE_DIRS}"
    "${PROJECT_SOURCE_DIR}/whisper.cpp/examples")
target_link_libraries(voice Fcitx5::Core whisper ${PLATFORM_LIB})
install(TARGETS voice DESTINATION "${FCITX_INSTALL_LIBDIR}/fcitx5")

fcitx5_translate_desktop_file(voice.conf.in voice.conf)
configure_file(voice-addon.conf.in.in voice-addon.conf.in)
fcitx5_translate_desktop_file("${CMAKE_CURRENT_BINARY_DIR}/voice-addon.conf.in" voice-addon.conf)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/voice.conf" DESTINATION "${FCITX_INSTALL_PKGDATADIR}/inputmethod" COMPONENT config)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/voice-addon.conf" RENAME voice.conf DESTINATION "${FCITX_INSTALL_PKGDATADIR}/addon" COMPONENT config)
