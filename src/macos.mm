#include "platform.h"
#include <AVFoundation/AVFoundation.h>
#include <Cocoa/Cocoa.h>
#include <fcitx-utils/log.h>

namespace fcitx {
bool checkPermission() {
    // If program is updated, permission is reset to undetermined.
    // This only returns permission when the process starts, so a call after
    // user grants/rejects still returns undermined. To mitigate this, we
    // terminate (restarted by OS) the process if permission is undetermined.
    auto permission =
        [AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeAudio];
    FCITX_INFO() << "Audio permission: " << permission;
    switch (permission) {
    case AVAuthorizationStatusNotDetermined:
        dispatch_async(dispatch_get_main_queue(), ^{
          [[NSApplication sharedApplication] terminate:nil];
        });
        return false;
    case AVAuthorizationStatusAuthorized:
        return true;
    default:
        return false;
    }
}
} // namespace fcitx
