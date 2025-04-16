
from __future__ import annotations
import objc
import Foundation

IOKit = Foundation.NSBundle.bundleWithIdentifier_("com.apple.framework.IOKit")

functions = [
    ("IOServiceGetMatchingService", b"II@"),
    ("IOServiceMatching", b"@*"),
    ("IORegistryEntryCreateCFProperties", b"IIo^@@I"),
]

objc.loadBundleFunctions(IOKit, globals(), functions)

def accelerator_performance_statistics() -> dict[str, int]:
    import CoreFoundation
    accelerator_info = IOServiceGetMatchingService(
        0, IOServiceMatching(b"IOAccelerator")
    )
    if accelerator_info == 0:
        CoreFoundation.CFRelease(accelerator_info)
        raise RuntimeError("IOAccelerator not found")

    err, props = IORegistryEntryCreateCFProperties(accelerator_info, None, None, 0)
    if err != 0:
        CoreFoundation.CFRelease(err)
        raise RuntimeError("IOAccelerator properties not found")
    # else...
    data = dict(props["PerformanceStatistics"])
    CoreFoundation.CFRelease(accelerator_info)
    CoreFoundation.CFRelease(props)
    return data
