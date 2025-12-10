# watchOS API Compatibility Reference

Quick reference for iOS/macOS framework availability on watchOS. Always verify with current Apple documentation.

## Fully Available

| Framework | watchOS Version | Notes |
|-----------|-----------------|-------|
| SwiftUI | 6.0+ | Full support, preferred UI framework |
| SwiftData | 10.0+ | Full support |
| Combine | 6.0+ | Full support |
| Swift Concurrency | 8.0+ | async/await, actors |
| HealthKit | 2.0+ | Core feature of watchOS |
| WorkoutKit | 9.0+ | Workout session management |
| WidgetKit | 9.0+ | Complications and Smart Stack |
| WatchConnectivity | 2.0+ | iPhone ↔ Watch communication |
| CloudKit | 3.0+ | Full support |
| CoreData | 2.0+ | Full support |
| CoreMotion | 2.0+ | Accelerometer, gyroscope |
| CoreBluetooth | 4.0+ | BLE peripherals |
| AVFoundation | 3.0+ | Audio only (no video) |
| UserNotifications | 3.0+ | Local and remote notifications |
| CoreLocation | 2.0+ | Limited—see notes |
| EventKit | 5.0+ | Calendar/reminder access |
| Contacts | 6.0+ | Read-only contact access |
| CryptoKit | 6.0+ | Full support |
| Network | 6.0+ | URLSession, NWConnection |
| OSLog | 8.0+ | Unified logging |
| StoreKit | 6.1+ | Limited—in-app purchases |

## Partially Available / Limited

| Framework | Limitation | Workaround |
|-----------|-----------|------------|
| CoreLocation | No continuous GPS, battery intensive | Use significant change monitoring, workout sessions for GPS |
| MapKit | No interactive maps (watchOS <10), limited API | Display static maps, link to iPhone app for full maps |
| AVFoundation | Audio only, no video playback | Stream audio, use iPhone for video |
| StoreKit | Limited UI, no subscription management UI | Handle purchases on iPhone, sync entitlements |
| CoreImage | Very limited filters | Pre-process images on iPhone |
| SpriteKit | Available but no 3D audio, limited | Use for simple 2D animations |
| SceneKit | Available but constrained | Use for simple 3D, no complex shaders |
| Network | Background URLSession limited | Use Watch Connectivity for critical data |

## NOT Available on watchOS

| Framework | Alternative |
|-----------|-------------|
| UIKit | SwiftUI (full replacement) |
| AppKit | SwiftUI |
| WebKit / WKWebView | Link to iPhone Safari, display text content natively |
| ARKit | Not applicable to Watch form factor |
| RealityKit | Not applicable |
| VisionKit | Use iPhone for document scanning |
| PDFKit | Process PDFs on iPhone, send results to Watch |
| GameKit | Limited or unavailable |
| Metal | Very limited, prefer CPU-based graphics |
| CoreML | Available but model size severely limited |
| Vision | Very limited, process on iPhone |
| NaturalLanguage | Limited, prefer iPhone processing |
| QuickLook | Not available |
| MessageUI | Use Watch Connectivity to trigger on iPhone |

## WidgetKit Complication Families

| Family | Description | Use Case |
|--------|-------------|----------|
| accessoryCircular | Small circle | App launchers, single metrics, icons |
| accessoryRectangular | Rectangle, multi-line | Smart Stack widget, detailed info, lists |
| accessoryInline | Single line text | Brief status, watch face text slots |
| accessoryCorner | Corner curved | watchOS only, gauge + label |

## watchOS Version Features

### watchOS 9
- WidgetKit for complications (ClockKit deprecated)
- CallKit support
- Workout API enhancements

### watchOS 10
- SwiftData support
- Smart Stack with widgets
- New navigation patterns (NavigationSplitView)
- Vertical TabView pagination
- New design language (full-screen apps)

### watchOS 11
- Live Activities from iPhone appear on Watch
- Double Tap gesture enhancements
- Interactive widgets in Smart Stack
- Relevance API improvements

### watchOS 26 (Beta)
- Liquid Glass design system
- New button styles (.glass)
- Enhanced accessibility APIs

## Common Migration Patterns

### UITableView → SwiftUI List
```swift
// iOS UIKit
tableView.register(...)
// watchOS SwiftUI
List(items) { item in ItemRow(item: item) }
```

### UIKit Navigation → NavigationStack
```swift
// iOS UIKit
navigationController?.pushViewController(...)
// watchOS SwiftUI
NavigationStack { NavigationLink(...) { ... } }
```

### UserDefaults Sharing → App Groups
```swift
// Shared UserDefaults
let defaults = UserDefaults(suiteName: "group.com.yourapp")
// Access from both iOS and watchOS targets
```

### Network Requests → Watch Connectivity
```swift
// For critical data, prefer Watch Connectivity over direct network
WCSession.default.transferUserInfo(data)
// Watch receives even if app not running
```