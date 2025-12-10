# watchOS Implementation Plan Template

Use this template when generating implementation plans. Customize sections based on project analysis.

---

# watchOS Implementation Plan: [App Name]

**Generated**: [Date]
**Source Project**: [iOS/macOS app name and path]
**Target watchOS Version**: [X.0]+

## Executive Summary

| Aspect | Details |
|--------|---------|
| App Type | [Companion / Standalone / Hybrid] |
| Primary Features | [2-4 key features for Watch] |
| Estimated Phases | [N] |
| API Compatibility Issues | [N] items requiring attention |
| Watch Connectivity | [Required / Optional / Not needed] |
| Complications | [Yes - types / No] |

### Recommended Approach
[1-2 sentences on the recommended architecture and approach based on the source project analysis]

---

## ⚠️ API Compatibility Warnings

The following APIs from your iOS app are unavailable or limited on watchOS:

| iOS API/Framework | watchOS Status | Impact | Alternative/Workaround |
|-------------------|----------------|--------|------------------------|
| [API Name] | ❌ Unavailable | [What breaks] | [Solution] |
| [API Name] | ⚠️ Limited | [Limitation] | [Adaptation needed] |

### Critical Blockers
[List any features that cannot be implemented on watchOS and why]

### Recommended Exclusions
[Features that could technically work but shouldn't be on watchOS for UX reasons]

---

## Phase 1: Project Setup & Shared Infrastructure
**Estimated effort**: [X hours/days]

### 1.1 Add watchOS Target
- [ ] File → New → Target → watchOS App
- [ ] Configure deployment target: watchOS [X.0]
- [ ] Select "Watch App" (single target, not legacy dual-target)
- [ ] Bundle identifier: `[com.yourapp].watchkitapp`

### 1.2 Configure App Groups
- [ ] Create App Group: `group.[com.yourapp]`
- [ ] Enable on iOS target
- [ ] Enable on watchOS target
- [ ] Update UserDefaults access to use shared suite

### 1.3 Share Code via Target Membership
Models to share:
- [ ] `[Model1].swift`
- [ ] `[Model2].swift`

Services to share:
- [ ] `[Service1].swift` (if platform-agnostic)

### 1.4 Dependencies
SPM packages to add to Watch target:
- [ ] [Package name] - [reason]

Packages NOT compatible with watchOS:
- [ ] [Package name] - [alternative]

---

## Phase 2: Core Watch App Structure
**Estimated effort**: [X hours/days]

### 2.1 App Entry Point
```swift
@main
struct [AppName]WatchApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
```

### 2.2 Navigation Structure
[Describe chosen navigation pattern]

- [ ] Implement `ContentView` with primary navigation
- [ ] Choose navigation style:
  - [ ] TabView with vertical pagination
  - [ ] NavigationStack for hierarchical
  - [ ] Single-screen focused UI

### 2.3 Feature Views

#### [Feature 1]: [Name]
Purpose: [Brief description]
Priority: [High/Medium/Low]
- [ ] Create `[Feature1]View.swift`
- [ ] Create `[Feature1]ViewModel.swift` (if needed)
- [ ] Adapt UI for small screen
- [ ] Add haptic feedback for actions

#### [Feature 2]: [Name]
[Repeat structure]

### 2.4 Watch-Specific Adaptations
- [ ] Reduce information density
- [ ] Implement glanceable layouts
- [ ] Use SF Symbols appropriately sized
- [ ] Support Dynamic Type

---

## Phase 3: Complications & Smart Stack Widgets
**Estimated effort**: [X hours/days]

### 3.1 Widget Extension Setup
- [ ] Add Widget Extension target
- [ ] Configure for watchOS
- [ ] Set up shared data access (App Groups)

### 3.2 Complication Families

#### accessoryCircular
Use case: [e.g., app launcher, single metric]
- [ ] Create `CircularComplicationView`
- [ ] Use `AccessoryWidgetBackground()` for consistent styling
- [ ] Implement `widgetLabel` for faces that support it

#### accessoryRectangular
Use case: [e.g., Smart Stack, detailed status]
- [ ] Create `RectangularComplicationView`
- [ ] Design for multiple lines of glanceable info
- [ ] Consider relevance scoring for Smart Stack positioning

#### accessoryInline (if applicable)
Use case: [e.g., brief status text]
- [ ] Create `InlineComplicationView`
- [ ] Keep text concise (system truncates)

### 3.3 Timeline Provider
- [ ] Implement `TimelineProvider` or `AppIntentTimelineProvider`
- [ ] Define `TimelineEntry` struct
- [ ] Implement `getTimeline()` with appropriate refresh policy
- [ ] Handle `placeholder()` and `getSnapshot()`

### 3.4 Relevance for Smart Stack
- [ ] Implement relevance scoring based on [context]
- [ ] Use `RelevantIntentManager` for time-based relevance
- [ ] Update relevance when app state changes

---

## Phase 4: Data Synchronization
**Estimated effort**: [X hours/days]

### 4.1 Watch Connectivity Setup
- [ ] Create `WatchSessionManager.swift`
- [ ] Implement `WCSessionDelegate`
- [ ] Activate session in app lifecycle

### 4.2 Data Transfer Strategy

| Data Type | Transfer Method | Direction |
|-----------|-----------------|-----------|
| [User preferences] | `updateApplicationContext` | iOS → Watch |
| [Real-time updates] | `sendMessage` | Bidirectional |
| [Queued data] | `transferUserInfo` | Bidirectional |
| [Complication data] | `transferCurrentComplicationUserInfo` | iOS → Watch |
| [Large files] | `transferFile` | Bidirectional |

### 4.3 Shared Data Storage
- [ ] Define shared UserDefaults keys
- [ ] Implement data serialization for transfer
- [ ] Handle offline scenarios gracefully

### 4.4 Background Handling
- [ ] Handle `session(_:didReceiveUserInfo:)` for queued data
- [ ] Implement complication update on data receive
- [ ] Test with app in background/terminated states

---

## Phase 5: Polish & Platform Features
**Estimated effort**: [X hours/days]

### 5.1 Haptic Feedback
- [ ] Add `.success` haptic for completed actions
- [ ] Add `.notification` for important alerts
- [ ] Add `.click` for selections
- [ ] Use `WKInterfaceDevice.current().play(_:)`

### 5.2 Always On Display Support
- [ ] Check `@Environment(\.isLuminanceReduced)`
- [ ] Reduce animations when dimmed
- [ ] Simplify UI for AOD state
- [ ] Reduce color saturation for OLED

### 5.3 Double Tap (watchOS 11+)
- [ ] Identify primary action for Double Tap
- [ ] Implement using `defaultAction` modifier
- [ ] Test gesture recognition

### 5.4 Accessibility
- [ ] Support Dynamic Type (all text sizes)
- [ ] Add accessibility labels
- [ ] Test with VoiceOver
- [ ] Ensure sufficient color contrast

### 5.5 Performance Optimization
- [ ] Use `TimelineSchedule` for metric views
- [ ] Batch UI updates
- [ ] Profile with Instruments
- [ ] Test on oldest supported Watch model

---

## Testing Checklist

### Simulator Testing
- [ ] 40mm Watch (smallest screen)
- [ ] 45mm Watch (largest screen)
- [ ] All supported watchOS versions

### Real Device Testing
- [ ] Fresh install
- [ ] Upgrade from previous version (if applicable)
- [ ] With iPhone paired
- [ ] Without iPhone (if standalone features exist)

### Complication Testing
- [ ] Verify on multiple watch faces
- [ ] Test timeline refresh
- [ ] Test Smart Stack positioning
- [ ] Verify tinted appearance on faces with tint

### Watch Connectivity Testing
- [ ] Initial sync on install
- [ ] Background updates received
- [ ] Large data transfer
- [ ] Airplane mode recovery
- [ ] iPhone app not running

### Edge Cases
- [ ] Low battery behavior
- [ ] Low storage handling
- [ ] Network unavailable
- [ ] Watch Connectivity unavailable

---

## File Structure

```
[ProjectName]/
├── [ProjectName]Watch/
│   ├── [ProjectName]WatchApp.swift
│   ├── ContentView.swift
│   ├── Views/
│   │   ├── [Feature1]View.swift
│   │   └── [Feature2]View.swift
│   ├── ViewModels/
│   │   └── [ViewModel].swift
│   └── WatchConnectivity/
│       └── WatchSessionManager.swift
├── [ProjectName]WatchWidget/
│   ├── [ProjectName]Widget.swift
│   ├── ComplicationViews/
│   │   ├── CircularView.swift
│   │   └── RectangularView.swift
│   └── TimelineProvider.swift
└── Shared/
    ├── Models/
    └── Services/
```

---

## Next Steps

1. Review this plan and confirm feature priorities
2. Address any questions about API compatibility warnings
3. Begin Phase 1 implementation
4. Schedule checkpoint reviews after each phase

---

*Plan generated by create-watchos-version skill*
