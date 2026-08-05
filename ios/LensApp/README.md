# LensApp — iOS SwiftUI skeleton

Drop these files into a new Xcode project and the full scan → stream → card flow works immediately.

## Xcode setup (5 minutes)

1. **New project** — Xcode → File → New → Project → iOS → App  
   - Product Name: `LensApp`  
   - Interface: SwiftUI  
   - Language: Swift  
   - Minimum Deployment: iOS 17

2. **Add LensClient package**  
   File → Add Package Dependencies → Add Local → select `ios/LensClient/`

3. **Copy source files** — drag everything from `ios/LensApp/Sources/` into the project  
   (replace the generated `ContentView.swift` and `LensApp.swift`)

4. **Info.plist keys** — add these two entries:
   ```
   NSCameraUsageDescription     → "Lens OS needs the camera to identify what you're looking at."
   NSLocationWhenInUseUsageDescription → "Lens OS uses your location to improve identification."
   ```

5. **Env vars** (Scheme → Run → Arguments → Environment Variables):
   ```
   LENS_API_URL   https://lens-os-api.fly.dev   (or http://localhost:8000 for local)
   LENS_API_KEY   lens-...
   ```

6. **Run** on device or simulator — tap Scan, pick a photo, watch the card stream in.

## File map

```
Sources/
  LensApp.swift             @main entry point
  ContentView.swift         Full-screen camera → scan → streaming → result sheet
  ScanViewModel.swift       ObservableObject driving the state machine
  Views/
    CameraView.swift        AVCaptureSession viewfinder + shutter button
    CardView.swift          NormalCardView + FallbackCardView + previews
```

## State machine

```
idle → (tap shutter) → scanning → streaming("The Eiffel…") → done(card)
                                                             ↘ failed("msg")
```

`ScanViewModel` owns the state. `ContentView` renders it. `CameraView` is a `UIViewControllerRepresentable` that runs a live `AVCaptureSession` — it stays running in the background while the result sheet is open so there's no black flash when the user dismisses it. `CardView` is pure display — no business logic.

## Simulator note

`AVCaptureSession` requires a physical device with a camera. In Simulator, the app will launch but the camera preview will be blank and the shutter button will not trigger a capture. To test the UI in Simulator, temporarily replace `CameraView` with a `Button("Simulate scan") { ... }` that loads a bundled test JPEG and calls `vm.scan(imageData:location:)` directly.
