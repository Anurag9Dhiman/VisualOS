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
  LensApp.swift          @main entry point
  ContentView.swift      PhotosPicker → scan → streaming → card
  ScanViewModel.swift    ObservableObject driving the state machine
  Views/
    CardView.swift       NormalCardView + FallbackCardView + previews
```

## State machine

```
idle → (tap Scan) → scanning → streaming("The Eiffel…") → done(card)
                                                         ↘ failed("msg")
```

`ScanViewModel` owns the state. `ContentView` renders it. `CardView` is pure display — no business logic.

## Replacing PhotosPicker with the real camera

`ContentView` uses `PhotosPicker` to keep the skeleton runnable in Simulator. For production, replace the `photosPicker` modifier with an `AVCaptureSession` view that captures a frame on button tap and passes the JPEG bytes to `vm.scan(imageData:location:)`.
