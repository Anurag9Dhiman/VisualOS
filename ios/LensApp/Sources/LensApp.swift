import SwiftUI

@main
struct LensApp: App {
    @State private var isOnboarded = UserSession.isOnboarded

    var body: some Scene {
        WindowGroup {
            if isOnboarded {
                ContentView()
            } else {
                OnboardingView(isOnboarded: $isOnboarded)
            }
        }
    }
}
