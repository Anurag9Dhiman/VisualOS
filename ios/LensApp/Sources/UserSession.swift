import Foundation

// Stable device identity and first-launch state.
// userID is generated once on first access and persisted forever in UserDefaults.
// It is never tied to any login — it is purely a local correlation key for
// the memory/personalization layer.
enum UserSession {
    static var userID: String {
        if let stored = UserDefaults.standard.string(forKey: "lens_user_id") {
            return stored
        }
        let id = UUID().uuidString
        UserDefaults.standard.set(id, forKey: "lens_user_id")
        return id
    }

    static var isOnboarded: Bool {
        get { UserDefaults.standard.bool(forKey: "lens_onboarded") }
        set { UserDefaults.standard.set(newValue, forKey: "lens_onboarded") }
    }

    // "en_US" → "en-US" so it matches the server's expected locale format.
    static var locale: String {
        Locale.current.identifier.replacingOccurrences(of: "_", with: "-")
    }
}
