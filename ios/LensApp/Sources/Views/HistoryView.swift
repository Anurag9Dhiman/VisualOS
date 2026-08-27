import LensClient
import SwiftUI

struct HistoryView: View {
    @StateObject private var vm = HistoryViewModel()
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Group {
                switch vm.state {
                case .loading:
                    ProgressView("Loading history…")
                        .frame(maxWidth: .infinity, maxHeight: .infinity)

                case .empty:
                    ContentUnavailableView(
                        "No scans yet",
                        systemImage: "camera.viewfinder",
                        description: Text("Point your camera at a building or monument to get started.")
                    )

                case .loaded(let sessions):
                    List(sessions) { session in
                        SessionRow(session: session)
                    }
                    .listStyle(.plain)

                case .failed(let msg):
                    ContentUnavailableView(
                        "Couldn't load history",
                        systemImage: "exclamationmark.triangle",
                        description: Text(msg)
                    )
                }
            }
            .navigationTitle("Scan History")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
            .task { await vm.load() }
        }
    }
}

// MARK: - Row

private struct SessionRow: View {
    let session: ScanSummary

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(session.cardHeadline)
                .font(.headline)
                .lineLimit(1)

            Text(session.cardBody)
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .lineLimit(2)

            HStack(spacing: 10) {
                Label(session.entityType.capitalized, systemImage: iconName(for: session.entityType))
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Spacer()

                Text(session.scannedAt, style: .relative) + Text(" ago")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
            .font(.caption)
        }
        .padding(.vertical, 4)
    }

    private func iconName(for type_: String) -> String {
        switch type_ {
        case "building":  return "building.2"
        case "monument":  return "building.columns"
        case "statue":    return "figure.stand"
        default:          return "cube"
        }
    }
}

// MARK: - ViewModel

@MainActor
final class HistoryViewModel: ObservableObject {
    enum State {
        case loading
        case empty
        case loaded([ScanSummary])
        case failed(String)
    }

    @Published var state: State = .loading

    private let client: SessionClient

    init() {
        let base = ProcessInfo.processInfo.environment["LENS_API_URL"] ?? "http://localhost:8000"
        let key  = ProcessInfo.processInfo.environment["LENS_API_KEY"] ?? ""
        client = SessionClient(baseURL: URL(string: base)!, apiKey: key)
    }

    func load() async {
        state = .loading
        do {
            let sessions = try await client.listSessions(userID: UserSession.userID)
            state = sessions.isEmpty ? .empty : .loaded(sessions)
        } catch {
            state = .failed(error.localizedDescription)
        }
    }
}

#Preview {
    HistoryView()
}
