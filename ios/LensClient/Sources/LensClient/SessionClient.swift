/// Client for scan session history — GET /sessions and GET /session/{id}.

import Foundation

public struct ScanSummary: Decodable, Sendable, Identifiable {
    public let sessionId: String
    public let entityName: String
    public let entityType: String
    public let confidenceLevel: String
    public let cardHeadline: String
    public let cardBody: String
    public let userId: String
    public let scannedAt: Date
    public let expiresAt: Date

    public var id: String { sessionId }

    private enum CodingKeys: String, CodingKey {
        case sessionId = "session_id"
        case entityName = "entity_name"
        case entityType = "entity_type"
        case confidenceLevel = "confidence_level"
        case cardHeadline = "card_headline"
        case cardBody = "card_body"
        case userId = "user_id"
        case scannedAt = "scanned_at"
        case expiresAt = "expires_at"
    }
}

public struct SessionClient: Sendable {
    public let baseURL: URL
    private let apiKey: String
    private let session: URLSession

    public init(baseURL: URL, apiKey: String, session: URLSession = .shared) {
        self.baseURL = baseURL
        self.apiKey = apiKey
        self.session = session
    }

    // MARK: - GET /sessions

    public func listSessions(userID: String, limit: Int = 20) async throws -> [ScanSummary] {
        var comps = URLComponents(url: baseURL.appendingPathComponent("sessions"),
                                  resolvingAgainstBaseURL: true)!
        comps.queryItems = [
            URLQueryItem(name: "user_id", value: userID),
            URLQueryItem(name: "limit", value: "\(limit)"),
        ]
        var request = URLRequest(url: comps.url!)
        request.setValue(apiKey, forHTTPHeaderField: "X-API-Key")
        request.timeoutInterval = 10

        let (data, response) = try await session.data(for: request)
        if let http = response as? HTTPURLResponse, http.statusCode != 200 {
            throw LensError.serverError(http.statusCode, "")
        }

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode([ScanSummary].self, from: data)
    }
}
