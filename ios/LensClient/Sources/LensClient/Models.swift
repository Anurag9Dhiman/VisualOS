/// Data contracts — mirrors src/contracts.py exactly.
/// If the Python schema changes, update this file to match.

import Foundation

// MARK: - Response card

public enum ResponseCard: Decodable, Sendable {
    case normal(NormalCard)
    case fallback(FallbackCard)

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type_ = try container.decode(String.self, forKey: .cardType)
        switch type_ {
        case "normal":
            self = .normal(try NormalCard(from: decoder))
        case "fallback":
            self = .fallback(try FallbackCard(from: decoder))
        default:
            throw DecodingError.dataCorruptedError(
                forKey: .cardType,
                in: container,
                debugDescription: "Unknown card_type: \(type_)"
            )
        }
    }

    private enum CodingKeys: String, CodingKey {
        case cardType = "card_type"
    }

    public var headline: String {
        switch self {
        case .normal(let c): return c.headline
        case .fallback(let c): return c.headline
        }
    }
}

// MARK: - Normal card

public struct NormalCard: Decodable, Sendable {
    public let headline: String
    public let body: String
    public let personalizedHooks: [PersonalizedHook]
    public let citations: [Citation]
    public let confidenceDisplayed: String
    public let sourceMix: SourceMix
    public let costUsdTotal: Double
    public let latencyMs: Int

    private enum CodingKeys: String, CodingKey {
        case headline, body, citations
        case personalizedHooks = "personalized_hooks"
        case confidenceDisplayed = "confidence_displayed"
        case sourceMix = "source_mix"
        case costUsdTotal = "cost_usd_total"
        case latencyMs = "latency_ms"
    }
}

public struct PersonalizedHook: Decodable, Sendable {
    public let fact: String
    public let citationTag: String

    private enum CodingKeys: String, CodingKey {
        case fact
        case citationTag = "citation_tag"
    }
}

public struct Citation: Decodable, Sendable {
    public let id: String
    public let sourceName: String
    public let url: String?
    public let asOf: String?

    private enum CodingKeys: String, CodingKey {
        case id, url
        case sourceName = "source_name"
        case asOf = "as_of"
    }
}

public struct SourceMix: Decodable, Sendable {
    public let usedVision: Bool
    public let usedMemory: Bool
    public let usedSearch: Bool

    private enum CodingKeys: String, CodingKey {
        case usedVision = "used_vision"
        case usedMemory = "used_memory"
        case usedSearch = "used_search"
    }
}

// MARK: - Fallback card

public struct FallbackCard: Decodable, Sendable {
    public let headline: String
    public let observation: String
    public let suggestion: String
}

// MARK: - Streaming events

public enum StreamEvent: Sendable {
    case token(String)
    case card(ResponseCard)
    case error(String)
}

// MARK: - Errors

public enum LensError: Error, Sendable, Equatable {
    case unauthorized
    case unsupportedMediaType
    case timeout
    case serverError(Int, String)
    case decodingError(Error)
    case networkError(Error)

    public static func == (lhs: LensError, rhs: LensError) -> Bool {
        switch (lhs, rhs) {
        case (.unauthorized, .unauthorized): return true
        case (.unsupportedMediaType, .unsupportedMediaType): return true
        case (.timeout, .timeout): return true
        case (.serverError(let lc, _), .serverError(let rc, _)): return lc == rc
        case (.decodingError, .decodingError): return true
        case (.networkError, .networkError): return true
        default: return false
        }
    }
}
