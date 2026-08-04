import LensClient
import SwiftUI

// MARK: - Router

public struct CardView: View {
    public let card: ResponseCard

    public var body: some View {
        switch card {
        case .normal(let c):  NormalCardView(card: c)
        case .fallback(let c): FallbackCardView(card: c)
        }
    }
}

// MARK: - Normal card

struct NormalCardView: View {
    let card: NormalCard

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            // Headline
            Text(card.headline)
                .font(.title2.bold())

            Divider()

            // Body
            Text(card.body)
                .font(.body)
                .foregroundStyle(.primary)

            // Personalised hooks
            if !card.personalizedHooks.isEmpty {
                VStack(alignment: .leading, spacing: 10) {
                    Label("Also interesting", systemImage: "sparkles")
                        .font(.subheadline.bold())
                        .foregroundStyle(.secondary)
                    ForEach(card.personalizedHooks, id: \.fact) { hook in
                        HookRow(hook: hook)
                    }
                }
                .padding()
                .background(.quaternary, in: RoundedRectangle(cornerRadius: 12))
            }

            // Meta row
            HStack(spacing: 16) {
                Label("\(card.latencyMs) ms", systemImage: "bolt")
                Label(String(format: "$%.4f", card.costUsdTotal), systemImage: "dollarsign.circle")
                if card.confidenceDisplayed == "hedged" {
                    Label("uncertain", systemImage: "questionmark.circle")
                        .foregroundStyle(.orange)
                }
            }
            .font(.caption)
            .foregroundStyle(.secondary)

            // Citations
            if !card.citations.isEmpty {
                CitationsView(citations: card.citations)
            }
        }
    }
}

private struct HookRow: View {
    let hook: PersonalizedHook

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: "arrow.right.circle.fill")
                .foregroundStyle(.tint)
                .font(.body)
            Text(hook.fact)
                .font(.subheadline)
        }
    }
}

private struct CitationsView: View {
    let citations: [Citation]

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Sources")
                .font(.caption.bold())
                .foregroundStyle(.secondary)
            ForEach(citations, id: \.id) { c in
                if let urlStr = c.url, let url = URL(string: urlStr) {
                    Link(c.sourceName, destination: url)
                        .font(.caption)
                } else {
                    Text(c.sourceName)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }
}

// MARK: - Fallback card

struct FallbackCardView: View {
    let card: FallbackCard

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Label(card.headline, systemImage: "questionmark.circle")
                .font(.title3.bold())

            Text(card.observation)
                .foregroundStyle(.secondary)

            Text(card.suggestion)
                .font(.subheadline)
                .padding()
                .background(.orange.opacity(0.12), in: RoundedRectangle(cornerRadius: 10))
        }
    }
}

// MARK: - Previews

#Preview("Normal card") {
    ScrollView {
        CardView(card: .normal(NormalCard(
            headline: "Lalbagh Botanical Garden — West Gate",
            body: "Founded in 1760 by Hyder Ali and expanded by Tipu Sultan, Lalbagh spans 240 acres in central Bengaluru. Its iconic 1864 glasshouse modelled on London's Crystal Palace hosts the Republic Day and Independence Day flower shows.",
            personalizedHooks: [
                PersonalizedHook(fact: "The garden houses a 3,000-million-year-old rock formation — one of the oldest exposed rocks on Earth.", citationTag: "wiki"),
            ],
            citations: [
                Citation(id: "wiki", sourceName: "Wikipedia", url: "https://en.wikipedia.org/wiki/Lalbagh", asOf: nil),
            ],
            confidenceDisplayed: "high",
            sourceMix: SourceMix(usedVision: true, usedMemory: false, usedSearch: true),
            costUsdTotal: 0.0008,
            latencyMs: 1340
        )))
        .padding()
    }
}

#Preview("Fallback card") {
    CardView(card: .fallback(FallbackCard(
        headline: "Not sure what this is.",
        observation: "I can see a weathered stone structure partially hidden by foliage.",
        suggestion: "Try a clearer angle or move closer."
    )))
    .padding()
}
