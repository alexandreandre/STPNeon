/**
 * Modèles recommandés pour l’assistant RAG / usage interne (ordre, préfixes OpenRouter, fiche méthodo).
 */

export type RelevantGuideKind = "proprietary_api" | "open_weights";

export interface RelevantModelGuideRow {
  order: number;
  matchPrefixes: string[];
  label: string;
  vendor: string;
  kind: RelevantGuideKind;
  staticPricingUsdPer1m?: { input: number; output: number };
  /** Préfixe ~ sur les montants (estimation hors catalogue). */
  pricingApprox?: boolean;
  /** Pas de tarif API pertinent (cible inférence locale). */
  hideApiPricing?: boolean;
  strengths: string[];
  weaknesses: string[];
}

export const RELEVANT_MODEL_GUIDE: RelevantModelGuideRow[] = [
  {
    order: 0,
    matchPrefixes: ["openai/gpt-4o-mini"],
    label: "GPT-4o Mini",
    vendor: "OpenAI",
    kind: "proprietary_api",
    staticPricingUsdPer1m: { input: 0.15, output: 0.6 },
    strengths: [
      "Excellent rapport qualité / prix",
      "Rapide et fiable, contexte 128k",
      "Très bon niveau en français",
    ],
    weaknesses: [
      "Moins performant sur les tâches très complexes"
    ],
  },
  {
    order: 1,
    matchPrefixes: [
      "google/gemini-3.1-flash-lite-preview",
      "google/gemini-3.1-flash-lite",
    ],
    label: "Gemini 3.1 Flash Lite",
    vendor: "Google",
    kind: "proprietary_api",
    staticPricingUsdPer1m: { input: 0.25, output: 1.5 },
    strengths: [
      "Très grande fenêtre de contexte (jusqu’à ~1M tokens selon offre)",
      "Pensé pour le fort volume ; gains vs Gemini 2.5 Flash Lite sur RAG, extraction, traduction, etc."
    ],
    weaknesses: [
      "Coût de sortie nettement plus élevé que l’entrée (à surveiller sur longues réponses)",
    ],
  },
  {
    order: 2,
    matchPrefixes: ["google/gemini-2.5-flash-lite"],
    label: "Gemini 2.5 Flash Lite",
    vendor: "Google",
    kind: "proprietary_api",
    staticPricingUsdPer1m: { input: 0.1, output: 0.4 },
    pricingApprox: true,
    strengths: ["Parmi les moins chers du segment", "Contexte long, adapté au fort volume"],
    weaknesses: ["Moins performant sur les tâches complexes"],
  },
  {
    order: 3,
    matchPrefixes: [
      "mistralai/mistral-small-3.2-24b-instruct",
      "mistralai/mistral-small-3.2-24b",
      "mistralai/mistral-small-3.2",
    ],
    label: "Mistral Small 3.2",
    vendor: "Mistral AI",
    kind: "proprietary_api",
    staticPricingUsdPer1m: { input: 0.1, output: 0.3 },
    strengths: [
      "Niveau SOTA sur sa catégorie, multimodal et multilingue",
      "Licence Apache 2.0 sur les poids (accès souvent via API Mistral / OpenRouter)",
      "API et fine-tuning disponibles côté Mistral",
      "Fournisseur européen (cadre RGPD à clarifier avec votre contrat)",
    ],
    weaknesses: [
      "Scénarios multimodaux ou agentiques plus exigeants qu’un simple texte-texte",
      "Prix de sortie plus élevé que certaines offres « lite » du marché",
    ],
  },
  {
    order: 4,
    matchPrefixes: ["meta-llama/llama-3.1-8b"],
    label: "Llama 3.1 8B Instruct",
    vendor: "Meta",
    kind: "open_weights",
    hideApiPricing: true,
    strengths: [
      "Très léger, rapide en local",
      "Bon rapport qualité / coût"
    ],
    weaknesses: ["GPU conseillé (≈8 Go VRAM mini)"],
  },
  {
    order: 5,
    matchPrefixes: [
      "mistralai/mistral-7b",
      "mistralai/mistral-nemo",
      "mistralai/open-mistral-7b",
      "mistral/mistral-7b",
      "mistral/mistral-nemo",
      "mistral/open-mistral",
      "mistral/open-mistral-7b",
    ],
    label: "Mistral 7B / Nemo",
    vendor: "Mistral AI",
    kind: "open_weights",
    hideApiPricing: true,
    strengths: ["Très efficace pour sa taille", "Excellent français"],
    weaknesses: [
      "Performance limitée sur le raisonnement très complexe"
    ],
  },
  {
    order: 6,
    matchPrefixes: [
      "qwen/qwen-2.5-7b",
      "qwen/qwen-2.5-14b",
      "qwen/qwen2.5-7b",
      "qwen/qwen2.5-14b",
    ],
    label: "Qwen 2.5 7B / 14B",
    vendor: "Alibaba",
    kind: "open_weights",
    hideApiPricing: true,
    strengths: [
      "Très bon français",
      "Excellent rapport performance / taille",
      "Contexte jusqu’à 128k (selon variante)",
    ],
    weaknesses: ["Écosystème un peu moins mature que les grands providers"],
  },
  {
    order: 7,
    matchPrefixes: ["google/gemma-2-9b", "google/gemma-2-27b"],
    label: "Gemma 2 9B / 27B",
    vendor: "Google DeepMind",
    kind: "open_weights",
    hideApiPricing: true,
    strengths: ["Compact, optimisé pour l’inférence", "Bonne qualité de raisonnement"],
    weaknesses: [
      "Moins bon que Llama 70B sur les tâches difficiles",
    ],
  },
];

type LicenseSplit = "proprietary" | "open_weights" | "unknown";

/**
 * Premier modèle dont l’id correspond à l’un des préfixes (ordre des préfixes du guide,
 * puis premier id parmi les correspondances après tri stable par id).
 * Même règle pour le tableau « Modèles pertinents » du comparateur et le sélecteur de l’assistant
 * (indépendant de l’ordre renvoyé par l’API).
 */
export function findCatalogMatchForRelevantGuide<T extends { id: string }>(
  catalog: T[] | undefined,
  prefixes: string[],
): T | undefined {
  if (!catalog?.length) return undefined;
  const pl = prefixes.map((p) => p.toLowerCase());
  const byId = [...catalog].sort((a, b) => a.id.localeCompare(b.id, undefined, { sensitivity: "base" }));
  for (const p of pl) {
    const hit = byId.find((m) => m.id.toLowerCase().startsWith(p));
    if (hit) return hit;
  }
  return undefined;
}

export function partitionModelsForAssistant<T extends { id: string; license_kind?: LicenseSplit }>(
  models: T[],
  guide: RelevantModelGuideRow[] = RELEVANT_MODEL_GUIDE,
): {
  relevantProprietary: T[];
  relevantOpenWeights: T[];
  restByLicense: { proprietary: T[]; openWeights: T[]; unknown: T[] };
} {
  const sortedGuide = [...guide].sort((a, b) => a.order - b.order);
  const used = new Set<string>();
  const relevantProprietary: T[] = [];
  const relevantOpenWeights: T[] = [];

  for (const row of sortedGuide) {
    const match = findCatalogMatchForRelevantGuide(models, row.matchPrefixes);
    if (match && !used.has(match.id)) {
      used.add(match.id);
      if (row.kind === "proprietary_api") relevantProprietary.push(match);
      else relevantOpenWeights.push(match);
    }
  }

  const proprietary: T[] = [];
  const openWeights: T[] = [];
  const unknown: T[] = [];
  for (const m of models) {
    if (used.has(m.id)) continue;
    const k = m.license_kind ?? "unknown";
    if (k === "proprietary") proprietary.push(m);
    else if (k === "open_weights") openWeights.push(m);
    else unknown.push(m);
  }

  return {
    relevantProprietary,
    relevantOpenWeights,
    restByLicense: { proprietary, openWeights, unknown },
  };
}
