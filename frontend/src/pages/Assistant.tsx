import { useState, useRef, useEffect, useCallback, useMemo, Fragment } from "react";
import { Link } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { supabase } from "@/integrations/supabase/client";
import AppLayout from "@/components/AppLayout";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";

import {
  Send,
  Loader2,
  Plus,
  MessageSquare,
  Trash2,
  Bot,
  Sparkles,
  BookOpen,
  FileText,
  Check,
  ChevronsUpDown,
  ThumbsDown,
  ThumbsUp,
} from "lucide-react";

import { useToast } from "@/hooks/use-toast";
import { cn } from "@/lib/utils";
import ReactMarkdown from "react-markdown";
import DocumentMention, { type Doc } from "@/components/assistant/DocumentMention";
import MentionedDocBadge from "@/components/assistant/MentionedDocBadge";
import { extractPdfText, isPdfPlaceholderContent } from "@/lib/pdf-text";
import { getApiBaseUrl } from "@/lib/api";
import { fetchAssistantGameQuestions, type AssistantGameQuestion } from "@/lib/assistantGameQuestions";
import { partitionModelsForAssistant } from "@/lib/relevantModels";

interface Msg {
  role: "user" | "assistant";
  content: string;
}

interface MessageMeta {
  provider: string;
  model: string;
  timing?: {
    response_time_ms?: number;
    first_token_ms?: number;
    retrieval_ms?: number;
  };
  usage?: {
    llm?: {
      prompt_tokens?: number;
      completion_tokens?: number;
      total_tokens?: number;
    };
    embeddings?: {
      prompt_tokens?: number;
      total_tokens?: number;
    };
    cost?: {
      llm_usd?: number;
      embeddings_usd?: number;
      total_usd?: number;
    };
    // Pour compat : usage brut renvoyé par OpenRouter (prompt_tokens, completion_tokens, total_tokens, cost, etc.)
    raw?: unknown;
  };
}

interface Conversation {
  id: string;
  title: string;
  updated_at: string;
}

type LicenseKind = "proprietary" | "open_weights" | "unknown";

interface OpenRouterModel {
  id: string;
  name: string;
  description?: string;
  context_length: number;
  license_kind?: LicenseKind;
  pricing_per_1m_usd?: {
    input?: number | null;
    output?: number | null;
  };
  pricing?: {
    prompt?: string;
    completion?: string;
  };
}

function formatPricingPer1mUsd(
  input: number | null | undefined,
  output: number | null | undefined,
): string | null {
  if (input == null && output == null) return null;
  const fmt = (n: number) =>
    n.toLocaleString("fr-FR", { minimumFractionDigits: 2, maximumFractionDigits: 4 });
  const inPart = input != null ? `${fmt(input)} $` : "—";
  const outPart = output != null ? `${fmt(output)} $` : "—";
  return `Entrée ${inPart} · Sortie ${outPart} / 1M tokens`;
}

const getChatUrl = () => `${getApiBaseUrl()}/chat`;

const STPNEON_OPENWEBUI_MODEL_ID = "stpneon/openwebui";

function chatProviderForModel(modelId: string): string {
  return modelId === STPNEON_OPENWEBUI_MODEL_ID ? "openwebui" : "openrouter";
}

/** Dernier modèle OpenRouter choisi ou ayant servi à une réponse réussie (préféré au default API). */
const LAST_OPENROUTER_MODEL_KEY = "stpneon_last_openrouter_model";

function readPersistedOpenRouterModel(): string | null {
  try {
    return window.localStorage.getItem(LAST_OPENROUTER_MODEL_KEY);
  } catch {
    return null;
  }
}

function persistOpenRouterModel(modelId: string) {
  try {
    window.localStorage.setItem(LAST_OPENROUTER_MODEL_KEY, modelId);
  } catch {
    /* quota / navigation privée */
  }
}

function resolveOpenRouterModelSelection(models: OpenRouterModel[], apiDefault?: string): string {
  const ids = new Set(models.map((m) => m.id));
  const saved = readPersistedOpenRouterModel();
  if (saved && (ids.has(saved) || saved === "openrouter/auto")) return saved;
  if (apiDefault) return apiDefault;
  return "openrouter/auto";
}

export default function Assistant() {
  const { user, profile, role } = useAuth();
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [selectedModel, setSelectedModel] = useState<string>("openrouter/auto");
  const [modelPopoverOpen, setModelPopoverOpen] = useState(false);
  const [availableModels, setAvailableModels] = useState<OpenRouterModel[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConvId, setActiveConvId] = useState<string | null>(null);
  const [mentionOpen, setMentionOpen] = useState(false);
  const [mentionQuery, setMentionQuery] = useState("");
  const [mentionedDocs, setMentionedDocs] = useState<Doc[]>([]);
  const [docCount, setDocCount] = useState(0);
  /** 1 = pouce bas, 2 = pouce haut (aligné sur l’API). */
  const [ratings, setRatings] = useState<Record<number, 1 | 2>>({});
  const [ratingsSent, setRatingsSent] = useState<Record<number, boolean>>({});
  const [responseTimeMs, setResponseTimeMs] = useState<number>(0);
  const [sendTime, setSendTime] = useState<number>(0);
  const [messageMetas, setMessageMetas] = useState<Record<number, MessageMeta | undefined>>({});
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();
  const [gameQuestions, setGameQuestions] = useState<AssistantGameQuestion[]>([]);

  const applyGameQuestion = useCallback((text: string) => {
    setInput(text);
    requestAnimationFrame(() => inputRef.current?.focus());
  }, []);

  const modelSelectorSections = useMemo(() => {
    const { relevantProprietary, relevantOpenWeights, restByLicense } =
      partitionModelsForAssistant(availableModels);
    const otherNonOpenSource = [...restByLicense.proprietary, ...restByLicense.unknown];

    type Sub = { key: string; subHeading: string; items: OpenRouterModel[] };
    type Block = { key: string; mainHeading: string; subsections: Sub[] };

    const blocks: Block[] = [];

    const relevantSubsections: Sub[] = [
      { key: "non-open", subHeading: "Non open-source", items: relevantProprietary },
      { key: "open", subHeading: "Open-source", items: relevantOpenWeights },
    ].filter((s) => s.items.length > 0);

    if (relevantSubsections.length > 0) {
      blocks.push({
        key: "relevant-rag",
        mainHeading: "Modèles pertinents pour notre usage (RAG interne)",
        subsections: relevantSubsections,
      });
    }

    const otherSubsections: Sub[] = [
      { key: "non-open", subHeading: "Non open-source", items: otherNonOpenSource },
      { key: "open", subHeading: "Open-source", items: restByLicense.openWeights },
    ].filter((s) => s.items.length > 0);

    if (otherSubsections.length > 0) {
      blocks.push({
        key: "other",
        mainHeading: "Autres modèles",
        subsections: otherSubsections,
      });
    }

    return blocks;
  }, [availableModels]);

  useEffect(() => {
    fetchAssistantGameQuestions()
      .then(setGameQuestions)
      .catch(() => setGameQuestions([]));
  }, []);

  // Fetch des modèles OpenRouter disponibles au montage
  useEffect(() => {
    fetch(`${getApiBaseUrl()}/api/llm/openrouter/models`)
      .then((r) => r.json())
      .then((data: { default_model?: string; models?: OpenRouterModel[] }) => {
        const list = Array.isArray(data.models) ? data.models : [];
        if (list.length) setAvailableModels(list);
        setSelectedModel(resolveOpenRouterModelSelection(list, data.default_model));
      })
      .catch(() => {
        // En cas d'erreur on garde le modèle par défaut "openrouter/auto"
      });
  }, []);

  const openDocById = async (docId: string) => {
    const { data } = await supabase
      .from("knowledge_documents")
      .select("title, file_path")
      .eq("id", docId)
      .single();
    if (!data?.file_path) return;

    const { data: blob, error } = await supabase.storage
      .from("knowledge-files")
      .download(data.file_path);
    if (error || !blob) {
      toast({ title: "Erreur", description: "Impossible de prévisualiser ce PDF.", variant: "destructive" });
      return;
    }

    const blobUrl = URL.createObjectURL(blob);
    // Ouverture en prévisualisation dans un nouvel onglet (sans téléchargement forcé)
    window.open(blobUrl, "_blank", "noopener,noreferrer");
    setTimeout(() => URL.revokeObjectURL(blobUrl), 60_000);
  };

  

  // Fetch doc count for display
  useEffect(() => {
    supabase.from("knowledge_documents").select("id", { count: "exact", head: true }).then(({ count }) => {
      setDocCount(count ?? 0);
    });
  }, []);

  const loadMessages = useCallback(async (convId: string) => {
    const { data } = await supabase
      .from("chat_messages")
      .select("role, content")
      .eq("conversation_id", convId)
      .order("created_at", { ascending: true });
    setMessages((data as Msg[]) ?? []);
  }, []);

  const loadConversations = useCallback(async () => {
    if (!user) return;
    const { data } = await supabase
      .from("conversations")
      .select("id, title, updated_at")
      .order("updated_at", { ascending: false });
    setConversations(data ?? []);
  }, [user]);

  useEffect(() => {
    loadConversations();
  }, [loadConversations]);

  // Au montage, on restaure la dernière conversation active si elle existe
  useEffect(() => {
    const savedId = window.localStorage.getItem("stpneon_active_conversation_id");
    if (savedId) {
      setActiveConvId(savedId);
      loadMessages(savedId);
    }
  }, [loadMessages]);

  // Persistance locale des métadonnées de messages et des notations par conversation
  useEffect(() => {
    if (!activeConvId) return;
    const metasRaw = window.localStorage.getItem(`stpneon_message_metas_${activeConvId}`);
    const ratingsRaw = window.localStorage.getItem(`stpneon_thumbs_${activeConvId}`);
    const ratingsSentRaw = window.localStorage.getItem(`stpneon_thumbs_sent_${activeConvId}`);
    if (metasRaw) {
      try {
        setMessageMetas(JSON.parse(metasRaw));
      } catch {
        // ignore JSON invalide
      }
    } else {
      setMessageMetas({});
    }
    if (ratingsRaw) {
      try {
        const parsed = JSON.parse(ratingsRaw) as Record<string, number>;
        const next: Record<number, 1 | 2> = {};
        for (const [k, v] of Object.entries(parsed)) {
          const idx = Number(k);
          if (!Number.isFinite(idx)) continue;
          if (v === 1 || v === 2) next[idx] = v;
        }
        setRatings(next);
      } catch {
        setRatings({});
      }
    } else {
      setRatings({});
    }
    if (ratingsSentRaw) {
      try {
        setRatingsSent(JSON.parse(ratingsSentRaw));
      } catch {
        setRatingsSent({});
      }
    } else {
      setRatingsSent({});
    }
  }, [activeConvId]);

  useEffect(() => {
    if (!activeConvId) return;
    window.localStorage.setItem(
      `stpneon_message_metas_${activeConvId}`,
      JSON.stringify(messageMetas),
    );
  }, [activeConvId, messageMetas]);

  useEffect(() => {
    if (!activeConvId) return;
    window.localStorage.setItem(`stpneon_thumbs_${activeConvId}`, JSON.stringify(ratings));
  }, [activeConvId, ratings]);

  useEffect(() => {
    if (!activeConvId) return;
    window.localStorage.setItem(
      `stpneon_thumbs_sent_${activeConvId}`,
      JSON.stringify(ratingsSent),
    );
  }, [activeConvId, ratingsSent]);

  const selectConversation = (convId: string) => {
    setActiveConvId(convId);
    window.localStorage.setItem("stpneon_active_conversation_id", convId);
    loadMessages(convId);
  };

  const startNewConversation = () => {
    if (activeConvId) {
      window.localStorage.removeItem(`stpneon_message_metas_${activeConvId}`);
      window.localStorage.removeItem(`stpneon_thumbs_${activeConvId}`);
      window.localStorage.removeItem(`stpneon_thumbs_sent_${activeConvId}`);
      window.localStorage.removeItem(`stpneon_ratings_${activeConvId}`);
      window.localStorage.removeItem(`stpneon_ratings_sent_${activeConvId}`);
    }
    setActiveConvId(null);
    window.localStorage.removeItem("stpneon_active_conversation_id");
    setMessages([]);
    setMentionedDocs([]);
    setMessageMetas({});
    setRatings({});
    setRatingsSent({});
  };

  const deleteConversation = async (convId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    await supabase.from("conversations").delete().eq("id", convId);
    if (activeConvId === convId) {
      window.localStorage.removeItem(`stpneon_message_metas_${convId}`);
      window.localStorage.removeItem(`stpneon_thumbs_${convId}`);
      window.localStorage.removeItem(`stpneon_thumbs_sent_${convId}`);
      window.localStorage.removeItem(`stpneon_ratings_${convId}`);
      window.localStorage.removeItem(`stpneon_ratings_sent_${convId}`);
      setActiveConvId(null);
      setMessages([]);
      setMessageMetas({});
      setRatings({});
      setRatingsSent({});
    }
    loadConversations();
  };

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages]);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    setInput(val);
    const atMatch = val.match(/@(\S*)$/);
    if (atMatch) {
      setMentionOpen(true);
      setMentionQuery(atMatch[1]);
    } else {
      setMentionOpen(false);
      setMentionQuery("");
    }
  };

  const handleDocSelect = (doc: Doc) => {
    const newInput = input.replace(/@\S*$/, "").trim();
    setInput(newInput);
    setMentionOpen(false);
    setMentionQuery("");
    if (!mentionedDocs.find((d) => d.id === doc.id)) {
      setMentionedDocs((prev) => [...prev, doc]);
    }
    inputRef.current?.focus();
  };

  const removeMentionedDoc = (docId: string) => {
    setMentionedDocs((prev) => prev.filter((d) => d.id !== docId));
  };

  const saveMessage = async (convId: string, msg: Msg) => {
    await supabase.from("chat_messages").insert({
      conversation_id: convId,
      role: msg.role,
      content: msg.content,
    });
  };

  const handleSend = async (overrideText?: string) => {
    const text = (overrideText || input).trim();
    if (!text || isLoading || !user) return;

    setIsLoading(true);

    let hydratedMentionedDocs = mentionedDocs;

    if (mentionedDocs.length > 0) {
      hydratedMentionedDocs = await Promise.all(
        mentionedDocs.map(async (doc) => {
          if (!isPdfPlaceholderContent(doc.content) || !doc.file_path) return doc;

          try {
            const { data: blob, error } = await supabase.storage
              .from("knowledge-files")
              .download(doc.file_path);

            if (error || !blob) return doc;

            const extractedContent = await extractPdfText(blob);
            if (!extractedContent.trim()) return doc;

            return { ...doc, content: extractedContent };
          } catch {
            return doc;
          }
        }),
      );

      const unresolvedDocs = hydratedMentionedDocs.filter((doc) =>
        isPdfPlaceholderContent(doc.content),
      );

      if (unresolvedDocs.length > 0) {
        toast({
          title: "Extraction PDF incomplète",
          description:
            "Impossible de lire le texte du PDF mentionné. Réuploadez le fichier depuis la Base documentaire pour l’indexer correctement.",
          variant: "destructive",
        });
        setIsLoading(false);
        return;
      }
    }

    let docContext = "";
    if (hydratedMentionedDocs.length > 0) {
      docContext = hydratedMentionedDocs
        .map((d) => `[Document: ${d.title}]\n${d.content}`)
        .join("\n\n");
    }

    const userMsg: Msg = { role: "user", content: text };
    const updatedMessages = [...messages, userMsg];
    setMessages(updatedMessages);
    setInput("");
    setMentionedDocs([]);

    let convId = activeConvId;

    if (!convId) {
      const title = text.length > 50 ? text.slice(0, 50) + "…" : text;
      const { data } = await supabase
        .from("conversations")
        .insert({ user_id: user.id, title })
        .select("id")
        .single();
      if (!data) {
        toast({ title: "Erreur", description: "Impossible de créer la conversation", variant: "destructive" });
        setIsLoading(false);
        return;
      }
      convId = data.id;
      setActiveConvId(convId);
      window.localStorage.setItem("stpneon_active_conversation_id", convId);
      loadConversations();
    } else {
      await supabase.from("conversations").update({ updated_at: new Date().toISOString() }).eq("id", convId);
    }

    await saveMessage(convId, userMsg);

    let assistantSoFar = "";

    const apiMessages = updatedMessages.map((m) => ({ role: m.role, content: m.content }));
    if (docContext) {
      const lastIdx = apiMessages.length - 1;
      apiMessages[lastIdx] = {
        ...apiMessages[lastIdx],
        content: `[Documents référencés]\n${docContext}\n\n[Question]\n${apiMessages[lastIdx].content}`,
      };
    }

    setSendTime(Date.now());
    const modelUsedForRequest = selectedModel;

    try {
      const resp = await fetch(getChatUrl(), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY}`,
        },
        body: JSON.stringify({
          messages: apiMessages,
          role_name: role?.role_name,
          department: profile?.department,
          provider: chatProviderForModel(selectedModel),
          model: selectedModel,
        }),
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ error: "Erreur réseau" }));
        toast({ title: "Erreur", description: err.error, variant: "destructive" });
        setIsLoading(false);
        return;
      }

      if (!resp.body) throw new Error("No response body");

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let textBuffer = "";
      let streamDone = false;

      while (!streamDone) {
        const { done, value } = await reader.read();
        if (done) break;
        textBuffer += decoder.decode(value, { stream: true });

        let newlineIndex: number;
        while ((newlineIndex = textBuffer.indexOf("\n")) !== -1) {
          let line = textBuffer.slice(0, newlineIndex);
          textBuffer = textBuffer.slice(newlineIndex + 1);

          if (line.endsWith("\r")) line = line.slice(0, -1);
          if (line.startsWith(":") || line.trim() === "") continue;
          if (!line.startsWith("data: ")) continue;

          const jsonStr = line.slice(6).trim();
          if (jsonStr === "[DONE]") {
            setResponseTimeMs(Date.now() - sendTime);
            streamDone = true;
            break;
          }

          try {
            const parsed = JSON.parse(jsonStr);
            // Event "meta" envoyé en fin de stream avec les stats détaillées
            if (parsed.meta) {
              setResponseTimeMs(parsed.meta?.timing?.response_time_ms ?? Date.now() - sendTime);
              // On attache toujours les métadonnées au DERNIER message de la liste,
              // qui est l'assistant en cours de génération.
              setMessages((prev) => {
                const lastIndex = Math.max(0, prev.length - 1);
                setMessageMetas((prevMeta) => ({
                  ...prevMeta,
                  [lastIndex]: parsed.meta as MessageMeta,
                }));
                return prev;
              });
              continue;
            }

            const content = parsed.choices?.[0]?.delta?.content as string | undefined;
            if (content) {
              assistantSoFar += content;
              setMessages((prev) => {
                const last = prev[prev.length - 1];
                if (last?.role === "assistant") {
                  return prev.map((m, i) =>
                    i === prev.length - 1 ? { ...m, content: assistantSoFar } : m,
                  );
                }
                return [...prev, { role: "assistant", content: assistantSoFar }];
              });
            }
          } catch {
            textBuffer = line + "\n" + textBuffer;
            break;
          }
        }
      }

      if (assistantSoFar && convId) {
        await saveMessage(convId, { role: "assistant", content: assistantSoFar });
      }
      persistOpenRouterModel(modelUsedForRequest);
    } catch (e) {
      console.error(e);
      toast({ title: "Erreur", description: "Impossible de contacter l'assistant.", variant: "destructive" });
    }

    setIsLoading(false);
    loadConversations();
  };

  const handleRating = async (messageIndex: number, rating: 1 | 2, response: string) => {
    setRatings((prev) => ({ ...prev, [messageIndex]: rating }));

    const userMessage = messages[messageIndex - 1]?.content ?? "";

    try {
      await fetch(`${getApiBaseUrl()}/api/feedback/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          provider: chatProviderForModel(selectedModel),
          model: selectedModel,
          prompt: userMessage,
          response,
          rating,
          response_time_ms: responseTimeMs,
          conversation_id: activeConvId ?? undefined,
        }),
      });
      setRatingsSent((prev) => ({ ...prev, [messageIndex]: true }));
    } catch (err) {
      console.error("Erreur envoi feedback:", err);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (mentionOpen) return;
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const selectedModelEntry = availableModels.find((m) => m.id === selectedModel);
  const selectedModelLabel = selectedModelEntry?.name ?? selectedModel ?? "Choisir un modèle";
  const selectedModelPricingLine = selectedModelEntry
    ? formatPricingPer1mUsd(
        selectedModelEntry.pricing_per_1m_usd?.input ?? undefined,
        selectedModelEntry.pricing_per_1m_usd?.output ?? undefined,
      )
    : null;

  const formatModelLabel = (modelIdOrName: string | undefined) => {
    if (!modelIdOrName) return "Modèle OpenRouter";
    const fromList = availableModels.find((m) => m.id === modelIdOrName || m.name === modelIdOrName);
    if (fromList?.name) return fromList.name;
    if (modelIdOrName === STPNEON_OPENWEBUI_MODEL_ID) return "STPNeon OpenWebUI";
    if (modelIdOrName.startsWith("openrouter/")) {
      const raw = modelIdOrName.replace("openrouter/", "");
      return raw
        .split(/[-_]/)
        .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
        .join(" ");
    }
    return modelIdOrName;
  };

  return (
    <AppLayout>
      <div className="flex gap-4 h-[calc(100vh-7rem)]">
        {/* Conversations sidebar */}
        <div className="w-56 shrink-0 hidden md:flex flex-col gap-2">
          <Button size="sm" variant="outline" className="w-full" onClick={startNewConversation}>
            <Plus className="mr-2 h-4 w-4" /> Nouvelle conversation
          </Button>
          <div className="flex-1 overflow-y-auto space-y-1">
            {conversations.map((conv) => (
              <button
                key={conv.id}
                onClick={() => selectConversation(conv.id)}
                className={`w-full text-left text-xs p-2 rounded-md flex items-center gap-2 group ${
                  activeConvId === conv.id ? "bg-muted text-foreground font-medium" : "text-muted-foreground hover:bg-muted/50"
                }`}
              >
                <MessageSquare className="h-3 w-3 shrink-0" />
                <span className="truncate flex-1">{conv.title}</span>
                <Trash2
                  className="h-3 w-3 shrink-0 opacity-0 group-hover:opacity-100 hover:text-destructive"
                  onClick={(e) => deleteConversation(conv.id, e)}
                />
              </button>
            ))}
          </div>
        </div>

        {/* Chat area */}
        <div className="flex-1 flex flex-col min-w-0">
          <div className="flex items-center gap-3 mb-4">
            <div className="h-10 w-10 rounded-full bg-primary/10 flex items-center justify-center">
              <Bot className="h-5 w-5 text-primary" />
            </div>
            <div>
              <h1 className="text-xl font-semibold text-foreground">Assistant IA Interne</h1>
              <div className="flex items-center gap-2">
                <p className="text-xs text-muted-foreground flex items-center gap-2">
                  <span className="flex items-center gap-1">
                    <BookOpen className="h-3 w-3" /> {docCount} documents indexés
                  </span>
                </p>
                <Badge variant="secondary" className="text-[11px] px-2 py-0.5 flex items-center gap-1">
                  <Sparkles className="h-3 w-3" />
                  <span className="font-medium">
                    {formatModelLabel(
                      availableModels.find((m) => m.id === selectedModel)?.name ?? selectedModel,
                    )}
                  </span>
                </Badge>
              </div>
            </div>
          </div>

          <div className="flex gap-2 mb-3 flex-wrap">
            <Badge variant="outline" className="text-xs">{role?.role_name ?? "Non assigné"}</Badge>
            <Badge variant="outline" className="text-xs">{profile?.department ?? "Non assigné"}</Badge>
          </div>

          <Card className="flex-1 flex flex-col min-h-0">
            <CardContent ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-3">
              {messages.length === 0 && (
                <div className="flex flex-col items-center justify-center h-full gap-6">
                  <div className="text-center space-y-2">
                    <div className="h-16 w-16 rounded-full bg-primary/10 flex items-center justify-center mx-auto mb-4">
                      <Sparkles className="h-8 w-8 text-primary" />
                    </div>
                    <h2 className="text-lg font-medium text-foreground">Comment puis-je vous aider ?</h2>
                    <p className="text-sm text-muted-foreground max-w-md">
                      Je connais l'intégralité de votre base documentaire. Posez-moi une question ou utilisez <kbd className="px-1.5 py-0.5 bg-muted rounded text-xs font-mono">@</kbd> pour cibler un document précis.
                    </p>
                  </div>
                  {gameQuestions.length > 0 ? (
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 w-full max-w-lg">
                      {gameQuestions.map((prompt, i) => (
                        <button
                          key={`${i}-${prompt.text.slice(0, 24)}`}
                          type="button"
                          onClick={() => applyGameQuestion(prompt.text)}
                          className="text-left p-3 rounded-lg border border-border hover:bg-muted/50 hover:border-primary/30 transition-colors text-sm"
                        >
                          <span className="mr-2">{prompt.icon}</span>
                          <span className="text-muted-foreground">{prompt.text}</span>
                        </button>
                      ))}
                    </div>
                  ) : (
                    <p className="text-sm text-muted-foreground text-center max-w-md">
                      Aucune suggestion pour le moment. Ajoutez-en depuis la page{" "}
                      <Link
                        to="/jeu-questions"
                        className="text-primary underline underline-offset-2 hover:text-primary/80 font-medium"
                      >
                        Jeu de questions
                      </Link>
                      .
                    </p>
                  )}
                </div>
              )}
              {messages.map((msg, i) => (
                <div key={i} className={`flex flex-col ${msg.role === "user" ? "items-end" : "items-start"}`}>
                  <div
                    className={`rounded-lg px-3 py-2 max-w-[75%] text-sm ${
                      msg.role === "user" ? "bg-primary text-primary-foreground" : "bg-muted text-foreground"
                    }`}
                  >
                    {msg.role === "assistant" ? (
                      <div className="prose prose-sm max-w-none dark:prose-invert [&_a]:text-primary [&_a]:underline [&_a]:cursor-pointer">
                        <ReactMarkdown
                          components={{
                            a: ({ href, children, ...props }) => {
                              const docMatch = href?.match(/\/knowledge-base\?doc=([a-f0-9-]+)/);
                              if (docMatch) {
                                return (
                                  <a
                                    {...props}
                                    href="#"
                                    onClick={(e) => {
                                      e.preventDefault();
                                      openDocById(docMatch[1]);
                                    }}
                                    className="inline-flex items-center gap-1 text-primary underline hover:text-primary/80 cursor-pointer"
                                  >
                                    <FileText className="h-3 w-3 inline" />
                                    {children}
                                  </a>
                                );
                              }
                              return (
                                <a {...props} href={href} target="_blank" rel="noopener noreferrer">
                                  {children}
                                </a>
                              );
                            },
                          }}
                        >
                          {msg.content}
                        </ReactMarkdown>
                      </div>
                    ) : (
                      msg.content
                    )}
                  </div>
                  {msg.role === "assistant" && (
                    <div className="flex flex-col items-start gap-1 mt-1 text-xs">
                      {(() => {
                        const meta = messageMetas[i];
                        if (!meta) return null;
                        const t = meta.timing;
                        const u = meta.usage as any | undefined;
                        const llmTokensTotal =
                          u?.llm?.total_tokens != null
                            ? u.llm.total_tokens
                            : u?.raw?.total_tokens;
                        const embedTokensTotal =
                          u?.embeddings?.total_tokens != null
                            ? u.embeddings.total_tokens
                            : undefined;
                        const tokensCombined =
                          llmTokensTotal != null || embedTokensTotal != null
                            ? (llmTokensTotal ?? 0) + (embedTokensTotal ?? 0)
                            : null;
                        const totalCostUsd =
                          u?.cost?.total_usd != null
                            ? u.cost.total_usd
                            : u?.raw?.cost;
                        return (
                          <div className="flex flex-col gap-1 px-3 py-2 rounded-lg bg-muted/70 text-muted-foreground border border-border/60 max-w-[420px]">
                            <div className="flex items-center justify-between gap-2">
                              <span className="font-semibold text-foreground text-[11px] uppercase tracking-wide">
                                Statistiques de cette réponse
                              </span>
                              <span className="text-[11px] text-muted-foreground/90">
                                {formatModelLabel(meta.model || selectedModel)}
                              </span>
                            </div>
                            <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-[11px]">
                              <div>
                                <div className="text-muted-foreground/80">Temps 1er token</div>
                                <div className="font-mono">
                                  {t?.first_token_ms != null
                                    ? `${t.first_token_ms.toLocaleString("fr-FR")} ms`
                                    : "—"}
                                </div>
                              </div>
                              <div>
                                <div className="text-muted-foreground/80">Temps total</div>
                                <div className="font-mono">
                                  {(t?.response_time_ms ?? responseTimeMs) > 0
                                    ? `~${(t?.response_time_ms ?? responseTimeMs).toLocaleString("fr-FR")} ms`
                                    : "—"}
                                </div>
                              </div>
                              <div>
                                <div className="text-muted-foreground/80">Tokens</div>
                                <div className="font-mono">
                                  {tokensCombined != null
                                    ? tokensCombined.toLocaleString("fr-FR")
                                    : "—"}
                                </div>
                              </div>
                              <div>
                                <div className="text-muted-foreground/80">Coût total</div>
                                <div className="font-mono">
                                  {totalCostUsd != null
                                    ? `$${totalCostUsd.toFixed(6)}`
                                    : "—"}
                                </div>
                              </div>
                            </div>
                          </div>
                        );
                      })()}
                      {messageMetas[i] && (
                        <div className="mt-1 max-w-[min(100%,420px)] rounded-lg border border-emerald-400/35 bg-gradient-to-r from-emerald-500/[0.14] via-sky-400/[0.1] to-rose-500/[0.14] px-2.5 py-2 shadow-sm ring-1 ring-inset ring-white/40 dark:border-emerald-500/25 dark:from-emerald-500/20 dark:via-sky-500/15 dark:to-rose-500/20 dark:ring-white/10">
                          {ratingsSent[i] ? (
                            <div
                              className={cn(
                                "flex items-center gap-2.5 text-xs font-medium leading-snug",
                                ratings[i] === 2
                                  ? "text-emerald-800 dark:text-emerald-300"
                                  : "text-rose-800 dark:text-rose-300",
                              )}
                            >
                              {ratings[i] === 2 ? (
                                <ThumbsUp
                                  className="h-8 w-8 shrink-0 fill-emerald-500 stroke-emerald-800 drop-shadow-sm dark:fill-emerald-400 dark:stroke-emerald-100"
                                  strokeWidth={1.5}
                                />
                              ) : (
                                <ThumbsDown
                                  className="h-8 w-8 shrink-0 fill-rose-500 stroke-rose-900 drop-shadow-sm dark:fill-rose-400 dark:stroke-rose-100"
                                  strokeWidth={1.5}
                                />
                              )}
                              <span>Merci pour votre retour</span>
                            </div>
                          ) : (
                            <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
                              <p className="text-xs font-medium leading-snug text-sky-900 dark:text-sky-200">
                                Cette réponse vous a-t-elle aidé ?
                              </p>
                              <div
                                className="flex items-center gap-4"
                                role="group"
                                aria-label="Évaluation de la réponse"
                              >
                                <button
                                  type="button"
                                  onClick={() => handleRating(i, 2, msg.content)}
                                  className={cn(
                                    "-m-0.5 rounded-md p-0.5 transition-transform duration-150",
                                    "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500/60 focus-visible:ring-offset-2 focus-visible:ring-offset-background",
                                    "active:scale-90",
                                  )}
                                  aria-label="Pouce haut — utile"
                                  aria-pressed={ratings[i] === 2}
                                >
                                  <ThumbsUp
                                    className={cn(
                                      "h-8 w-8 shrink-0 transition-colors duration-150 drop-shadow-sm",
                                      ratings[i] === 2
                                        ? "fill-emerald-500 stroke-emerald-800 dark:fill-emerald-400 dark:stroke-emerald-100"
                                        : "fill-none stroke-emerald-600 hover:stroke-emerald-700 hover:drop-shadow-md dark:stroke-emerald-400/90 dark:hover:stroke-emerald-300",
                                    )}
                                    strokeWidth={ratings[i] === 2 ? 1.5 : 2.25}
                                  />
                                </button>
                                <button
                                  type="button"
                                  onClick={() => handleRating(i, 1, msg.content)}
                                  className={cn(
                                    "-m-0.5 rounded-md p-0.5 transition-transform duration-150",
                                    "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-rose-500/60 focus-visible:ring-offset-2 focus-visible:ring-offset-background",
                                    "active:scale-90",
                                  )}
                                  aria-label="Pouce bas — pas utile"
                                  aria-pressed={ratings[i] === 1}
                                >
                                  <ThumbsDown
                                    className={cn(
                                      "h-8 w-8 shrink-0 transition-colors duration-150 drop-shadow-sm",
                                      ratings[i] === 1
                                        ? "fill-rose-500 stroke-rose-900 dark:fill-rose-400 dark:stroke-rose-100"
                                        : "fill-none stroke-rose-600 hover:stroke-rose-700 hover:drop-shadow-md dark:stroke-rose-400/90 dark:hover:stroke-rose-300",
                                    )}
                                    strokeWidth={ratings[i] === 1 ? 1.5 : 2.25}
                                  />
                                </button>
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
              {isLoading && messages[messages.length - 1]?.role !== "assistant" && (
                <div className="flex justify-start">
                  <div className="bg-muted rounded-lg px-3 py-2 text-sm flex items-center gap-2">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    <span className="text-xs text-muted-foreground">Recherche dans la base documentaire...</span>
                  </div>
                </div>
              )}
            </CardContent>

            <div className="border-t border-border p-4 space-y-2">
              {/* Sélecteur de modèle OpenRouter (combobox avec recherche intégrée) */}
              {availableModels.length > 0 && (
                <div className="flex flex-col gap-1 text-xs text-muted-foreground">
                  <span className="shrink-0">Modèle OpenRouter :</span>
                  <Popover open={modelPopoverOpen} onOpenChange={setModelPopoverOpen}>
                    <PopoverTrigger asChild>
                      <Button
                        variant="outline"
                        role="combobox"
                        aria-expanded={modelPopoverOpen}
                        className="h-auto min-h-8 max-w-md justify-between text-xs py-1.5 px-3"
                      >
                        <div className="flex flex-col items-start gap-0.5 min-w-0 mr-2 text-left">
                          <span className="truncate w-full">{selectedModelLabel}</span>
                          {selectedModelPricingLine ? (
                            <span className="text-[10px] text-muted-foreground font-normal leading-tight truncate w-full">
                              {selectedModelPricingLine}
                            </span>
                          ) : null}
                        </div>
                        <ChevronsUpDown className="h-3 w-3 shrink-0 opacity-60" />
                      </Button>
                    </PopoverTrigger>
                    <PopoverContent className="w-[min(100vw-2rem,380px)] p-0" align="start">
                      <Command>
                        <CommandInput placeholder="Rechercher un modèle…" className="h-8 text-xs" />
                        <CommandList className="max-h-[min(60vh,320px)]">
                          <CommandEmpty className="text-[11px] text-muted-foreground px-2 py-2">
                            Aucun modèle ne correspond à la recherche.
                          </CommandEmpty>
                          {modelSelectorSections.map((block, blockIdx) => (
                            <Fragment key={block.key}>
                              <div
                                className={cn(
                                  "px-2 pb-1 text-sm font-bold text-foreground leading-snug",
                                  blockIdx > 0 ? "pt-3 border-t border-border mt-1" : "pt-1.5",
                                )}
                              >
                                {block.mainHeading}
                              </div>
                              {block.subsections.map((sub) => (
                                <CommandGroup
                                  key={`${block.key}-${sub.key}`}
                                  heading={sub.subHeading}
                                  className="[&_[cmdk-group-heading]]:font-normal"
                                >
                                  {sub.items.map((m) => {
                                    const priceLine = formatPricingPer1mUsd(
                                      m.pricing_per_1m_usd?.input ?? undefined,
                                      m.pricing_per_1m_usd?.output ?? undefined,
                                    );
                                    return (
                                      <CommandItem
                                        key={m.id}
                                        value={`${m.name ?? m.id} ${m.id} ${priceLine ?? ""}`}
                                        onSelect={() => {
                                          setSelectedModel(m.id);
                                          persistOpenRouterModel(m.id);
                                          setModelPopoverOpen(false);
                                        }}
                                        className="text-xs py-2"
                                      >
                                        <Check
                                          className={cn(
                                            "mr-2 h-3.5 w-3.5 shrink-0",
                                            selectedModel === m.id ? "opacity-100" : "opacity-0",
                                          )}
                                        />
                                        <div className="flex flex-col items-start gap-0.5 min-w-0 flex-1">
                                          <span className="truncate w-full font-medium leading-tight">
                                            {m.name ?? m.id}
                                          </span>
                                          {priceLine ? (
                                            <span className="text-[10px] text-muted-foreground/85 leading-tight">
                                              {priceLine}
                                            </span>
                                          ) : (
                                            <span className="text-[10px] text-muted-foreground/60 italic">
                                              Tarif API non communiqué
                                            </span>
                                          )}
                                        </div>
                                      </CommandItem>
                                    );
                                  })}
                                </CommandGroup>
                              ))}
                            </Fragment>
                          ))}
                        </CommandList>
                      </Command>
                    </PopoverContent>
                  </Popover>
                </div>
              )}

              {mentionedDocs.length > 0 && (
                <div className="flex gap-2 flex-wrap">
                  {mentionedDocs.map((doc) => (
                    <MentionedDocBadge key={doc.id} title={doc.title} content={doc.content} onRemove={() => removeMentionedDoc(doc.id)} />
                  ))}
                </div>
              )}

              {gameQuestions.length > 0 && messages.length > 0 && (
                <div className="flex flex-col gap-1.5">
                  <span className="text-[11px] text-muted-foreground">Idées du jeu de questions</span>
                  <div className="flex gap-2 overflow-x-auto pb-0.5 -mx-0.5 px-0.5 [scrollbar-width:thin]">
                    {gameQuestions.map((q, i) => (
                      <button
                        key={`chip-${i}-${q.text.slice(0, 16)}`}
                        type="button"
                        onClick={() => applyGameQuestion(q.text)}
                        className="flex items-center gap-1.5 shrink-0 max-w-[min(100%,300px)] text-left rounded-full border border-border bg-muted/40 hover:bg-muted/70 px-3 py-1.5 text-xs text-foreground transition-colors"
                        title={q.text}
                      >
                        <span className="shrink-0">{q.icon}</span>
                        <span className="truncate min-w-0">{q.text}</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              <div className="relative flex gap-2">
                <DocumentMention
                  isOpen={mentionOpen}
                  query={mentionQuery}
                  onSelect={handleDocSelect}
                  onClose={() => setMentionOpen(false)}
                />
                <Input
                  ref={inputRef}
                  placeholder="Posez votre question... (@ pour cibler un document)"
                  value={input}
                  onChange={handleInputChange}
                  onKeyDown={handleKeyDown}
                  disabled={isLoading}
                />
                <Button onClick={() => handleSend()} size="icon" disabled={isLoading}>
                  {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
                </Button>
              </div>
            </div>
          </Card>
        </div>



      </div>

    </AppLayout>
  );
}
