// API Query keys constants

export const queryKeyFn = (defaultKey: string[]) => (queryKey: string[]) => [
  ...defaultKey,
  ...queryKey,
];

const STOCK_QUERY_KEYS = {
  watchlist: ["watch"],
  stockList: ["stock"],
  stockDetail: queryKeyFn(["stock", "detail"]),
  stockSearch: queryKeyFn(["stock", "search"]),
  stockPrice: queryKeyFn(["stock", "price"]),
  stockHistory: queryKeyFn(["stock", "history"]),
} as const;

const AGENT_QUERY_KEYS = {
  agentList: queryKeyFn(["agent", "list"]),
  agentInfo: queryKeyFn(["agent", "info"]),
  conversationList: ["conversation"],
} as const;

export const CONVERSATION_QUERY_KEYS = {
  conversationList: ["conversation"],
  conversationHistory: queryKeyFn(["conversation", "history"]),
  conversationTaskList: queryKeyFn(["conversation", "task"]),
  allConversationTaskList: ["all", "conversation", "task"],
} as const;

export const SETTING_QUERY_KEYS = {
  memoryList: ["memory"],
} as const;

export const API_QUERY_KEYS = {
  STOCK: STOCK_QUERY_KEYS,
  AGENT: AGENT_QUERY_KEYS,
  CONVERSATION: CONVERSATION_QUERY_KEYS,
  SETTING: SETTING_QUERY_KEYS,
} as const;

/**
 * UI/API language
 * Priority: VITE_USER_LANGUAGE env > browser language > zh-Hans
 */
export const USER_LANGUAGE =
  (import.meta.env.VITE_USER_LANGUAGE as string | undefined) ||
  (typeof navigator !== "undefined" ? navigator.language : undefined) ||
  "zh-Hans";
