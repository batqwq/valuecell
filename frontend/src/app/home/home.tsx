import { useState } from "react";
import { useNavigate } from "react-router";
import { useAllPollTaskList } from "@/api/conversation";
import ScrollContainer from "@/components/valuecell/scroll/scroll-container";
import { HOME_STOCK_SHOW } from "@/constants/stock";
import { agentSuggestions } from "@/mock/agent-data";
import ChatInputArea from "../agent/components/chat-conversation/chat-input-area";
import {
  AgentSuggestionsList,
  AgentTaskCards,
  SparklineStockList,
} from "./components";
import { useSparklineStocks } from "./hooks/use-sparkline-stocks";

function Home() {
  const navigate = useNavigate();
  const [inputValue, setInputValue] = useState<string>("");

  const { data: allPollTaskList } = useAllPollTaskList();
  const { sparklineStocks } = useSparklineStocks(HOME_STOCK_SHOW);

  const handleAgentClick = (agentId: string) => {
    navigate(`/agent/${agentId}`);
  };

  return (
    <div className="flex h-full min-w-[800px] flex-col gap-3">
      <SparklineStockList stocks={sparklineStocks} />

      {allPollTaskList && allPollTaskList.length > 0 ? (
        <section className="flex flex-1 flex-col items-center justify-between gap-4 overflow-hidden">
          <ScrollContainer>
            <AgentTaskCards tasks={allPollTaskList} />
          </ScrollContainer>

          <ChatInputArea
            className="w-full"
            value={inputValue}
            onChange={(value) => setInputValue(value)}
            onSend={() =>
              navigate("/agent/ValueCellAgent", {
                state: {
                  inputValue,
                },
              })
            }
          />
        </section>
      ) : (
        <section className="flex flex-1 flex-col items-center justify-center gap-8 rounded-lg bg-white py-8">
          <div className="space-y-4 text-center text-gray-950">
            <h1 className="font-medium text-3xl">👋 你好，投资者！</h1>
            <p>在这里你可以分析并跟踪所有感兴趣的股票与加密资产信息。</p>
          </div>

          <ChatInputArea
            className="w-3/4 max-w-[800px]"
            value={inputValue}
            onChange={(value) => setInputValue(value)}
            onSend={() =>
              navigate("/agent/ValueCellAgent", {
                state: {
                  inputValue,
                },
              })
            }
          />

          <AgentSuggestionsList
            suggestions={agentSuggestions.map((suggestion) => ({
              ...suggestion,
              onClick: () => handleAgentClick(suggestion.id),
            }))}
          />
        </section>
      )}
    </div>
  );
}

export default Home;
