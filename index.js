import { ChatOllama } from "@langchain/ollama";
import { tool } from "@langchain/core/tools";
import { z } from "zod";

import {
  StateGraph,
  MessagesAnnotation,
  START,
  END,
} from "@langchain/langgraph";

import {
  ToolNode,
  toolsCondition,
} from "@langchain/langgraph/prebuilt";

// ---------------- LLM ----------------

const llm = new ChatOllama({
  model: "llama3.1:8b",
  temperature: 0,
});

// ---------------- Tools ----------------

const add = tool(
  async ({ a, b }) => String(a + b),
  {
    name: "add",
    description: "Add two numbers",
    schema: z.object({
      a: z.number().describe("First number"),
      b: z.number().describe("Second number"),
    }),
  }
);

const multiply = tool(
  async ({ a, b }) => String(a * b),
  {
    name: "multiply",
    description: "Multiply two numbers",
    schema: z.object({
      a: z.number().describe("First number"),
      b: z.number().describe("Second number"),
    }),
  }
);

const divide = tool(
  async ({ a, b }) => {
    if (b === 0) {
      throw new Error("Cannot divide by zero");
    }

    return String(a / b);
  },
  {
    name: "divide",
    description: "Divide two numbers",
    schema: z.object({
      a: z.number().describe("First number"),
      b: z.number().describe("Second number"),
    }),
  }
);

// ---------------- Bind tools ----------------

const tools = [add, multiply, divide];

const modelWithTools = llm.bindTools(tools);

// ---------------- LLM Node ----------------

async function callModel(state) {
  const response = await modelWithTools.invoke([
    {
      role: "system",
      content:
        "You are a helpful assistant that performs arithmetic using the available tools.",
    },
    ...state.messages,
  ]);

  return {
    messages: [response],
  };
}

// ---------------- Graph ----------------

const graph = new StateGraph(MessagesAnnotation)
  .addNode("agent", callModel)
  .addNode("tools", new ToolNode(tools))

  .addEdge(START, "agent")

  .addConditionalEdges(
    "agent",
    toolsCondition,
    ["tools", END]
  )

  .addEdge("tools", "agent")

  .compile();

// ---------------- Run ----------------

const result = await graph.invoke({
  messages: [
    {
      role: "user",
      content: "Add 3 and 4",
    },
  ],
});

console.log(result.messages.at(-1).content);