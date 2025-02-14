/*******************************************************************
 * deepResearch.ts
 *
 * Firecrawlによる検索＆スクレイピング + LLMによる分析およびレポート生成を行うサンプルです。
 * このコードは、日本語で最終アウトプットを返し、内部のCoT（思考過程）は最終出力に含めません。
 *******************************************************************/

import FirecrawlApp, { SearchResponse } from "@mendable/firecrawl-js";
import { generateObject } from "ai";
import { compact } from "lodash-es";
import { z } from "zod";
import pLimit from "p-limit";

// Chatモデルの作成などは各自で用意してください
import { createModel, trimPrompt } from "./ai/providers";
import { systemPrompt } from "./prompt";

/**
 * リサーチ結果の型
 */
export type ResearchResult = {
  learnings: string[];
  visitedUrls: string[];
};

/**
 * deepResearch関数に渡すオプション
 */
export type DeepResearchOptions = {
  query: string;
  breadth?: number; // 1回のSERP生成で作るクエリ数
  depth?: number; // 将来的な段階リサーチ用（今回は未使用）
  learnings?: string[]; // 既に得られている知見があれば
  visitedUrls?: string[]; // 既に訪れたURLリストがあれば
  onProgress?: (update: string) => Promise<void>;
  model: ReturnType<typeof createModel>;
  firecrawlKey?: string;
};

/**
 * Firecrawl初期化
 */
const getFirecrawl = (apiKey?: string) =>
  new FirecrawlApp({
    apiKey: apiKey ?? process.env.FIRECRAWL_KEY ?? "",
  });

/**
 * 進捗ログの表示をまとめたヘルパ
 */
const formatProgress = {
  generating: (count: number, query: string) =>
    `検索クエリを最大 ${count} 件生成中: ${query}`,
  created: (count: number, queries: string) =>
    `生成された検索クエリ ${count} 件: ${queries}`,
  researching: (query: string) => `検索中: ${query}`,
  found: (count: number, query: string) =>
    `検索結果 ${count} 件を発見: ${query}`,
  ran: (query: string, count: number) =>
    `クエリ "${query}" を実行、${count} 件のコンテンツを取得`,
  generated: (count: number, query: string) =>
    `クエリ "${query}" により ${count} 件の学習ポイントを生成`,
};

/**
 * 進捗を外部に通知するヘルパ
 */
async function logProgress(
  message: string,
  onProgress?: (update: string) => Promise<void>
) {
  if (onProgress) {
    try {
      await onProgress(message);
    } catch (error) {
      // クライアントとの接続が切断されている等の場合、エラーをキャッチして無視する
      console.warn("進捗更新に失敗しました（接続切断の可能性）:", error);
    }
  }
}

/**
 * (1) ユーザープロンプトから検索キーワードを生成する
 */
async function generateSerpQueries({
  query,
  numQueries = 3,
  learnings,
  onProgress,
  model,
}: {
  query: string;
  numQueries?: number;
  learnings?: string[];
  onProgress?: (update: string) => Promise<void>;
  model: ReturnType<typeof createModel>;
}): Promise<string[]> {
  await logProgress(formatProgress.generating(numQueries, query), onProgress);

  const res = await generateObject({
    model,
    system: systemPrompt(),
    prompt: `
以下のユーザークエリに基づいて、追加調査に役立つ日本語の検索クエリを最大 ${numQueries} 件提案してください。  
ユーザークエリ: "${query}"
既存の知見: ${learnings?.join("\n") ?? "(なし)"}

提案する検索クエリとその理由を、以下のJSON形式で出力してください：
{
  "queries": [
    {
      "query": "検索クエリ1",
      "rationale": "このクエリを提案する理由"
    },
    ...
  ]
}
`,
    schema: z.object({
      queries: z.array(
        z.object({
          query: z.string(),
          rationale: z.string(),
        })
      ),
    }),
  });

  const queriesList = res.object.queries.map((q) => q.query).join(", ");
  await logProgress(
    formatProgress.created(res.object.queries.length, queriesList),
    onProgress
  );

  return res.object.queries.slice(0, numQueries).map((q) => q.query);
}

/**
 * (2) 検索結果をLLMにまとめ、学習ポイントを抽出する
 * ※内部の段階的思考は行いますが、最終出力には含めず、重要な学習ポイントのみ日本語で出力します。
 */
async function processSerpResult({
  userQuery,
  serpQuery,
  result,
  onProgress,
  model,
  numLearnings = 3,
}: {
  userQuery: string;
  serpQuery: string;
  result: SearchResponse;
  onProgress?: (update: string) => Promise<void>;
  model: ReturnType<typeof createModel>;
  numLearnings?: number;
}) {
  const markdowns = compact(result.data.map((item) => item.markdown)).map(
    (md) => trimPrompt(md, 25000)
  );

  await logProgress(
    formatProgress.ran(serpQuery, markdowns.length),
    onProgress
  );

  const res = await generateObject({
    model,
    system: systemPrompt(),
    prompt: `
ユーザーの質問: "${userQuery}"
使用したサブクエリ: "${serpQuery}"
以下は検索結果から抽出された一部のMarkdownテキストです（長い場合は一部を切り捨てています）：

---
${markdowns.map((m) => `### スニペット\n${m.slice(0, 3000)}\n---\n`).join("\n")}
---

上記の情報に基づき、以下の項目を日本語で出力してください。
1) 内部的な分析は行いますが、最終出力には含めないでください。
2) 最大 ${numLearnings} 件の重要な学習ポイント ("learnings") を抽出する。
3) 必要に応じて、今後の調査のための追加入力候補 ("followUpQuestions") も提案する。

以下のJSON形式で出力してください：
{
  "learnings": [
    "学習ポイント1",
    "学習ポイント2"
  ],
  "followUpQuestions": [
    "追加入力候補1",
    "追加入力候補2"
  ]
}
`,
    schema: z.object({
      learnings: z.array(z.string()),
      followUpQuestions: z.array(z.string()),
    }),
  });

  await logProgress(
    formatProgress.generated(res.object.learnings.length, serpQuery),
    onProgress
  );

  console.log("res.object", res.object);

  // 内部の思考過程は出力せず、学習ポイントと追加入力候補のみ返す
  return {
    learnings: res.object.learnings,
    followUpQuestions: res.object.followUpQuestions,
  };
}

/**
 * (3) レポートをMarkdown形式で生成する関数
 */
export async function writeFinalReport({
  prompt,
  learnings,
  visitedUrls,
  model,
}: {
  prompt: string;
  learnings: string[];
  visitedUrls: string[];
  model: ReturnType<typeof createModel>;
}): Promise<string> {
  const bulletPoints = learnings.map((x) => `- ${x}`).join("\n");

  const res = await generateObject({
    model,
    system: systemPrompt(),
    prompt: `
ユーザーの要求: 「${prompt}」
以下の学習ポイントを踏まえ、最終的なレポートを日本語でMarkdown形式で作成してください：
${bulletPoints}

レポートは以下のセクションを含むこと：
- 主要な発見
- 今後のアクション案
- 見通しや考察（必要な場合）

以下のJSON形式で出力してください：
{
  "reportMarkdown": "レポート全文（日本語）"
}
`,
    schema: z.object({
      reportMarkdown: z.string(),
    }),
  });

  const sourceSection = visitedUrls.length
    ? "\n\n## 参考URL\n" + visitedUrls.map((u) => `- ${u}`).join("\n")
    : "";

  return `# リサーチレポート

${res.object.reportMarkdown}${sourceSection}`;
}

/**
 * メインのリサーチ関数
 */
export async function deepResearch({
  query,
  breadth = 3,
  depth = 2,
  learnings = [],
  visitedUrls = [],
  onProgress,
  model,
  firecrawlKey,
}: DeepResearchOptions): Promise<ResearchResult> {
  try {
    const firecrawl = getFirecrawl(firecrawlKey);
    const limit = pLimit(1); // 並列実行数を制限

    const results: ResearchResult[] = [];

    await logProgress(formatProgress.generating(breadth, query), onProgress);
    const serpQueries = await generateSerpQueries({
      query,
      learnings,
      numQueries: breadth,
      onProgress,
      model,
    });
    await logProgress(
      formatProgress.created(serpQueries.length, serpQueries.join(", ")),
      onProgress
    );

    const tasks = serpQueries.map((serpQuery) =>
      limit(async () => {
        try {
          await logProgress(formatProgress.researching(serpQuery), onProgress);

          const searchResults = await firecrawl.search(serpQuery, {
            timeout: 60000,
            limit: 5,
          });

          const foundUrls = searchResults.data
            .map((r) => r.url)
            .filter((u): u is string => !!u);

          // 各URLをクロール
          for (const url of foundUrls) {
            try {
              await firecrawl.crawlUrl(url, {
                scrapeOptions: {
                  timeout: 60000,
                  formats: ["markdown"],
                },
              });
            } catch (err) {
              console.error("URLクロール中のエラー:", url, err);
            }
          }

          await logProgress(
            formatProgress.found(searchResults.data.length, serpQuery),
            onProgress
          );

          if (searchResults.data.length > 0) {
            await logProgress(
              formatProgress.ran(serpQuery, searchResults.data.length),
              onProgress
            );

            const analysis = await processSerpResult({
              userQuery: query,
              serpQuery,
              result: searchResults,
              onProgress,
              model,
            });

            await logProgress(
              formatProgress.generated(analysis.learnings.length, serpQuery),
              onProgress
            );

            return {
              learnings: analysis.learnings,
              visitedUrls: foundUrls,
            } as ResearchResult;
          } else {
            return { learnings: [], visitedUrls: [] };
          }
        } catch (err) {
          console.error(`エラー発生 (${serpQuery}):`, err);
          await logProgress(`エラー (${serpQuery}): ${err}`, onProgress);
          return { learnings: [], visitedUrls: [] };
        }
      })
    );

    const resolved = await Promise.all(tasks);
    results.push(...resolved);

    const aggregated: ResearchResult = {
      learnings: Array.from(new Set(results.flatMap((r) => r.learnings))),
      visitedUrls: Array.from(new Set(results.flatMap((r) => r.visitedUrls))),
    };

    return aggregated;
  } catch (error) {
    console.error("deepResearchで致命的なエラーが発生:", error);
    throw error;
  }
}
