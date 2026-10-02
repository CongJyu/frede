> **SUPERSEDED — do not use.** This prompt produced a run that was generated
> by a script rather than by the model. Use `prompts/MIMO_RERUN_PROMPT.md`
> instead, which states the prohibition explicitly and gives the agent a
> legitimate way to stop partway. Kept only as a record of what was asked.

# 任務：生成機器撰寫嘅評論（全跑 — 300 個 shard）

你係一個數據生成 worker。你要為每一條輸入記錄，**用你嘅模型即時生成**一篇評論，寫入輸出檔案。

你呢個 agent 就係嗰個模型 —— 評論文字由你直接生成，唔可以由外部工具或者人手代寫。

> **你之前已經做過 3 個 shard 嘅 pilot，結果 `USABLE`：30 條全部相異、改寫同原文重疊 0.47、保留原文本意 r=0.956。今次係同一個任務嘅完整版，標準一樣。**
>
> **唯一分別：輸入文字已經清理過。** 之前原文含字面 `\n`（反斜線+n）垃圾，你嗰時自動清理咗。而家原文係乾淨嘅真換行 —— 即係話，**原文有分段嘅話，你改寫都要保留分段**。呢點好重要，下面第四節第 7 條有講。

---

## 一、輸入

```
data/gen_packet_small/shards/shard_000.jsonl
data/gen_packet_small/shards/shard_001.jsonl
...
data/gen_packet_small/shards/shard_299.jsonl
```

**300 個檔案，每個 10 行**，每行一個 JSON 記錄：

```jsonc
{
  "custom_id": "p000042_machine_rewrite",
  "condition": "machine_rewrite",     // machine_rewrite | machine_generate | human_edit
  "stars": 4,
  "source_text": "原文…",              // machine_generate 呢個係 null
  "business": null,
  "prompt": "完整指示…"                // 每一行都唔同，係你唯一要跟嘅指示
}
```

## 二、每條記錄要做嘅事

1. 攞記錄嘅 `prompt` 欄位，**原句**作為 user message。
2. 攞下面嘅 system prompt，**原句**使用：

   > You are generating synthetic restaurant and local-business reviews for an academic dataset on machine-generated text detection. Follow the user's formatting instructions exactly. Never mention that you are an AI, never mention this task, and never add commentary, headings, or explanation outside the requested JSON.

3. 生成評論，將**評論文字**寫入輸出檔案。

**唔可以改動、縮短、總結或者「優化」`prompt` 欄位。** 唔可以加自己嘅開場白或者重述任務。

## 三、輸出

```
data/gen_results/mimo-v2.5-full/shard_000.jsonl
data/gen_results/mimo-v2.5-full/shard_001.jsonl
...
data/gen_results/mimo-v2.5-full/shard_299.jsonl
```

檔名同輸入一樣。每行一個 JSON object，**只有兩個 key**：

```json
{"custom_id": "p000042_machine_rewrite", "text": "評論全文"}
```

- `custom_id` 必須由輸入**逐字元複製**。唔可以改、唔可以重新編號、唔可以加後綴。
- `text` 必須係**純評論文字**。唔可以有嵌套 JSON、唔可以有 ` ``` ` 圍欄、唔可以有「Here is the review:」呢類前綴、唔可以有句號以外嘅解釋。
- **一行入，一行出。** 10 行入就 10 行出。冇失敗就唔應該少過 10 行。
- 唔可以加其他欄位。
- 如果某一條真係做唔到，**略過該行**。唔可以寫 placeholder、唔可以寫半截、唔可以寫錯誤訊息。

---

## 四、硬規則

**1. 每一條 `text` 都要為佢自己嗰條記錄即時生成。**
唔可以重複使用、稍作改動、或者喺卡住嘅時候用同一段文字代替。如果你發現自己開始重複，**停低並報告**，唔好夾硬做完 —— 重複嘅數據會令訓練出嚟嘅偵測器只認得嗰幾句。

**2. `machine_rewrite` 一定要真係讀原文。**
輸出必須保留原文嘅：**星級、整體好壞、具體事實**（食咗乜、服務點、價錢、等幾久、發生咩事）。唔可以寫一篇同原文無關嘅通用評論。原文係負評就要維持負評，係好評就要維持好評。

**3. 長度要夠。**

| condition | 要求長度 |
|---|---|
| `machine_rewrite` | 同原文相若（0.6–1.5 倍）|
| `machine_generate` | **60–160 字** |
| `human_edit` | 同原文一樣（±20% 以內）|

**4. `human_edit` 只可以改錯別字、文法、大小寫、標點。**
唔可以改寫、唔可以換詞、唔可以令佢更流暢專業、唔可以改句子順序。輸出應該同原文**近乎逐字一樣**。如果原文冇錯，就原封不動返還。

**5. 文字必須由模型生成，唔可以自己手寫。**
如果你自己作，量度嘅就係你嘅文風而唔係模型嘅，成批數據報廢。呢一條係最重要嘅一條。

**6. 唔可以有 emoji、星號評分、或者 meta 內容。**
唔可以出現 `⭐⭐⭐⭐⭐`、`As an AI`、`I cannot`、`I'm sorry, but I can't`。

**7. 保留原文嘅段落結構。**
原文可能有分段（空行、多段）。`machine_rewrite` 同 `human_edit` 都要**保留相同嘅分段**，唔可以壓縮成一大段，亦都唔可以加新段落。`machine_generate` 就自己決定一段定兩段。

**8. 超長原文（500 字以上）要特別小心。**
已知有記錄喺 500 字以上會喺中途被截斷。呢類要慢慢做，確保寫完整篇，唔好寫到一半就收。

---

## 五、做法

**一個 shard 一個 shard 咁做。** 每做完一個：

1. 數輸出檔嘅行數，應該等於輸入行數（減去你刻意略過嘅）。
2. 做夠 10 個 shard 之後，跑一次下面嘅檢查，確認仍然 `USABLE` 先繼續。
3. 輸出檔已存在而且完整，就跳過該 shard（可以隨時停低再續，唔會白做）。

**唯一輸出目錄係 `data/gen_results/mimo-v2.5-full/`。** 唔好寫入 `mimo-v2.5-opencode`（舊嘅失敗 run）、`mimo-pilot`（pilot）、或者任何其他目錄。

檢查指令：

```bash
.venv/bin/python -m scripts.check_pairs \
    --results data/gen_results/mimo-v2.5-full \
    --generator mimo-v2.5-full
```

中途跑都可以，佢會讀你目前為止寫咗嘅所有 shard。最後要報 verdict 同覆蓋率。

---

## 六、完成之後

報告：

1. 完成咗幾多個 shard，有冇略過嘅（同原因）。
2. `check_pairs` 嘅 verdict 同完整輸出。
3. 任何你覺得有問題或者含糊嘅地方。發現 packet 有問題係有價值嘅輸出，唔好掩蓋。
