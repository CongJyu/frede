# 任務：生成機器撰寫嘅評論（PILOT — 只做 3 個 shard）

你係一個數據生成 worker。你要為每一條輸入記錄，**用你嘅模型即時生成**一篇評論，寫入輸出檔案。

你呢個 agent 就係嗰個模型 —— 評論文字由你直接生成，唔可以由外部工具或者人手代寫。

---

## 一、輸入

```
data/gen_packet_pilot/shards/shard_000.jsonl
data/gen_packet_pilot/shards/shard_001.jsonl
data/gen_packet_pilot/shards/shard_002.jsonl
```

每個檔案 **10 行**，每行一個 JSON 記錄：

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
data/gen_results/mimo-pilot/shard_000.jsonl
data/gen_results/mimo-pilot/shard_001.jsonl
data/gen_results/mimo-pilot/shard_002.jsonl
```

檔名同輸入一樣。每行一個 JSON object，**只有兩個 key**：

```json
{"custom_id": "p000042_machine_rewrite", "text": "評論全文"}
```

- `custom_id` 必須由輸入**逐字元複製**。唔可以改、唔可以重新編號、唔可以加後綴。
- `text` 必須係**純評論文字**。唔可以有嵌套 JSON、唔可以有 ` ``` ` 圍欄、唔可以有「Here is the review:」呢類前綴、唔可以有句號以外嘅解釋。
- **一行入，一行出。** 10 行入就 10 行出。冇失敗就唔應該少過 10 行。
- 唔可以加其他欄位（唔要 `condition`、`stars`、`model` 之類）。
- 如果某一條真係做唔到，**略過該行**。唔可以寫 placeholder、唔可以寫半截、唔可以寫錯誤訊息。

---

## 四、硬規則 —— 呢啲係上次失敗嘅原因

上次有人交咗 3000 條記錄，檔案齊全、每個 shard 都「完成」，但數據完全冇用。以下每一條都係實際發生過嘅事，必須避免：

**1. 每一條 `text` 都要為佢自己嗰條記錄即時生成。**
上次 3000 條入面只有 1230 條係唔同嘅 —— 同一句開頭重複咗 295 次。呢種數據會令訓練出嚟嘅偵測器只認得嗰幾句，完全冇用。如果你發現自己開始重複，**停低並報告**，唔好夾硬做完。

**2. `machine_rewrite` 一定要真係讀原文。**
輸出必須保留原文嘅：**星級、整體好壞、具體事實**（食咗乜、服務點、價錢、等幾久、發生咩事）。唔可以寫一篇同原文無關嘅通用評論。上次同原文嘅詞彙重疊中位數只有 0.10（正常應該 0.4–0.8）。原文係負評就要維持負評，係好評就要維持好評。

**3. 長度要夠。**
上次輸出中位數只有 27 字，遠低於要求：

| condition | 要求長度 |
|---|---|
| `machine_rewrite` | 同原文相若（0.6–1.5 倍）|
| `machine_generate` | **60–160 字** |
| `human_edit` | 同原文一樣（±20% 以內）|

**4. `human_edit` 只可以改錯別字、文法、大小寫、標點。**
唔可以改寫、唔可以換詞、唔可以令佢更流暢專業、唔可以改句子順序。輸出應該同原文**近乎逐字一樣**。如果原文冇錯，就原封不動返還。

**5. 文字必須由模型生成，唔可以自己手寫。**
如果你自己作，量度嘅就係你嘅文風而唔係模型嘅，成批數據報廢。

**6. 唔可以有 emoji、星號評分、或者 meta 內容。**
唔可以出現 `⭐⭐⭐⭐⭐`、`As an AI`、`I cannot`、`I'm sorry, but I can't`。

---

## 五、完成之後

**只做呢 3 個 shard。** 唔好自己擴展去 `data/gen_packet_small/` 或者任何其他 packet。

然後執行：

```bash
.venv/bin/python -m scripts.check_pairs \
    --results data/gen_results/mimo-pilot \
    --generator mimo-pilot
```

將完整輸出報告返俾我。如果 verdict 係 `NOT USABLE`，**停低唔好繼續**，並如實報告原因 —— 早啲發現問題比勉強交貨有價值。
