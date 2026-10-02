# 任務：生成機器撰寫嘅評論（重跑 — 300 個 shard）

你係一個數據生成 worker。你要為每一條輸入記錄，**用你自己（你嘅模型）即時生成**一篇評論，寫入輸出檔案。

---

## 零、先讀呢一節：上次係點失敗嘅

上一次跑呢個任務，交出嚟嘅 3000 條**唔係模型生成嘅**。有人寫咗一個 Python 腳本，用模板規則同同義詞替換拼出評論。結果：

- 逐條檢查**全部通過**（長度正常、星級正確、冇重複記錄）
- 但 3000 條裡面，**55.7% 嘅 5-gram 短語係跨記錄重複使用**嘅 —— 同一句 `"the moment we walked in"` 出現咗 35 次
- 真正嘅模型輸出呢個數字係 **4.3%**

**所以：**

> ### 🚫 唔准寫任何腳本、程式、模板或者規則去生成評論文字。
>
> 唔准用 Python、shell、任何自動化工具產生 `text` 欄位嘅內容。
> 每一條評論都必須由你（模型）即時逐條寫出嚟。
>
> 如果你覺得 3000 條太多做唔完 —— **如實報告你做完幾多，然後停低**。
> 部分完成係可以接受嘅。用腳本假裝完成係唔可以接受嘅，而且**一定會被驗出嚟**。

（如果你需要寫程式去**讀取／寫入檔案**，咁冇問題。禁令只針對「用程式產生評論文字」。）

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

**唔可以改動、縮短、總結或者「優化」`prompt` 欄位。**

## 三、輸出

```
data/gen_results/mimo-v2.5-rerun/shard_000.jsonl
...
data/gen_results/mimo-v2.5-rerun/shard_299.jsonl
```

檔名同輸入一樣。每行一個 JSON object，**只有兩個 key**：

```json
{"custom_id": "p000042_machine_rewrite", "text": "評論全文"}
```

- `custom_id` 必須由輸入**逐字元複製**。
- `text` 必須係**純評論文字**：唔可以有嵌套 JSON、唔可以有 ` ``` ` 圍欄、唔可以有前綴或解釋。
- **一行入，一行出。** 10 行入就 10 行出。
- 做唔到嘅就**略過該行**，唔好寫 placeholder。

**唯一輸出目錄係 `data/gen_results/mimo-v2.5-rerun/`。** 唔好寫入其他任何目錄。

---

## 四、文字質素要求

**1. 每一條都要獨立生成。** 唔可以喺唔同記錄之間重用句式、開頭、或者招牌短語。如果你發現自己反覆用同一句，**停低並報告**。

**2. `machine_rewrite` 係重寫，唔係換詞。**
將 `restaurant` 換成 `venue`、`food` 換成 `cuisine`，句子結構原封不動 —— 呢個唔係重寫，會被自動拒絕。要用你自己嘅講法**重新組織每個句子**：意思要保留，措辭唔可以保留。唔可以將詞語黏埋（例如 `placeis`），唔可以製造錯別字。

**3. `machine_rewrite` 一定要真係讀原文。** 保留星級、整體好壞、同具體事實（食咗乜、服務、價錢、等幾久）。原文係負評就要維持負評。

**4. 長度。**

| condition | 要求 |
|---|---|
| `machine_rewrite` | 同原文相若（0.6–1.5 倍）|
| `machine_generate` | **60–160 字** |
| `human_edit` | 同原文一樣（±20% 以內）|

**5. `human_edit` 只改錯別字、文法、大小寫、標點。** 唔可以改寫、唔可以換詞、唔可以令佢更流暢。原文冇錯就原封不動返還。

**6. 保留原文段落結構。** 原文有分段就要保留相同分段。500 字以上嘅原文要特別小心寫完整，唔好中途截斷。

**7. 唔可以有 emoji、星號評分、或者 `As an AI` 呢類 meta 內容。**

---

## 五、做法

**一個 shard 一個 shard 咁做。** 每做完一個：

1. 數輸出檔行數，應該等於輸入行數（減去刻意略過嘅）。
2. 做夠 **10 個 shard** 之後，跑一次下面嘅檢查，確認 `USABLE` 先繼續。
3. 輸出檔已存在而且完整就跳過（可以隨時停低再續）。

檢查指令：

```bash
.venv/bin/python -m scripts.check_pairs \
    --results data/gen_results/mimo-v2.5-rerun \
    --generator mimo-v2.5-rerun
```

呢個檢查會驗：模板重用率（跨記錄短語重複）、改寫係咪真係改寫、有冇保留原文本意、有冇截斷。中途跑都得。

---

## 六、完成之後

報告：

1. 完成咗幾多個 shard，有冇略過（同原因）。
2. `check_pairs` 嘅完整輸出同 verdict。
3. 任何你覺得含糊或者做唔到嘅地方 —— 如實講，唔好掩蓋。
