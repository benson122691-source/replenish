# 05 SDS：細部設計

版本：0.1（2026-11-08）

## 1. 資料欄位（長表）

| 欄位 | 型別 | 說明 |
|---|---|---|
| id | str | `ITEM_STORE_evaluation`，序列識別 |
| item_id / dept_id / cat_id / store_id | str | 品項、部門、類別、門市 |
| date | datetime | 日期 |
| sales | float | 當日銷量（≥ 0） |
| sell_price | float | 當週售價 |
| event | int (0/1) | 當日是否有事件 |
| snap | int (0/1) | 當日是否為補助日 |
| wday / month / year | int | 日曆欄位 |

特徵（見 `features.py`）：lag_28/35/42/56、roll_mean_7、roll_mean_28、roll_std_28、sell_price、price_ratio、wday、month、dayofmonth、weekofyear、event、snap，以及 item/store/dept 類別編碼。

## 2. 補貨公式（FR-5）

符號：μ = 預測期間每日平均需求（取模型對未來 L 天的預測平均）；σ = 每日需求標準差（來源待確認：保留期殘差或滾動標準差，M2 決定）；L = 交期（天）；z = 服務水準對應的標準常態分位數。

```
補貨點 ROP      = μ·L + z·σ·√L
庫存位置 IP     = 現有庫存 + 在途量（MVP 在途量 = 0，待確認）
建議補貨量 Q    = 0                            若 IP > ROP
                 max(MOQ, ceil(ROP − IP))       若 IP ≤ ROP
```

模擬用的補充水位（每日盤點，R = 1）：order-up-to 水位 S = μ·(L+R) + z·σ·√(L+R)，當庫存位置 < S 時補貨 `max(MOQ, ceil(S − IP))`，到貨為 L 天後；缺貨為流失銷售（不補單）。σ = 各序列訓練期最後 56 天的每日銷量標準差，各策略共用。

成本（FR-6）：持有成本 = 單價 × 25%/365 × 每日期末庫存；缺貨成本 = 缺貨比例（預設 0.4）× 單價 × 缺貨量。兩者皆為**假設值（待確認）**，並以敏感度分析檢查結論是否依賴它們（比例 0.2 / 0.4 / 0.8 / 1.6）。

比較的策略：seasonal naive、28 天移動平均、LightGBM、oracle（完美預測，成本下界）。所有策略共用相同的起始庫存、σ、服務水準與成本，**差別只在預測**，因此基準不再使用「固定安全庫存」（原草稿，已修正）。

## 3. API 規格

### GET /forecast?item_id=&store_id=
回應 200：
```json
{"item_id": "FOODS_1_001", "store_id": "CA_1", "data_source": "synthetic",
 "forecast": [{"date": "2024-08-30", "pred": 3.2}]}
```
錯誤：404 `{"detail": "item_id not found"}`

### POST /replenish
請求：
```json
{"item_id": "FOODS_1_001", "store_id": "CA_1", "on_hand": 10,
 "lead_time_days": 7, "service_level": 0.95, "moq": 12}
```
回應 200：
```json
{"reorder_point": 28.4, "order_qty": 19, "mu_daily": 3.1, "sigma_daily": 1.8, "z": 1.645}
```
錯誤：422 輸入驗證（BR-4）；404 品項不存在（BR-1）。

### GET /compare
回應 200：
```json
{"data_source": "synthetic", "horizon_days": 28,
 "strategies": [{"name": "baseline_ma28", "total_cost": 0, "stockout_units": 0},
                {"name": "lightgbm", "total_cost": 0, "stockout_units": 0}]}
```
（範例數字為格式示意，非實測結果。）

## 4. 主要 UI 狀態

| 狀態 | 顯示 |
|---|---|
| 初始 | 品項、門市選單；無圖 |
| 載入中 | 圖表區顯示「載入中」 |
| 成功 | 預測折線圖、補貨建議卡片、成本比較表、資料來源標籤 |
| 輸入錯誤 | 欄位下方顯示錯誤原因，不送出請求 |
| 伺服器錯誤 | 顯示錯誤訊息與重試按鈕 |
