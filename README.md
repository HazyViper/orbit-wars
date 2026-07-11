# orbit-wars-agents

Kaggle Simulation コンペ **[Orbit Wars](https://www.kaggle.com/competitions/orbit-wars)** に4ヶ月参戦したときのエージェント群と、公式実装との一致をゴールデンテストで保証した自作シミュレータの公開リポジトリです。

解説記事（Zenn）: **ローカルで92%の検証精度のエージェントが実戦勝率1%だった — Claude CodeとKaggleに挑んだ4ヶ月**（リンクは公開後に追記）

> ⚠️ 本文・下表の勝率は特記ない限り**ローカル自己対戦**の値で、対戦相手プールに依存します。ローカル評価がいかに嘘をつくかは記事本文の主題です。

## エージェント一覧

記事の「幕」との対応つき。各エージェントは `def agent(obs)` を持つ単一ファイルで、そのまま Kaggle 提出可能な形式です。

| ファイル | 幕 | アプローチ | ローカル評価（当時） |
|---|---|---|---|
| `agents/v6_simplified.py` | 第1幕 | ルールベース初期版 | Kaggle 894.8（151/1610位） |
| `agents/v8h_combined.py` | 第1幕 | v6 の改良。長らく基準相手 | — |
| `agents/v14a_aggressive.py` | 第1幕 | Top10 統計のハードコード（結果論の罠の実例） | vs v8h 38% |
| `agents/v16_aggressive.py` | 第4幕 | 攻撃閾値 1.20→1.05 ほか小変更 | vs v8h 93% |
| `agents/v55_iterative_roi.py` | 第4幕 | 必要艦数の反復収束 + ROI スコアリング | 2P 65.4%（対前世代） |
| `agents/v57d_enemy500.py` | 第4幕 | 敵 ROI / 中立 ROI の重み分離 | 2P 82%（対v49） |
| `agents/v60b_solo_fix.py` | 第4幕 | スコア整合 + ソロチェック距離修正 | 2P ~62%（対前世代） |
| `agents/v100_dist25.py` | 第4幕 | 12パラメータグリッドサーチの生き残り（DIST_PENALTY=25） | 2P 55.4%（対前世代） |
| `agents/v105_gar50.py` | 第4幕 | 駐留ボーナス係数=50 | 2P 60%（対前世代） |
| `agents/v117_dp15.py` | 第4幕 | ヒューリスティック最終形（DIST_PENALTY=15） | 2P 55.3%（対v115）/ 4P 1位率 28.6% |
| `producer_v11.py` | 第5幕 | **The Producer V2** 移植 + 増援リスク項 | 2P 69.4%（対自作v10, z=7.70） |
| `producer_v13_rollout.py` | 第5幕 | value-free 深い MC rollout（深さ12-16） | vs v11 100% / vs v117 96.7%（ただし本番では…記事参照） |

記事の第2幕（模倣学習）・第3幕（PPO / LightZero）は学習パイプラインと重みが必要なため、本リポジトリには含めていません。

## ディレクトリ構成

```
agents/                  # ヒューリスティック系（stdlib のみ・各1ファイル）
producer_v11.py          # The Producer V2 ベースのフロープランナー（要 torch）
producer_v13_rollout.py  # v11 + 自作シミュレータで深読みする MC rollout（要 torch）
orbit_lite/              # The Producer V2 の PyTorch 移植コア（producer_* が使用）
simulator/               # 自作ゲームエンジン（stdlib のみ）+ テスト44件
```

## 動かし方

```bash
pip install -r requirements.txt   # torch / kaggle-environments / pytest
```

対戦を1試合回す:

```python
from kaggle_environments import make

env = make("orbit_wars")
env.run(["agents/v117_dp15.py", "agents/v16_aggressive.py"])
print(env.toJSON()["rewards"])
```

`producer_v13_rollout.py` を対戦させる場合はリポジトリのルートで実行してください（`orbit_lite/` と `simulator/` を自動検出します）。挙動は環境変数で調整できます（既定で `actTimeout=1s` に収まるよう自己校正します）:

- `EXP4_LA_DEPTH_2P=16` / `EXP4_LA_DEPTH_4P=12` — ロールアウト深さ上限（12 未満は非推奨）
- `EXP4_LA_BUDGET=0.80` — 1ターンの時間予算（秒）
- `EXP4_LA_PROD_W=16` — リーフ評価の生産力項（0 で v12 相当）

## シミュレータとゴールデンテスト

`simulator/` は kaggle_environments の公式実装を1ステップ単位で再現する自作エンジンです。`GameState.copy()` を deepcopy から手書き再構築に置き換えて高速化してあり（step 3.12ms→0.81ms）、`producer_v13_rollout.py` の深い先読みを 1 秒制限内に収める土台になっています。

```bash
python -m pytest simulator/ -q   # 44 tests
```

`test_golden.py` は公式実装（`kaggle-environments` パッケージ）と盤面が完全一致することを、彗星出現ステップ境界を含めて検証します。

## Attribution

- `orbit_lite/` と `producer_v11.py` は、slawekbiel 氏が Kaggle で公開したノートブック **[The Producer V2](https://www.kaggle.com/code/slawekbiel/the-producer-v2)** の設計（フロー差分スコアラ、増援リスク項）を PyTorch へ移植・改変したものです。素晴らしい設計を公開してくださった原作者に深く感謝します。
- `agents/v117_dp15.py` の Indirect Wealth ボーナスは、Halite 3 の公開ボット oddshrimp の `indWealth` のアイデアを参考にしています。
- `simulator/test_golden.py` は Kaggle 公式の [kaggle-environments](https://github.com/Kaggle/kaggle-environments)（Apache-2.0）を比較対象として利用します。

## License

Apache License 2.0（`LICENSE` / `NOTICE` を参照）。The Producer V2 由来の部分については、原ノートブックのライセンス条件が優先されます。
