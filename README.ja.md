# ComfyUI-MonarchRT（日本語の概要）

[MonarchRT](https://github.com/Infini-AI-Lab/MonarchRT)（arXiv 2602.12271）を
ComfyUI から使うための非公式統合です。MonarchRT の著者・CMU・Self-Forcing /
CausVid の著者・Alibaba (Wan)・Comfy Org とは無関係です。

- **training-free**：公開されている dense な Self-Forcing DMD 重み
  （`gdhe17/Self-Forcing` の `self_forcing_dmd.pt`、Wan2.1-T2V-1.3B ベース）に、
  推論時だけ Monarch attention を適用します。論文の「学習済み MonarchRT 重み」は
  公開されておらず（upstream issue #1）、ここでの結果は論文の学習済み結果ではありません。
- **既定プロファイル `monarch_h2`**：`h_reduce=2`（実効スパース率 約90%、
  upstream issue #2 で training-free 向けとされた設定）。`monarch_h1` は upstream 設定の
  既定（約95%）、`dense` は同じ重み・prompt・noise・seed の比較用ベースラインです。
- **実行方式**：ComfyUI の Python には何もインストールしません。管理者が登録した
  別環境（WSL2 の Linux venv、または Linux の venv）で、commit 固定の upstream コードを
  無改変で import して実行し、本物の `VIDEO` 出力と JSON レポート
  （実際の attention dispatch 回数、フェーズ別時間、VRAM）を返します。
  WSL2 は「Windows ネイティブ」ではありません。
- **安全設計**：workflow から実行ファイルやコマンドは指定できません（runtime_id で
  管理者設定を選ぶだけ）。`shell=True` なし、環境変数は allowlist、キャンセル・
  タイムアウト時は自分のジョブのプロセスグループだけを停止します。サンドボックスではありません。

- **v0.2.0: persistent worker**：`backend: persistent` で、ComfyUI の別々の
  queue job 間でモデルと Triton の autotune 結果を保持します（実測: Monarch の
  2件目以降のジョブは約 10〜12 秒、one-shot は毎回約 195〜216 秒。出力は one-shot と
  バイト一致）。待機中も GPU 約 18 GB を保持するため、`worker_idle_seconds`
  （既定 300 秒）経過、Worker ノードの unload、ComfyUI の終了・異常終了で停止します。
  既定は従来どおり one-shot です。

速度・品質の比較結果、測定範囲、制約は [README.md](README.md) と
[docs/BENCHMARKS.md](docs/BENCHMARKS.md) を参照してください。16 FPS 達成や
dense と同等品質を主張するものではありません。

セットアップ：[docs/SETUP.md](docs/SETUP.md)（英語）
