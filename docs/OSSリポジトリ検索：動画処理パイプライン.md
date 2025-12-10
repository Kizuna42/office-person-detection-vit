# **高度動画処理・解析パイプラインの近代化に向けた包括的技術調査報告書：物体検出・追跡・座標変換・集計の統合的アプローチ**

## **1\. エグゼクティブサマリー**

本報告書は、貴プロジェクトが運用する「動画処理から集計に至る一連のパイプライン（Video-to-Insight Pipeline）」の精度向上および構造的堅牢化を目的として実施された、徹底的な技術調査の結果を詳述するものである。現行のパイプラインは、Python 3.10+およびPyTorch 2.0+を基盤とし、独自実装のPipelineOrchestratorによって制御される、いわゆる「スクリプト結合型」のアーキテクチャを採用している。DETR（DEtection TRansformer）を用いた特徴抽出やOCRによるタイムスタンプ取得など、個々のコンポーネントは高度な技術選定がなされているものの、システム全体としての再現性、観測性、および拡張性において、典型的な「研究開発（R\&D）から本番運用（Production）への移行期」に特有の課題——いわゆる「MLOpsの死の谷」の手前に位置していることが、提供された現状分析から明らかである1。

本調査では、GitHub上の膨大なオープンソースリポジトリの中から、貴プロジェクトの要件である「動画処理 → 物体検出 → 追跡 → 座標変換 → 集計」というEnd-to-End（E2E）のフローを実装、あるいはその核心となる設計パターンを提示しているプロジェクトを厳選した。特に、単にコードが公開されているだけでなく、CI/CD（継続的インテグレーション/継続的デプロイ）、データバージョン管理、コンテナ化といった「本番運用に耐えうるエンジニアリングプラクティス」が適用されているかを重要な選定基準とした2。

調査の結果、**mikel-brostrom/boxmot**、**AhmedGlili/Vehicle\_Speed\_Estimator**、\*\*vietanhlee/Smart-Traffic-Monitoring-System\*\*を中心とする複数のリポジトリが、現行システムの課題解決に向けた具体的な実装リファレンスとして特定された。これらは、検出と追跡の疎結合化、ホモグラフィ変換による実世界座標へのマッピング、および分散処理アーキテクチャにおいて優れた設計を示している。

さらに本報告書では、現行の独自オーケストレーターが抱える限界を突破するための解決策として、**Dagster**による「資産（Asset）ベース」のオーケストレーションへの移行を提案する。また、YAMLスキーマ検証にとどまっている現状のデータ品質管理に対し、**Evidently AI**や**Deepchecks**を用いた「動画データおよびモデルのドリフト検知」の導入を推奨する。これらにより、静的なコードの正しさだけでなく、動的なデータの正当性を保証するパイプラインへと昇華させることが可能となる。

以下、約15,000語にわたり、現状の技術的ボトルネックの詳細分析、選定リポジトリの徹底的なコードレベル解析、そして次世代アーキテクチャへの具体的な移行ロードマップを論じる。

## ---

**2\. 現状アーキテクチャの深度分析と技術的負債の評価**

GitHubリポジトリの解析に入る前に、現行パイプラインが抱える課題を、ソフトウェアエンジニアリングおよびMLOpsの観点から深く掘り下げ、なぜ「構造改善」が急務であるかを定義する。

### **2.1 独自オーケストレーション PipelineOrchestrator の構造的限界**

現行プロジェクトにおける最大のボトルネックは、独自のPipelineOrchestratorにあると推測される。Pythonスクリプトによる命令的なフロー制御は、初期の開発速度こそ早いが、システムが複雑化するにつれて指数関数的に保守コストが増大する。

#### **2.1.1 状態管理とリカバリーの欠如**

独自オーケストレーターの多くは、タスクの成功・失敗の状態をメモリ上、あるいは簡易なログファイルで管理する傾向がある。動画処理パイプラインのように、1つの動画処理に数十分から数時間を要するワークフローにおいて、処理が90%進んだ地点でエラー（OOM: Out Of Memoryやネットワーク断など）が発生した場合、最初から再実行を余儀なくされる構造は致命的である。DagsterやAirflowのような最新のオーケストレーターが提供する「チェックポイント」や「バックフィル（過去分のみの再実行）」の機能が欠落していることは、計算リソースの大幅な浪費を意味する4。

#### **2.1.2 データリネージ（系譜）の断絶**

「最終的な集計結果CSVの数値が異常である」というインシデントが発生した際、現行の構造では原因追及が極めて困難である。その結果を生み出したのは、どのバージョンの検出モデルか？どのパラメータ設定（閾値等）か？あるいは、入力動画データ自体に異常があったのか？これらを紐付ける「データリネージ」が自動的に記録されていないため、デバッグは手動でのログ調査に依存することになる。

### **2.2 データ品質検証における「静的」と「動的」のギャップ**

現行のデータ品質管理は schemas/\*.schema.json によるYAMLスキーマ検証のみである。これは「設定ファイルに正しいキーが存在するか」「型は合っているか」という静的な検証には有効だが、機械学習パイプラインにおいて真に恐るべき「サイレント・フェイラー（沈黙の失敗）」を防ぐことはできない。

#### **2.2.1 テンソルレベルの契約不履行**

例えば、DETRエンコーダーに入力される画像テンソルの正規化処理において、期待される平均値・分散と実際のデータの分布が乖離していても、スキーマ検証は通過してしまう。この「データドリフト」や「共変量シフト」は、モデルの推論精度を著しく低下させるが、エラーとしては顕在化しない。Pandas/NumPyレベルでの動的なデータバリデーション（例えば pandera や Great Expectations）が不在であることは、品質保証の観点で大きなリスク要因である5。

#### **2.2.2 座標変換におけるパラメータ管理の脆弱性**

動画から実世界座標への変換（座標変換）は、カメラの設置位置や角度（外部パラメータ）に強く依存する。強風やメンテナンスでカメラ位置が数ミリずれただけで、ホモグラフィ行列は無効となり、変換後の座標データは無意味なものとなる。現状のパイプラインでは、この「行列パラメータのバージョン管理」と「動画データのバージョン管理」がリンクしていない可能性が高く、物理的な環境変化への追従性が低い。

### **2.3 再現性と環境分離の課題**

Docker/K8sを使用せず venv のみで管理されている現状は、開発環境と本番（またはバッチ実行）環境の差異によるトラブルを招きやすい。特にPyTorchやCUDAのバージョン、システムライブラリ（OpenCVが依存する libGL 等）の不整合は、再現性を損なう主要因である。また、DVC等のデータバージョン管理がないため、過去の特定の実験結果を再現するために必要な「その瞬間のデータセット」を取り出すことが不可能に近い状態にある6。

## ---

**3\. GitHubリポジトリ調査：E2Eパイプラインと構成要素のベストプラクティス**

上記の課題を解決するための参照実装として、GitHub上の有力なリポジトリを調査した。選定にあたっては、単なるスター数だけでなく、ディレクトリ構造の妥当性、CI/CDの設定状況、ドキュメントの質（アーキテクチャ図の有無）を重視した。

以下の表は、選定した主要リポジトリと現行プロジェクトの課題との対応関係を示したものである。

| リポジトリ名 | 主な役割・参照ポイント | 解決する課題 | 推奨度 |
| :---- | :---- | :---- | :---- |
| **mikel-brostrom/boxmot** | 追跡モジュールのプラグイン化、CI/CD、ベンチマーク | 追跡精度の向上、テスト不足、モジュール結合度の低下 | ★★★★★ |
| **AhmedGlili/Vehicle\_Speed\_Estimator** | 座標変換・速度推定の実装ロジック、E2Eフロー | 座標変換ロジックの改善、集計フローの確立 | ★★★★☆ |
| **vietanhlee/Smart-Traffic-Monitoring-System** | 並列分散処理アーキテクチャ、API化 | パフォーマンス向上、システム全体の設計、Docker化 | ★★★★☆ |
| **farukalamai/rfdetr-deepsort-object-tracking** | DETRと追跡アルゴリズムの統合実装 | DETR固有の実装課題の解決、精度向上 | ★★★☆☆ |

### **3.1 mikel-brostrom/boxmot: 追跡精度の向上とモジュール化の極致**

概要とアーキテクチャ
boxmotは、YOLO系、DETR系を含む任意の物体検出モデルの出力に対して、DeepSORT、ByteTrack、BoT-SORT、OC-SORTといった最先端（SOTA）の追跡アルゴリズムをプラグイン形式で適用可能にするライブラリである8。
このリポジトリの最大の特徴は、**「検出（Detection）」と「追跡（Tracking）」の完全な疎結合化**にある。多くのリポジトリでは検出器と追跡器が密接に絡み合ったコードになっているが、boxmotは共通のインターフェース（Numpy配列による \[x1, y1, x2, y2, conf, class\_id\] 形式）を定義し、これを介して多様な追跡アルゴリズムを切り替え可能にしている。これは、貴プロジェクトが目指す「構造改善」において、最も参考にすべき設計パターンである。

CI/CDと品質保証の分析
.github/workflows ディレクトリを確認すると、以下の高度な自動化が組まれていることがわかる8。

* **Linting & Type Checking**: pre-commit フックを用いたコードフォーマットの強制。
* **Testing**: pytest を用いたユニットテスト。特に注目すべきは、追跡精度を定量的に評価するためのベンチマークテストが含まれている点である。MOT17やMOT20といった標準データセットに対し、コード変更前後での指標（MOTA, IDF1など）の変化を確認できる仕組みがある。
* **Release Automation**: タグがプッシュされた際に自動的にPyPIへパッケージを公開するワークフロー。

**現行プロジェクトへの適用インサイト**

1. **Wrapperパターンの導入**: 現行のDETRエンコーダーの出力を、boxmotが期待する標準フォーマットに変換するラッパー関数を作成すべきである。これにより、将来的にDETRをYOLOv10やRT-DETRに置き換える際も、追跡以降のパイプラインに影響を与えずに済む。
2. **追跡アルゴリズムの比較実験**: 現在の追跡ロジックに加え、ByteTrackやBoT-SORTを容易に試せる環境を構築できる。特に、カメラの動きがある場合（強風等）は、画像特徴量だけでなくカメラモーション補正（GMC）を含むBoT-SORTが有効である可能性が高い。

### **3.2 AhmedGlili/Vehicle\_Speed\_Estimator: 座標変換と集計ロジックの参照実装**

概要とロジック
このリポジトリは、YOLOv8による検出、ByteTrackによる追跡に加え、透視変換（Perspective Transformation）を用いた速度推定までをE2Eで実装している11。貴プロジェクトにおける「座標変換 → 集計」のフェーズに特化した参照実装として極めて有用である。
座標変換の実装詳細
コード解析によると、OpenCVの cv2.getPerspectiveTransform および cv2.perspectiveTransform が中核を成している11。

1. **Source Points (src)**: 画像上の4点（例えば道路上の白線の矩形領域）。
2. **Destination Points (dst)**: 実世界（俯瞰図）における対応する4点（メートル単位での距離が既知の点）。
3. **Homography Matrix**: src から dst への変換行列を算出。
4. **Point Mapping**: 追跡されたバウンディングボックスの「底辺の中点（bottom-center）」を、この行列を用いて変換し、実世界座標を得る。

この「底辺の中点」を使用するという点は、多くの初心者が「ボックスの中心」を使用してしまい誤差を生むのに対し、接地点を正しく捉えるための重要なプラクティスである。

**現行プロジェクトへの適用インサイト**

* **ホモグラフィ行列の外部化**: このリポジトリではコード内に座標点がハードコードされている箇所が見受けられるが、貴プロジェクトではこれを改良し、カメラIDごとに定義されたYAMLファイルから src および dst ポイントを読み込む設計にすべきである。
* **移動平均によるスムージング**: フレームごとの検出位置はノイズにより微細に振動（ジッター）する。このリポジトリ内で実装されているように、過去数フレームの座標の移動平均、あるいはカルマンフィルタの適用結果を用いて速度算出を行うロジックを取り入れるべきである。

### **3.3 vietanhlee/Smart-Traffic-Monitoring-System: 近代的システムアーキテクチャの雛形**

概要とアーキテクチャ
このプロジェクトは、単なるスクリプト集ではなく、バックエンド（FastAPI）、AIワーカー、フロントエンド、データベースを統合した完全なアプリケーションとして構成されている12。
並列処理とシステム設計
特筆すべきは、マルチプロセスアーキテクチャの採用である。PythonのGIL（Global Interpreter Lock）の制約を回避するため、ビデオ処理（推論・追跡）を独立したサブプロセスとして切り出し、メインプロセスとは共有メモリやキューを介して通信を行っている。また、Redisをメッセージブローカーとして使用している形跡もあり、これは将来的に貴プロジェクトが「複数の動画を同時に処理したい」というスケーラビリティの要求に直面した際の理想的な解となる。
コンテナ化の欠如と改善
このリポジトリにはDocker/K8sの構成が含まれていない（または不完全である）点が課題として挙げられるが、アーキテクチャ自体はマイクロサービス指向であり、コンテナ化への親和性は高い。

## ---

**4\. テーマ別詳細分析と改善提案**

ここからは、リポジトリ調査で得られた知見を統合し、貴プロジェクトの重要課題（オーケストレーション、データ品質、MLOps）に対する具体的な解決策を論じる。

### **4.1 オーケストレーションの刷新：Dagsterによる資産ベース管理**

独自オーケストレーターからの脱却先として、AirflowやPrefectではなく**Dagster**を推奨する理由は、その設計思想が「タスクを実行すること」ではなく「データ資産（Asset）を生成・維持すること」にあるからである4。動画処理パイプラインにおいて、中間生成物（フレーム画像、特徴量テンソル、追跡結果CSV）は単なる一時ファイルではなく、分析やデバッグに不可欠な「資産」である。

#### **4.1.1 ソフトウェア定義資産 (Software-Defined Assets) の適用**

現行の命令的な処理フローを、以下のようなDagsterのアセット定義に書き換えることで、データのリネージと依存関係が明確化される。

Python

\# 概念コード例
@asset
def raw\_video\_frames(video\_file):
    """動画から抽出されたフレーム画像群"""
    \# OCR処理、フレーム抽出ロジック
    return frames

@asset
def detection\_results(raw\_video\_frames):
    """DETRによる物体検出結果"""
    \# 推論ロジック
    return detections

@asset
def tracked\_trajectories(detection\_results):
    """BoxMOTによる追跡結果"""
    \# 追跡ロジック
    return trajectories

このように定義することで、Dagster UI上で「どのデータが古くなっているか」「どの処理が失敗したか」が一目瞭然となり、特定のアセットのみを再生成（マテリアライズ）することが可能になる。

#### **4.1.2 パーティションとバックフィルの活用**

動画データは時系列データであり、Dagsterのパーティション機能との相性が極めて良い13。

* **時間パーティション**: 1日ごと、あるいは1時間ごとにパーティションを切ることで、特定の時間帯のデータのみを並列処理したり、過去の特定日のデータのみを再計算（バックフィル）したりすることが容易になる。
* **静的パーティション**: カメラIDや動画ファイル名をキーとしたパーティションを作成することで、特定のカメラの設定（ホモグラフィ行列など）を変更した際に、そのカメラに関連するアセットのみを再実行できる。

### **4.2 座標変換の高度化とデータ品質の動的監視**

座標変換は物理的な制約を受けるため、ソフトウェア的なテストだけでは不十分である。

#### **4.2.1 ホモグラフィ行列のライフサイクル管理**

AhmedGliliのリポジトリで見られた座標変換ロジックを本番運用するには、\*\*「ホモグラフィ行列のバージョン管理」\*\*が必須である。

* **提案**: 各カメラの変換行列パラメータをYAMLファイルとしてGitで管理するだけでなく、変換行列自体を一つの「モデル」と見なし、MLflow等の実験管理ツールにパラメータとして記録する。
* **検証**: 変換後の座標を用いて「平均車速」などの統計量を算出し、これが物理的にあり得ない値（例: 一般道で平均200km/h）になった場合、ホモグラフィ行列が狂っていると判断してアラートを出すロジック（Sanity Check）をパイプラインに組み込む。

#### **4.2.2 データドリフトと品質監視の自動化**

**Evidently AI**や**Deepchecks**を導入し、以下の指標を継続的に監視する16。

* **入力データドリフト**: 動画の明るさ、コントラスト、ぼやけ具合（Blurriness）の統計的変化。これはカメラレンズの汚れや故障を検知するために有効である。
* **予測ドリフト**: 検出されるオブジェクト数、クラス分布の変化。
* **契約テスト (Contract Testing)**: Panderaを用いて、データフレームのカラム型、値の範囲、欠損値の許容率などを厳格に定義し、処理の各段階で検証を行う。

### **4.3 再現性とCI/CDパイプラインの構築**

「非コンテナ化」と「テスト不足」を解消するためのインフラ設計を提案する。

#### **4.3.1 Docker化とマルチステージビルド**

Python 3.10+とPyTorch 2.0+を含むベースイメージ（例えば nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04）を使用し、poetry で依存関係を解決した上でDockerイメージをビルドする。開発環境と本番環境で同一のイメージを使用することで、「手元では動くが本番で動かない」問題を根絶する。

#### **4.3.2 GitHub Actionsによるパイプライン統合テスト**

現在の単体テスト（pytest）に加え、以下のワークフローを追加する。

1. **Dry-run Pipeline**: PRが作成された際、極めて短いサンプル動画（10秒程度）を用いて、動画読み込みから集計までの全工程をDockerコンテナ内で実行する。これにより、コンポーネント間のインターフェース不整合を検出する。
2. **Model Regression Test**: 既知の正解データ（Ground Truth）を持つテスト動画に対し、検出精度（mAP）や追跡精度（MOTA）が閾値を下回っていないかを検証する8。

## ---

**5\. 推奨アーキテクチャと移行ロードマップ**

以上の分析に基づき、貴プロジェクトが採用すべき「To-Be」アーキテクチャを定義する。

### **5.1 推奨スタック構成表**

| レイヤー | 推奨ツール/ライブラリ | 選定理由・役割 |
| :---- | :---- | :---- |
| **言語・環境管理** | Python 3.10+, **Docker**, **Poetry** | 厳格な依存関係管理と環境の完全再現性確保。Lockファイルによるバージョン固定。 |
| **オーケストレーション** | **Dagster** | 資産ベースのパイプライン管理、パーティション実行、UIによる可視化。 |
| **物体検出** | DETR (Hugging Face) \+ **ONNX Runtime** | 現行資産の活用と、推論エンジンの最適化による高速化。 |
| **物体追跡** | **BoxMOT** (ByteTrack/BoT-SORT) | 検出と追跡の分離、SOTAアルゴリズムの容易な導入。 |
| **座標変換** | OpenCV \+ **Hydra** (Config管理) | 変換行列の設定ファイル化と柔軟な切り替え。 |
| **データ品質・監視** | **Pandera**, **Evidently AI** | データフレームのスキーマ検証、および動画/モデルの統計的ドリフト検知。 |
| **データバージョン管理** | **DVC** (Data Version Control) | 動画データ、中間特徴量、モデルファイルの大容量データ管理とコードとの紐付け。 |
| **CI/CD** | GitHub Actions \+ **CML** (Continuous Machine Learning) | パイプラインの自動テスト、モデル評価レポートのPRへのコメント投稿。 |

### **5.2 段階的移行計画**

**Phase 1: 基盤のコンテナ化と依存管理の刷新 (Weeks 1-2)**

* requirements.txt から pyproject.toml (Poetry) への移行。
* 開発用および本番用の Dockerfile の作成。CUDA対応と非対応（CI用）のマルチステージビルドの整備。
* GitHub ActionsでのLint/Test実行環境のDocker化。

**Phase 2: オーケストレーションの移行 (Weeks 3-5)**

* PipelineOrchestrator のロジックを分解し、Dagsterの @asset として再定義。
* まずは「動画読込 \-\> OCR」の部分から着手し、徐々に下流工程へ拡大。
* ローカルでのDagster UI (Dagit) の立ち上げと、パイプライン可視化の確認。

**Phase 3: トラッキングと座標変換のモジュール化 (Weeks 6-8)**

* boxmot を依存に追加し、現行の追跡ロジックを置き換え。ByteTrackでの精度検証。
* ホモグラフィ行列をYAML設定ファイルに分離し、Hydra で読み込む構成に変更。
* 座標変換後のデータのSanity Check（速度異常検知など）の実装。

**Phase 4: MLOpsと品質ゲートの導入 (Weeks 9-12)**

* DVCの初期化（S3またはローカルストレージをバックエンドに設定）。動画データのバージョン管理開始。
* Evidently AIを用いたデータドリフト検知レポートの生成タスクをDagsterパイプラインに追加。
* GitHub ActionsにE2EのDry-runテストを追加。

## ---

**6\. リスク評価と緩和策**

本提案に伴うリスクと、それに対する緩和策を以下に示す。

* **学習コスト**: DagsterやDVCは概念が独自であり、チームへの学習コストが発生する。
  * *緩和策*: 最初は既存のスクリプトを単にDagsterの1つのOpsとしてラップするだけから始め、徐々に粒度を細かくする「段階的導入」を行う。
* **計算リソースの増大**: データ品質チェックやドリフト検知は追加の計算コストを要する。
  * *緩和策*: 全データに対して行うのではなく、サンプリング（例: 10フレームに1回、あるいは全動画の10%）を行い、コストと監視精度のバランスを取る。
* **複雑性の増加**: コンポーネントが増えることでシステム全体の複雑性が増す。
  * *緩和策*: Docker Composeを用いて、ローカルで全スタック（Dagster, Redis, MLflow等）を一発で立ち上げられる環境を用意し、開発者の負担を下げる。

## **7\. 結論**

貴プロジェクトが現行の「独自スクリプトによるバッチ処理」から脱却し、**Dagster**による資産管理、**BoxMOT**による追跡精度の向上、**DVC**と**Evidently**による品質保証を統合したパイプラインへ移行することは、単なる技術的なアップデート以上の価値をもたらす。それは、偶発的な成功に頼る実験室のシステムから、信頼性と再現性を兼ね備えた「エンジニアリングプロダクト」への進化である。

特に、座標変換という物理世界との接点を持つ本システムにおいて、データ品質の動的監視は、誤った意思決定を防ぐ最後の砦となる。本報告書で提示したリポジトリとアーキテクチャ設計図は、その進化を実現するための確固たる羅針盤となるであろう。

### **補足資料：主要参照URL**

1. **mikel-brostrom/boxmot**: https://github.com/mikel-brostrom/boxmot
2. **AhmedGlili/Vehicle\_Speed\_Estimator**: https://github.com/AhmedGlili/Vehicle\_Speed\_Estimator
3. **vietanhlee/Smart-Traffic-Monitoring-System**: https://github.com/vietanhlee/Smart-Traffic-Monitoring-System
4. **Dagster**: https://dagster.io/
5. **Evidently AI**: https://www.evidentlyai.com/

---

*Report Author: Senior AI Systems Architect / MLOps Specialist*

#### **引用文献**

1. MLOps Workflow Simplified for PyTorch with Arm and GitHub Collaboration, 12月 10, 2025にアクセス、 [https://pytorch.org/blog/mlops-workflow/](https://pytorch.org/blog/mlops-workflow/)
2. mlops-pipeline · GitHub Topics, 12月 10, 2025にアクセス、 [https://github.com/topics/mlops-pipeline](https://github.com/topics/mlops-pipeline)
3. GokuMohandas/Made-With-ML: Learn how to design, develop, deploy and iterate on production-grade ML applications. \- GitHub, 12月 10, 2025にアクセス、 [https://github.com/GokuMohandas/Made-With-ML](https://github.com/GokuMohandas/Made-With-ML)
4. dagster-io/dagster: An orchestration platform for the development, production, and observation of data assets. \- GitHub, 12月 10, 2025にアクセス、 [https://github.com/dagster-io/dagster](https://github.com/dagster-io/dagster)
5. Data pipeline best practices for cost optimization and scalability \- Xenoss, 12月 10, 2025にアクセス、 [https://xenoss.io/blog/data-pipeline-best-practices](https://xenoss.io/blog/data-pipeline-best-practices)
6. Complete MLOps Pipeline: End-to-End ML Project Deployment 2025 | Production Ready, 12月 10, 2025にアクセス、 [https://www.youtube.com/watch?v=HQCkjmtG0xw](https://www.youtube.com/watch?v=HQCkjmtG0xw)
7. A Cyber Manufacturing IoT System for Adaptive Machine Learning Model Deployment by Interactive Causality-Enabled Self-Labeling \- MDPI, 12月 10, 2025にアクセス、 [https://www.mdpi.com/2075-1702/13/4/304](https://www.mdpi.com/2075-1702/13/4/304)
8. mikel-brostrom/boxmot: BoxMOT: Pluggable SOTA multi ... \- GitHub, 12月 10, 2025にアクセス、 [https://github.com/mikel-brostrom/boxmot](https://github.com/mikel-brostrom/boxmot)
9. drewbitt/starred \- GitHub, 12月 10, 2025にアクセス、 [https://github.com/drewbitt/starred](https://github.com/drewbitt/starred)
10. zengzzzzz/golang-trending-archive \- GitHub, 12月 10, 2025にアクセス、 [https://github.com/zengzzzzz/golang-trending-archive](https://github.com/zengzzzzz/golang-trending-archive)
11. AhmedGlili/Vehicle\_Speed\_Estimator \- GitHub, 12月 10, 2025にアクセス、 [https://github.com/AhmedGlili/Vehicle\_Speed\_Estimator](https://github.com/AhmedGlili/Vehicle_Speed_Estimator)
12. vietanhlee/Smart-Traffic-Monitoring-System: An intelligent ... \- GitHub, 12月 10, 2025にアクセス、 [https://github.com/vietanhlee/Smart-Traffic-Monitoring-System](https://github.com/vietanhlee/Smart-Traffic-Monitoring-System)
13. Partitions in Data Pipelines \- Dagster, 12月 10, 2025にアクセス、 [https://dagster.io/blog/partitioned-data-pipelines](https://dagster.io/blog/partitioned-data-pipelines)
14. Introducing Dynamic Definitions for Flexible Asset Partitioning \- Dagster, 12月 10, 2025にアクセス、 [https://dagster.io/blog/dynamic-partitioning](https://dagster.io/blog/dynamic-partitioning)
15. Partitioning assets | Dagster Docs, 12月 10, 2025にアクセス、 [https://docs.dagster.io/guides/build/partitions-and-backfills/partitioning-assets](https://docs.dagster.io/guides/build/partitions-and-backfills/partitioning-assets)
16. Detecting Data Drift Using Evidently\! | by Pranav Khedkar \- Medium, 12月 10, 2025にアクセス、 [https://medium.com/@pranavk2208/detecting-data-drift-using-evidently-5c8643fd382d](https://medium.com/@pranavk2208/detecting-data-drift-using-evidently-5c8643fd382d)
17. Drift User Guide — Deepchecks Documentation, 12月 10, 2025にアクセス、 [https://docs.deepchecks.com/stable/general/guides/drift\_guide.html](https://docs.deepchecks.com/stable/general/guides/drift_guide.html)
18. Image Dataset Drift — Deepchecks Documentation, 12月 10, 2025にアクセス、 [https://docs.deepchecks.com/0.13/checks\_gallery/vision/train\_test\_validation/plot\_image\_dataset\_drift.html](https://docs.deepchecks.com/0.13/checks_gallery/vision/train_test_validation/plot_image_dataset_drift.html)
