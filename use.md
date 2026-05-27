# Research Wiki 系统使用手册

## 这是什么

一个基于 LLM 的研究知识库系统。你把论文扔进来，它会帮你提取关键信息、建立概念索引、发现研究空白、生成可验证的假设。所有操作通过和 Claude Code 对话完成，不需要手动整理笔记。

**核心思路**：raw/ 放原始资料（只读），wiki/ 放策展过的知识（链接、综合、思考）。你只动手扔论文进来，剩下的结构化整理由 AI 完成。

---

## 快速开始

### 第一步：把论文放进来

将论文的 markdown 文件（推荐用 MinerU 将 PDF 转为 md）放到 `raw/papers/` 目录下：

```
raw/papers/2501.12345.md   # 以 arXiv ID 命名
```

如果只有非论文的零散资料（arXiv 摘要、推特长截图等），放到 `raw/notes/`；论文中的图表放到 `raw/assets/`。

### 第二步：摄入论文

直接对 Claude 说：

> "ingest this" 或 "摄入 raw/papers/2501.12345.md"

系统会自动：
1. 提取论文核心主张（一句话）
2. 提取 2-3 条关键结论
3. 识别涉及的概念，检查是否已有记录
4. 在 `wiki/papers/` 生成结构化笔记
5. 更新或新建 `wiki/concepts/` 条目
6. 将开放问题写入 `wiki/gaps/questions.md`
7. 更新 `wiki/index.md` 和 `wiki/log.md`

你什么都不用管，等结果就行。

### 第三步：查询知识库

想了解某个主题时，直接问：

> "what does the wiki say about FGVR?" 或 "wiki 里关于层级视觉识别有什么记录？"

系统会：
- 搜索 `wiki/` 下的相关论文、概念、实体
- 如果合适，自动生成 `wiki/comparisons/` 对比条目
- 指出已知内容和未知空白

---

## 五个核心操作

### 1. Ingest（摄入）— 读新论文

**触发方式**：给 Claude 一篇 raw 论文，说 "ingest this"

**产出物**：
- `wiki/papers/[id]-[short-title].md` — 结构化笔记
- 更新/新建 `wiki/concepts/` 条目
- `wiki/gaps/questions.md` 新增开放问题
- 更新 `wiki/index.md` 和 `wiki/log.md`

**多久用一次**：每读完一篇论文就用一次。不要攒。

---

### 2. Query（查询）— 理解已有知识

**触发方式**：问 "what does the wiki say about X?"

**适用场景**：
- 写 related work 前，快速了解 wiki 对某方向的覆盖
- 比较两个方法的优劣
- 检查某个概念是否已有记录

---

### 3. Idea（生成想法）— 从已有知识推导新假设

**触发方式**：说 "generate ideas"

**工作流程**：
1. 读取 `wiki/synthesis/shared-assumptions.md`（领域共识假设）
2. 读取 `wiki/gaps/confirmed-gaps.md`（公认未解决的问题）
3. 交叉比对：哪些假设如果错了，就能打开哪些问题的解法？
4. 生成 2-3 条可验证假设，写入 `wiki/gaps/hypotheses.md`

**每条假设必须满足**：如果 X 成立，那么 Y 应该能被观测到。

**多久用一次**：不要每篇论文后都跑。攒够 3-5 篇新论文，或发现某个假设特别有意思时再跑。

---

### 4. Lint（健康检查）— 维基维护

**触发方式**：说 "lint the wiki"

**检查项目**：
- 孤立概念（没有论文链接到它）
- 超过一个月仍为 draft 状态的假设
- 被 3 篇以上论文提及但仍无独立条目的概念
- `overview.md` 和 `field-map.md` 是否最新
- 所有 wikilink 是否指向存在的文件

**多久用一次**：约每两周一次。不要频繁跑。

---

### 5. Shared Research（协作讨论）— 深度思考

**触发方式**：给出一个研究方向或问题，比如 "我们来讨论一下 TARA 和 Fine-R1 能不能结合"

**工作流程**：
1. Claude 读取当前 wiki 状态
2. 以对话形式展开讨论
3. 讨论结果保存到 `wiki/synthesis/discussion-[日期].md`
4. 关键发现记录到 `wiki/log.md`

**适用场景**：你想深入思考一个方向，需要有人和你对话、追问、质疑。

---

## 文件地图

```
raw/                          # 原始资料 — 只读，不要手动编辑
├── papers/[arxiv-id].md      # PDF → Markdown（MinerU 转换）
├── notes/                    # arXiv 摘要、推特长截图等
└── assets/                   # 论文图表

wiki/                         # 策展知识 — 这里发生思考
├── index.md                  # 入口：有什么、为什么
├── log.md                    # 所有重要操作的时间戳日志
├── overview.md               # 领域一页纸综述
├── papers/[id]-short-title.md   # 结构化论文笔记
├── concepts/[method-name].md    # 核心方法/概念（用你自己的话解释）
├── entities/[name].md           # 基准、数据集、组织、关键人物
├── comparisons/[topic]-comparison.md  # 方法头对头对比
├── gaps/                     # 研究前沿
│   ├── confirmed-gaps.md     # 领域公认未解决的问题（3+论文确认）
│   ├── hypotheses.md         # 你的可验证假设（draft → testing → confirmed/rejected）
│   └── questions.md          # 阅读中产生的开放问题
└── synthesis/                # 高阶思考
    ├── field-map.md          # 子领域关系图
    ├── shared-assumptions.md # 领域共识（可能被推翻的前提）
    └── discussion-[date].md  # 讨论/思考会话记录
```

---

## 核心约定

### Wikilink
用 `[[path/filename]]` 链接 wiki 条目。这是知识网络的连接组织。例如：
- 论文笔记里写 `[[concepts/TAPO]]` 链接到概念页
- 概念页里写 `[[papers/Fine-R1-fine-grained-recognition]]` 链回论文

### 置信度标签
每篇论文和假设都要标注置信度：`high` / `medium` / `low`。诚实标注。

### 假设生命周期
`draft` → `testing`（你正在跑实验验证）→ `confirmed` 或 `rejected`

### 记录一切
每次摄入、每次想法生成、每次讨论 → 都追加到 `wiki/log.md`

### Raw 和 Wiki 严格分离
`wiki/` 目录只放经过处理、链接、综合的笔记。原始转储留在 `raw/`。

---

## 典型工作流

### 场景 A：日常读论文

```
1. 下载论文 PDF → MinerU 转 md → 放到 raw/papers/
2. 对 Claude 说 "ingest this"
3. 看生成的结构化笔记，确认理解正确
4. 如果有新想法，追加到 questions.md
```

### 场景 B：准备写 Related Work

```
1. 问 Claude "我们这个方向目前 wiki 里有哪些论文和概念"
2. 让 Claude 生成对比表（触发 comparisons/ 创建）
3. 确认空白（哪些重要论文还没摄入）
```

### 场景 C：找研究想法

```
1. 摄入 3-5 篇新论文后，说 "generate ideas"
2. 审阅生成的假设，挑最有意思的
3. 把想验证的假设状态改为 testing
4. 跑实验后更新为 confirmed 或 rejected
```

### 场景 D：深度思考某个方向

```
1. 对 Claude 说 "我们来讨论一下 [你的问题]"
2. 进入对话模式，让 Claude 追问和质疑
3. 讨论记录自动保存到 synthesis/discussion-[日期].md
```

---

## 常见问题

**Q: 论文太多，raw/ 和 wiki/ 会不会混乱？**
A: 不会。raw/ 只放原始文件（按 arXiv ID 命名），wiki/ 只放策展笔记。两边命名规则不同，不会冲突。

**Q: 概念条目什么时候该新建？**
A: 当某个方法/技术被至少一篇论文作为核心贡献提出时。如果只是一个论文里顺带提的，不需要单独建。

**Q: confirmed-gaps 和 questions 有什么区别？**
A: questions 是你个人在阅读中产生的疑问（可以很随意）。confirmed-gaps 需要 3 篇以上论文共同承认该问题未解决（是领域共识）。

**Q: 可以多人协作吗？**
A: 系统设计为 git 友好（所有文件都是 markdown）。多人通过 git 协作即可，注意 wiki/log.md 可能会有冲突。
