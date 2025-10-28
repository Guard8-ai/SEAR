# SEAR Claude Code Skill

This directory contains the Claude Code Skill for SEAR (Semantic Enhanced Augmented Retrieval).

## What is a Claude Code Skill?

Skills are modular capabilities that extend Claude's functionality in Claude Code. When you install the SEAR skill, Claude will automatically recognize when you need document search, RAG, or PDF/DOCX conversion capabilities and guide you through using SEAR effectively.

## Installation

### Prerequisites

1. **Claude Code** must be installed (VSCode extension or desktop app)
2. **SEAR** must be installed in your environment:
   ```bash
   cd /path/to/Summarization-EnhanceAugmentedRetrieval
   pip install -e ".[all]"
   ollama pull all-minilm qwen2.5:0.5b
   ```

### Install the Skill

Copy the skill directory to Claude Code's skills folder:

```bash
# Clone or download this repository first
git clone https://github.com/Guard8-ai/SEAR.git

# Copy the skill to Claude Code
cp -r SEAR/claude-skill ~/.claude/skills/sear
```

**Alternative locations:**
- **Windows:** `%USERPROFILE%\.claude\skills\sear`
- **macOS/Linux:** `~/.claude/skills/sear`

### Verify Installation

1. Open Claude Code
2. Start a conversation and ask: "Can you help me search my documents semantically?"
3. Claude should recognize the SEAR skill and guide you through the workflow

## What This Skill Does

When installed, Claude will automatically:

✅ **Recognize document search needs** - When you mention PDF/DOCX conversion, semantic search, or RAG
✅ **Guide you through SEAR workflows** - Convert → Index → Search → Extract
✅ **Provide command examples** - With proper syntax and options
✅ **Suggest optimizations** - GPU usage, quality thresholds, multi-corpus strategies
✅ **Troubleshoot issues** - Help with installation, configuration, and common errors

## Example Interactions

### Before Installing Skill:
```
You: "I need to search through 100 PDF research papers"
Claude: "You could use various tools for PDF processing..."
```

### After Installing Skill:
```
You: "I need to search through 100 PDF research papers"
Claude: "I'll use SEAR to help you with that! Let's start by converting
        the PDFs to searchable markdown using the doc-converter:

        sear convert paper1.pdf

        Then we'll index them and enable semantic search..."
```

## Skill Features

### Automatic Invocation
Claude detects when SEAR is relevant based on keywords:
- "semantic search"
- "RAG" or "retrieval augmented generation"
- "PDF conversion" or "DOCX to markdown"
- "index documents"
- "search corpus"

### Guided Workflows
The skill teaches Claude about SEAR's complete pipeline:
1. Document conversion (PDF/DOCX → Markdown with OCR)
2. Indexing (embedding + FAISS index creation)
3. Semantic search (with LLM synthesis)
4. Content extraction (pure retrieval without LLM)

### Best Practices
Claude learns SEAR's recommended practices:
- GPU vs CPU selection based on corpus size
- Quality threshold tuning
- Multi-corpus strategies
- LLM provider selection (Ollama vs Anthropic)

## Skill Structure

```
claude-skill/
├── SKILL.md              # Main skill definition (Claude reads this)
├── README.md             # This file (installation instructions)
└── examples/             # Concrete usage examples
    ├── basic-workflow.md
    ├── pdf-conversion.md
    └── multi-corpus-search.md
```

## How Skills Work

**Model-Invoked (Automatic):**
- Skills are invoked by Claude automatically based on context
- You don't need to manually activate them (unlike slash commands)
- Claude reads the skill when relevant to your request

**Progressive Disclosure:**
- Only ~50 tokens overhead when skill is available
- Full details loaded only when Claude needs them
- Minimal impact on context window

**Token Efficient:**
- Skills don't bloat your context
- Information loaded on-demand
- Examples and details only when needed

## Updating the Skill

When SEAR is updated with new features:

```bash
# Pull latest changes
cd SEAR
git pull

# Re-copy the skill
cp -r claude-skill ~/.claude/skills/sear
```

Claude Code will automatically use the updated skill in new conversations.

## Uninstalling

Remove the skill directory:

```bash
rm -rf ~/.claude/skills/sear
```

## Troubleshooting

### Skill Not Recognized

**Issue:** Claude doesn't seem to know about SEAR

**Solutions:**
1. Verify installation path: `ls ~/.claude/skills/sear/SKILL.md`
2. Restart Claude Code
3. Try explicitly mentioning: "I want to use SEAR for document search"

### SEAR Commands Failing

**Issue:** Claude suggests SEAR commands but they fail

**Solutions:**
1. Ensure SEAR is installed: `pip show sear` or `sear --help`
2. Install required extras: `pip install -e ".[all]"`
3. Check Ollama models: `ollama list` (should show all-minilm and qwen2.5:0.5b)

### Wrong Command Syntax

**Issue:** Claude provides outdated syntax

**Solutions:**
1. Update the skill (see "Updating the Skill" above)
2. Check SEAR version: Compare with latest on GitHub
3. Manually correct and inform Claude: "Actually, the syntax is..."

## Contributing

Found an issue or want to improve the skill?

1. Submit issues on GitHub: https://github.com/Guard8-ai/SEAR/issues
2. Suggest improvements to SKILL.md
3. Add examples to examples/ directory

## Learn More

- **SEAR Repository:** https://github.com/Guard8-ai/SEAR
- **Installation Guide:** [INSTALL.md](../INSTALL.md)
- **GPU Support:** [GPU_SUPPORT.md](../GPU_SUPPORT.md)
- **Benchmarks:** [BENCHMARK_RESULTS.md](../BENCHMARK_RESULTS.md)
- **Claude Skills Docs:** https://docs.claude.com/en/docs/claude-code/skills

## License

MIT License with Commons Clause - Same as SEAR main project

Copyright (c) 2025 Guard8.ai
