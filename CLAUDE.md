# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
# Install dependencies
uv sync

# Install development dependencies (code quality tools)
uv sync --group dev

# Set up environment variables (required)
# Create .env file with: ANTHROPIC_API_KEY=your_key_here
```

### Running the Application
```bash
# Quick start (recommended)
./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Development Server
- Web Interface: http://localhost:8000
- API Documentation: http://localhost:8000/docs

### Code Quality Tools
```bash
# Format code with Black and sort imports
./scripts/format.sh

# Run all quality checks (linting, type checking)
./scripts/lint.sh

# Format and check code in one command
./scripts/check.sh

# Manual commands (if needed)
uv run --group dev black .          # Format code
uv run --group dev isort .          # Sort imports
uv run --group dev flake8 .         # Lint code
uv run --group dev mypy .           # Type check
```

## Best Practices
- Always use uv to run the server, do not use pip directly
- Make sure to use uv to manage all dependencies
- Run `./scripts/format.sh` before committing code changes
- Use `./scripts/check.sh` to ensure code quality before pushing

## Architecture Overview

This is a **Retrieval-Augmented Generation (RAG) chatbot** that answers questions about course materials using semantic search and Claude AI.

### Core Architecture Pattern
The system follows a **tool-based RAG architecture** where Claude intelligently decides when to search course content:

1. **Smart Query Routing**: Claude analyzes queries and only searches when course-specific content is needed
2. **Semantic Search**: ChromaDB with sentence transformers for vector similarity
3. **Context Synthesis**: Claude combines search results with conversation history and built-in knowledge
4. **Session Management**: Maintains conversation context across interactions

### Key Components Flow

**Document Processing Pipeline** (`document_processor.py`):
```
Course Files → Structure Parsing → Text Chunking → Vector Storage
```
- Parses structured course documents (title, instructor, lessons)
- Chunks text with sentence-based splitting and configurable overlap
- Adds contextual prefixes: "Course [title] Lesson [#] content: [chunk]"

**Query Processing Chain** (`rag_system.py`):
```
User Query → Session Context → Claude + Tools → Vector Search (if needed) → Response
```

**Vector Storage** (`vector_store.py`):
- ChromaDB with separate collections for course metadata and content chunks
- Uses all-MiniLM-L6-v2 sentence transformer embeddings
- Stores structured metadata (course_title, lesson_number, chunk_index)

### Data Models

**Core entities** (`models.py`):
- `Course`: Container with title, instructor, lessons list
- `Lesson`: Individual lesson with number, title, optional link
- `CourseChunk`: Vector storage unit with content, metadata, and position tracking

### Configuration System

**Environment-based config** (`config.py`):
- API keys loaded from `.env` file
- Chunk size (800), overlap (100), max results (5) configurable
- Claude model: claude-sonnet-4-20250514
- Embedding model: all-MiniLM-L6-v2

### Tool System Architecture

**Dynamic tool registration** (`search_tools.py`):
- Abstract `Tool` base class for extensibility
- `CourseSearchTool` provides semantic search with course/lesson filtering
- `ToolManager` handles tool registration and execution
- Claude receives tool definitions and makes autonomous search decisions

### Document Format Expectations

Course files should follow this structure:
```
Course Title: [title]
Course Link: [optional_url]
Course Instructor: [instructor]

Lesson 0: Introduction
Lesson Link: [optional_url]
[lesson content]

Lesson 1: Next Topic
[lesson content]
```

### Frontend Integration

**Vanilla JavaScript SPA** (`frontend/`):
- Fetch API calls to `/api/query` and `/api/courses` endpoints
- Markdown rendering with marked.js library
- Session persistence and loading states
- Real-time course statistics display

### Database Persistence

**ChromaDB storage**:
- Persistent database in `./chroma_db/` directory
- Automatic document loading from `docs/` folder on startup
- Duplicate prevention based on course titles
- Vector embeddings generated on-the-fly

### Key Integration Points

When modifying this system:

1. **Adding new tools**: Extend `Tool` abstract class and register with `ToolManager`
2. **Document processing**: Modify `DocumentProcessor.chunk_text()` for different text splitting strategies
3. **Search ranking**: Adjust similarity thresholds in `VectorStore.search_course_content()`
4. **AI behavior**: Update system prompt in `AIGenerator.SYSTEM_PROMPT` for different response styles
5. **Configuration**: Add new settings to `Config` dataclass with environment variable support

### Session and State Management

- Sessions created per conversation with unique IDs
- Conversation history limited by `MAX_HISTORY` setting (default: 2 messages)
- ChromaDB handles persistence automatically
- No user authentication - single-user application design