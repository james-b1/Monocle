---
name: learn
description: After implementing a code step or explaining a concept in the Monocle project, provide relevant official documentation, blog posts, and learning resources. Use this after every code block or implementation step — it is required by CLAUDE.md. Fetch live docs from context7 when available.
allowed-tools: Read, Grep, Glob
---

# Learning Companion

You just wrote code or explained a concept. Now do this:

1. **Identify what was introduced** — list every library, API, language feature, or concept that appeared in the code or explanation (e.g., `ctypes`, ARM Neon intrinsics, `sentence-transformers`, LangGraph `StateGraph`, cosine similarity, etc.)

2. **For each item, provide:**
   - The official documentation link (from context7 if the library is in its index, otherwise the canonical source)
   - A 1-2 sentence description of *what it is* and *why it matters here*
   - One high-quality blog post, tutorial, or paper if it exists and is worth reading

3. **Flag any prerequisite concepts** James might need before this makes full sense — point to where he can pick those up first.

4. **Format as a "Further Reading" section** at the bottom of the response, structured like this:

---

## Further Reading

### [Library or Concept Name]
- **Docs**: [link]
- **Why it matters here**: one sentence
- **Also worth reading**: [blog post or tutorial title + link] — one sentence on what it covers

---

Keep it tight. Don't list every page on the internet — pick the 1-2 sources that would genuinely help James understand what he just built.
