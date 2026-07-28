"""Streamlit application for the Universal Text-to-SQL Agent.

Run with:
    streamlit run app.py

Environment variables (.env or shell):
    GROQ_API_KEY               – required
    DATABASE_URL               – optional (defaults to in-memory SQLite demo)
    GROQ_MODEL                 – optional (defaults to llama-3.3-70b-versatile)
    MAX_RETRIES                – optional (defaults to 3)
    ENABLE_METADATA_ENRICHMENT – optional (defaults to true)
    ENABLE_QUERY_MEMORY        – optional (defaults to true)
"""

from __future__ import annotations

import logging
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s – %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Universal Text-to-SQL Agent",
    page_icon="🤖",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Lazy imports (keep startup fast)
# ---------------------------------------------------------------------------
from universal_text2sql.bootstrap import AgentContext, bootstrap  # noqa: E402

# ---------------------------------------------------------------------------
# Session-level initialisation (cached)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Connecting · discovering schema · building knowledge graph …")
def _get_context(db_url: str, max_retries: int, self_consistency_samples: int) -> AgentContext:
    return bootstrap(
        database_url=db_url,
        max_retries=max_retries,
        self_consistency_samples=self_consistency_samples,
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def main() -> None:
    st.title("🤖 Universal Text-to-SQL Agent")
    st.caption(
        "Powered by **LangGraph** + **Groq** · Auto knowledge graph · Auto business glossary · "
        "Self-consistency · Self-reflection · RL-inspired query memory"
    )

    # ---- Sidebar: configuration ----
    with st.sidebar:
        st.header("⚙️ Configuration")

        api_key = st.text_input(
            "Groq API Key",
            value=os.getenv("GROQ_API_KEY", ""),
            type="password",
            help="Get your key at https://console.groq.com",
        )
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key

        db_url = st.text_input(
            "Database URL",
            value=os.getenv("DATABASE_URL", "sqlite:///:memory:"),
            help=(
                "Examples:\n"
                "- sqlite:///./mydb.sqlite3\n"
                "- postgresql://user:pw@host/db\n"
                "- mysql+pymysql://user:pw@host/db"
            ),
        )

        model = st.selectbox(
            "Groq Model",
            [
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "mixtral-8x7b-32768",
                "gemma2-9b-it",
            ],
            index=0,
        )
        os.environ["GROQ_MODEL"] = model

        max_retries = st.slider("Max Self-Reflection Retries", min_value=1, max_value=5, value=3)
        self_consistency_samples = st.slider(
            "Self-Consistency Candidates",
            min_value=1,
            max_value=5,
            value=1,
            help=(
                "Sample multiple SQL candidates for non-trivial questions and pick the "
                "majority-vote result (costs extra LLM calls; 1 = disabled)."
            ),
        )

        st.markdown("---")
        st.subheader("📊 Database Info")

    # Lazy-init resources (connect, discover schema, build KG, generate glossary)
    try:
        ctx = _get_context(db_url, max_retries, self_consistency_samples)
    except Exception as exc:
        st.error(f"Failed to connect to database: {exc}")
        st.stop()
    schema = ctx.schema

    # Show schema + auto-generated metadata in sidebar
    with st.sidebar:
        for tname, tmeta in schema.tables.items():
            with st.expander(f"📋 {tname} ({tmeta.row_count} rows)"):
                if tmeta.description:
                    st.caption(tmeta.description)
                col_info = [
                    {
                        "Column": c.name,
                        "Type": c.data_type,
                        "PK": "✓" if c.primary_key else "",
                        "Nullable": "✓" if c.nullable else "",
                        "Meaning": c.business_meaning,
                    }
                    for c in tmeta.columns
                ]
                st.dataframe(pd.DataFrame(col_info), use_container_width=True, hide_index=True)

        st.markdown("---")
        with st.expander("🕸️ Knowledge Graph"):
            st.graphviz_chart(ctx.knowledge_graph.to_graphviz())
            st.text(ctx.describe_knowledge_graph())
        with st.expander("📖 Auto-Generated Glossary"):
            st.text(ctx.describe_glossary())

    # ---- Main area: query input ----
    st.markdown("### 💬 Ask a question about your data")

    example_questions = [
        "How many customers are from the USA?",
        "What are the top 3 best-selling products by total revenue?",
        "Show me all orders placed in 2023 with their customer names.",
        "Which customer has spent the most money overall?",
        "What is the average order value by country?",
    ]

    col1, col2 = st.columns([4, 1])
    with col1:
        question = st.text_input(
            "Your question",
            placeholder="e.g. What are the top 5 products by revenue?",
            label_visibility="collapsed",
        )
    with col2:
        ask_btn = st.button("🚀 Ask", use_container_width=True)

    with st.expander("💡 Example questions"):
        for eq in example_questions:
            if st.button(eq, key=eq):
                question = eq
                ask_btn = True

    if not question:
        st.info("Enter a question above to get started.")
        return

    if not api_key:
        st.warning("Please enter your Groq API key in the sidebar.")
        return

    if ask_btn or question:
        with st.spinner("🤔 Thinking …"):
            try:
                result = ctx.ask(question)
            except Exception as exc:
                st.error(f"Agent error: {exc}")
                logger.exception("Agent error")
                return

        # ---- Results ----
        st.markdown("---")

        success = result.get("success", False)
        if success:
            st.success("✅ Query executed successfully")
        else:
            st.warning("⚠️ Query completed with issues")

        tab_answer, tab_sql, tab_data, tab_trace = st.tabs(
            ["💬 Answer", "🔍 SQL", "📊 Data", "🔄 Trace"]
        )

        with tab_answer:
            answer = result.get("final_answer", "No answer generated.")
            st.markdown(f"**{answer}**")

        with tab_sql:
            sql = result.get("generated_sql", "")
            st.code(sql, language="sql")
            if result.get("complexity"):
                st.caption(f"🧮 Classified complexity: {result['complexity']}")
            candidates = result.get("sql_candidates") or []
            if len(candidates) > 1:
                st.info(f"🗳️ Self-consistency: {len(candidates)} candidates sampled and voted on.")
            retries = result.get("retry_count", 0)
            if retries > 0:
                st.info(f"🔁 Self-reflection was used {retries} time(s) to correct the query.")

        with tab_data:
            df = result.get("execution_result")
            if df is not None and not df.empty:
                st.dataframe(df, use_container_width=True)
                st.caption(f"{len(df)} row(s) returned")
            elif result.get("execution_error"):
                st.error(f"Execution error: {result['execution_error']}")
            else:
                st.info("No data returned.")

        with tab_trace:
            messages = result.get("messages", [])
            for msg in messages:
                role = "🧠 Agent" if hasattr(msg, "content") else "👤 User"
                content = msg.content if hasattr(msg, "content") else str(msg)
                with st.chat_message("assistant" if "Agent" in role else "user"):
                    st.markdown(content)


if __name__ == "__main__":
    main()
