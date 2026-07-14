"""
Workflow visualization component — renders the LangGraph execution flow
as a visual diagram with active/inactive/completed node states.
"""

import streamlit as st


def render_workflow_diagram(events: list, step_count: int = 0) -> None:
    """Render the recruitment workflow diagram with node status.

    Nodes are color-coded based on execution state:
    - ⚪ Pending (not yet executed)
    - 🔵 Active (currently executing)
    - ✅ Completed (executed successfully)
    - ❌ Error (executed with error)

    Args:
        events: List of workflow events from graph.stream().
        step_count: Current step count from state.
    """
    # Define the workflow graph structure
    nodes = [
        {"id": "resume_analyst", "label": "📄 Resume Analyst", "description": "Parse resumes & detect injections"},
        {"id": "scorer", "label": "📊 Scorer", "description": "Score candidates vs JD"},
        {"id": "verifier", "label": "🔍 Verifier", "description": "Blind re-score borderline"},
        {"id": "decider", "label": "⚖️ Decider", "description": "Generate ranked shortlist"},
        {"id": "human_approval_gate", "label": "👤 Human Approval", "description": "Approve before scheduling"},
        {"id": "scheduler", "label": "📅 Scheduler", "description": "Generate interview invites"},
    ]

    # Determine which nodes have executed based on events
    executed_nodes = set()
    errored_nodes = set()
    for event in events:
        for node_name in event.keys():
            if node_name.startswith("resume_analyst"):
                executed_nodes.add("resume_analyst")
            elif node_name.startswith("scorer"):
                executed_nodes.add("scorer")
            elif node_name.startswith("verifier"):
                executed_nodes.add("verifier")
            elif node_name.startswith("decider"):
                executed_nodes.add("decider")
            elif node_name.startswith("human_approval"):
                executed_nodes.add("human_approval")
            elif node_name.startswith("scheduler"):
                executed_nodes.add("scheduler")
            # Check for errors
            node_output = event.get(node_name, {})
            if isinstance(node_output, dict) and node_output.get("errors"):
                errored_nodes.add(node_name)

    # Render as a vertical flow with arrows
    rendered_nodes = []
    for i, node in enumerate(nodes):
        nid = node["id"]

        if nid in errored_nodes:
            status_icon = "❌"
            status_color = "#ff4444"
        elif nid in executed_nodes:
            status_icon = "✅"
            status_color = "#00cc66"
        else:
            status_icon = "⏳"
            status_color = "#888888"

        rendered_nodes.append(f"""
        <div style="
            border: 2px solid {status_color};
            border-radius: 8px;
            padding: 8px 16px;
            margin: 4px 0;
            background-color: {'#f0fff0' if nid in executed_nodes and nid not in errored_nodes else '#fff0f0' if nid in errored_nodes else '#f8f8f8'};
            display: flex;
            align-items: center;
            gap: 10px;
        ">
            <span style="font-size: 1.2em;">{status_icon}</span>
            <div>
                <strong>{node['label']}</strong><br>
                <small style="color: #666;">{node['description']}</small>
            </div>
        </div>
        """)

        # Add arrow between nodes (not after the last one)
        if i < len(nodes) - 1:
            rendered_nodes.append("""
            <div style="text-align: center; color: #888; font-size: 0.8em; margin: -2px 0;">
                ▼
            </div>
            """)

    st.markdown("### 🔄 Workflow Execution Flow")
    st.markdown(f"**Step Count:** {step_count}")

    for html in rendered_nodes:
        st.markdown(html, unsafe_allow_html=True)

    # Summary stats
    if executed_nodes:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Executed Nodes", len(executed_nodes))
        with col2:
            st.metric("Errored Nodes", len(errored_nodes))