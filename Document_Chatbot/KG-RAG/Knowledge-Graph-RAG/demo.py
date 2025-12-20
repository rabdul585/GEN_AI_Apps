"""
Knowledge Graph vs Traditional RAG Demo

This script demonstrates the differences between Traditional RAG and Knowledge Graph-based RAG
using the CloudStore API documentation as sample data.
"""

import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

from traditional_rag import TraditionalRAG
from knowledge_graph import KnowledgeGraphRAG
from comparison import compare_systems, run_comparison_suite, plot_comparison_metrics, visualize_graph

console = Console()


# Sample questions that highlight KG advantages
DEMO_QUESTIONS = [
    "What is the purpose of bandwidth partitioning on a single physical interface?",
    "How do fixed circuits differ from burstable circuits?",
    "Why might rate limits be applied to a burstable circuit?",
    "If my 10G interface needs to support both production and backup workloads, how should I allocate circuits?",
    "A customer wants to avoid latency during traffic spikes. Which circuit type should they deploy?",
    "Can IP Transit and DCI traffic run on the same physical port without conflict? Why?",
    "Does Flexential Fabric require virtual partitioning when using multiple services on one interface?",
    "How can bandwidth allocation ensure financial cost control for burstable circuits?",

    "What is the main difference between IPv4 and IPv6 addressing formats?",
    "Why was IPv6 created?",
    "List the advantages IPv6 has over IPv4.",
    "Explain why dual-stack is recommended for IP bandwidth services.",
    "Less than 1% of networks use IPv6. Should I choose IPv6 for my environment? What should I verify first?",

    "What uptime SLA is provided for services on a single interface?",
    "What uptime SLA is provided when redundant interfaces are used?",
    "Where can I find the official SLA terms and conditions?",

    "How does Flexential calculate overages for IP Transit services?",
    "What data is polled every 5 minutes for percentile calculations?",
    "Which value—ingress or egress—is used when determining overages?",
    "Why does DCI and Interconnection Mesh calculate only ingress 95th percentile?",
    "How does excluding egress help prevent double billing?",
    "If my ingress 95th percentile exceeds my committed bandwidth, how will billing be handled?",

    "What is the purpose of the WAN Transit assignment block?",
    "Why does Flexential typically assign a /31 WAN block by default?",
    "Explain the difference between WAN assignments for /31, /30, and /29 blocks.",
    "Why does a /31 not have network or broadcast addresses?",
    "How are IPs assigned in a /29 block for FHRP configurations?",
    "Where is the Virtual IP (VIP) located in a /29 WAN assignment?",

    "If a customer wants 100% uptime and wants to run IP Transit + DCI, what interface configuration should they choose?",
    "How does bandwidth partitioning interact with WAN IP block assignments?",
    "What considerations should a customer make regarding IPv6 adoption when using dual-stack with WAN /29 blocks?",
    "How can overage billing be minimized when using burstable circuits on a shared 10G interface?",

    "Why am I receiving overage charges even though my usage seems normal?",
    "Can I run both IPv4 and IPv6 on the same IP Transit service?",
    "My equipment doesn’t support /31 WAN blocks. What are my options?",
    "Is it safe to run DCI and IP Transit together on one port without performance impact?",
    "What happens if my backup circuit exceeds the rate limit?"
]



def setup_environment():
    """Load and validate environment variables."""
    load_dotenv()

    required_vars = [
        "OPENAI_API_KEY",
        "NEO4J_URI",
        "NEO4J_USERNAME",
        "NEO4J_PASSWORD"
    ]

    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        console.print(f"[bold red]Error: Missing required environment variables:[/bold red]")
        for var in missing_vars:
            console.print(f"  - {var}")
        console.print("\n[yellow]Please create a .env file based on .env.example[/yellow]")
        return False

    return True


async def initialize_systems():
    """Initialize both RAG systems."""
    console.print("\n[bold cyan]Initializing Systems...[/bold cyan]\n")

    # Load configuration
    openai_api_key = os.getenv("OPENAI_API_KEY")
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_username = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    model_name = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    # Initialize Traditional RAG
    console.print("[yellow]1. Initializing Traditional RAG...[/yellow]")
    rag_system = TraditionalRAG(
        openai_api_key=openai_api_key,
        model_name=model_name,
        embedding_model=embedding_model
    )

    # Load and index documents
    doc_path = Path("sample_data/user_guide.txt")
    if not doc_path.exists():
        console.print(f"[bold red]Error: Sample data not found at {doc_path}[/bold red]")
        return None, None

    documents = rag_system.load_documents(str(doc_path))
    rag_system.build_index(documents)
    console.print("[green][OK] Traditional RAG initialized[/green]\n")

    # Initialize Knowledge Graph RAG
    console.print("[yellow]2. Initializing Knowledge Graph RAG...[/yellow]")
    kg_system = KnowledgeGraphRAG(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_username,
        neo4j_password=neo4j_password,
        openai_api_key=openai_api_key,
        model_name=model_name
    )

    # Build required Neo4j indexes and constraints
    await kg_system.graphiti.build_indices_and_constraints()

    # Check if we should rebuild the graph
    stats = kg_system.get_graph_statistics()
    if stats['total_nodes'] > 0:
        console.print(f"[yellow]Found existing graph with {stats['total_nodes']} nodes[/yellow]")
        rebuild = Confirm.ask("Do you want to rebuild the knowledge graph?", default=False)
        if rebuild:
            kg_system.clear_graph()
            stats = kg_system.get_graph_statistics()

    # Build knowledge graph if needed
    if stats['total_nodes'] == 0:
        console.print("[yellow]Building knowledge graph (this may take a few minutes)...[/yellow]")
        # Split documents for KG
        doc_texts = [doc.page_content for doc in documents]
        await kg_system.add_documents_to_graph(doc_texts, source="user_guide")

        stats = kg_system.get_graph_statistics()
        console.print(f"[green][OK] Knowledge Graph initialized[/green]")
        console.print(f"  - Nodes: {stats['total_nodes']}")
        console.print(f"  - Relationships: {stats['total_relationships']}")
        console.print(f"  - Entities: {stats['num_entities']}")
        console.print(f"  - Episodes: {stats['num_episodes']}\n")
    else:
        console.print(f"[green][OK] Using existing Knowledge Graph[/green]")
        console.print(f"  - Nodes: {stats['total_nodes']}")
        console.print(f"  - Relationships: {stats['total_relationships']}")
        console.print(f"  - Entities: {stats['num_entities']}")
        console.print(f"  - Episodes: {stats['num_episodes']}\n")

    return rag_system, kg_system


async def run_single_comparison(rag_system, kg_system):
    """Run a single question comparison."""
    console.print("\n[bold cyan]Single Question Comparison[/bold cyan]\n")

    # Show available demo questions
    console.print("[yellow]Suggested questions:[/yellow]")
    for i, q in enumerate(DEMO_QUESTIONS, 1):
        console.print(f"  {i}. {q}")

    console.print("\n[yellow]Enter a question number (1-7) or type your own question:[/yellow]")
    user_input = Prompt.ask("Question")

    # Parse input
    if user_input.isdigit() and 1 <= int(user_input) <= len(DEMO_QUESTIONS):
        question = DEMO_QUESTIONS[int(user_input) - 1]
    else:
        question = user_input

    # Run comparison
    await compare_systems(rag_system, kg_system, question, verbose=True)


async def run_full_comparison_suite(rag_system, kg_system):
    """Run the full comparison suite with all demo questions."""
    console.print("\n[bold cyan]Running Full Comparison Suite[/bold cyan]\n")
    console.print(f"This will test both systems with {len(DEMO_QUESTIONS)} predefined questions.\n")

    confirm = Confirm.ask("Continue?", default=True)
    if not confirm:
        return

    # Run suite
    results = await run_comparison_suite(rag_system, kg_system, DEMO_QUESTIONS)

    # Generate visualizations
    console.print("\n[yellow]Generating comparison visualizations...[/yellow]")
    plot_comparison_metrics(results, "comparison_metrics.png")
    console.print("[green][OK] Metrics plot saved to: comparison_metrics.png[/green]")


def visualize_knowledge_graph(kg_system):
    """Generate knowledge graph visualization."""
    console.print("\n[bold cyan]Generating Knowledge Graph Visualization[/bold cyan]\n")

    visualize_graph(
        neo4j_uri=os.getenv("NEO4J_URI"),
        neo4j_user=os.getenv("NEO4J_USERNAME"),
        neo4j_password=os.getenv("NEO4J_PASSWORD"),
        output_file="knowledge_graph.html",
        max_nodes=100
    )

    console.print("[green][OK] Visualization saved to: knowledge_graph.html[/green]")
    console.print("[yellow]Open this file in a web browser to explore the graph interactively[/yellow]")


async def interactive_mode(rag_system, kg_system):
    """Run interactive question-answering mode."""
    console.print("\n[bold cyan]Interactive Mode[/bold cyan]\n")
    console.print("[yellow]Ask questions about the CloudStore API (type 'exit' to quit)[/yellow]\n")

    while True:
        question = Prompt.ask("\n[bold]Your question[/bold]")

        if question.lower() in ['exit', 'quit', 'q']:
            break

        await compare_systems(rag_system, kg_system, question, verbose=True)


async def main():
    """Main demo function."""
    console.print(Panel.fit(
        "[bold green]Knowledge Graph vs Traditional RAG Demo[/bold green]\n"
        "Demonstrating the advantages of Knowledge Graph-based RAG over Traditional RAG",
        border_style="green"
    ))

    # Setup
    if not setup_environment():
        return

    # Initialize systems
    rag_system, kg_system = await initialize_systems()
    if not rag_system or not kg_system:
        return

    # Main menu
    while True:
        console.print("\n" + "=" * 80)
        console.print("[bold cyan]Demo Menu[/bold cyan]")
        console.print("=" * 80)
        console.print("1. Run single question comparison")
        console.print("2. Run full comparison suite (all demo questions)")
        console.print("3. Visualize knowledge graph")
        console.print("4. Interactive mode (ask your own questions)")
        console.print("5. View graph statistics")
        console.print("6. Exit")

        choice = Prompt.ask("\n[bold]Select an option[/bold]", choices=["1", "2", "3", "4", "5", "6"])

        if choice == "1":
            await run_single_comparison(rag_system, kg_system)
        elif choice == "2":
            await run_full_comparison_suite(rag_system, kg_system)
        elif choice == "3":
            visualize_knowledge_graph(kg_system)
        elif choice == "4":
            await interactive_mode(rag_system, kg_system)
        elif choice == "5":
            stats = kg_system.get_graph_statistics()
            console.print("\n[bold cyan]Knowledge Graph Statistics:[/bold cyan]")
            console.print(f"  - Total Nodes: {stats['total_nodes']}")
            console.print(f"  - Total Relationships: {stats['total_relationships']}")
            console.print(f"  - Entities: {stats['num_entities']}")
            console.print(f"  - Episodes: {stats['num_episodes']}")
        elif choice == "6":
            console.print("\n[bold green]Thank you for using the demo![/bold green]")
            kg_system.close()
            break


if __name__ == "__main__":
    asyncio.run(main())
