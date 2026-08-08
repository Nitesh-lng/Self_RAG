from src.pipeline.graph import app
png = app.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png)

print("Graph saved as graph.png")